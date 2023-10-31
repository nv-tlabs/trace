import numpy as np

from tbsim.envs.base import BatchedEnv
import tbsim.utils.tensor_utils as TensorUtils

from tbsim.utils.geometry_utils import batch_nd_transform_points_np
from tbsim.utils.guidance_metrics import guidance_metrics_from_config

def guided_rollout(
    env,
    policy,
    policy_model,
    n_step_action=1,
    guidance_config=None,
    scene_indices=None,
    device=None,
    obs_to_torch=True,
    horizon=None,
    use_gt=False,
    start_frames=None,
):
    """
    Rollout an environment.
    Args:
        env (BaseEnv): a base simulation environment (gym-like)
        policy (RolloutWrapper): a policy that controls agents in the environment
        policy_model (LightningModule): the traffic model underlying the policy with set_guidance implemented.
        n_step_action (int): number of steps to take between querying models
        guidance_config: which guidance functions to use
        scene_indices (tuple, list): (Optional) scenes indices to rollout with
        device: device to cast observation to
        obs_to_torch: whether to cast observation to torch
        horizon (int): (Optional) override horizon of the simulation
        use_gt (bool) : whether the given policy is returning GT or not.
        start_frames (list) : (Optional) a list of starting frame index for each scene index.

    Returns:
        stats (dict): A dictionary of rollout stats for each episode (metrics, rewards, etc.)
        info (dict): A dictionary of environment info for each episode
    """
    stats = {}
    info = {}
    is_batched_env = isinstance(env, BatchedEnv)

    # set up guidance and associated metrics
    added_metrics = [] # save for removal later
    if guidance_config is not None:
        # reset so that we can get an example batch to initialize guidance more efficiently
        env.reset(scene_indices=scene_indices, start_frame_index=start_frames)
        ex_obs = env.get_observation()
        if obs_to_torch:
            device = policy.device if device is None else device
            ex_obs = TensorUtils.to_torch(ex_obs, device=device, ignore_if_unspecified=True)
        if not use_gt:
            policy_model.set_guidance(guidance_config, ex_obs['agents'])
        guidance_metrics = guidance_metrics_from_config(guidance_config)
        env._metrics.update(guidance_metrics)  
        added_metrics += guidance_metrics.keys()

    # metrics are reset here too, so have to run again after adding new metrics
    env.reset(scene_indices=scene_indices, start_frame_index=start_frames)

    done = env.is_done()
    counter = 0
    while not done:
        obs = env.get_observation()
        if obs_to_torch:
            device = policy.device if device is None else device
            obs_torch = TensorUtils.to_torch(obs, device=device, ignore_if_unspecified=True)
        else:
            obs_torch = obs
        action = policy.get_action(obs_torch, step_index=counter)

        env.step(action, num_steps_to_take=n_step_action, render=False) 
        counter += n_step_action

        done = env.is_done()
        
        if horizon is not None and counter >= horizon:
            break

    metrics = env.get_metrics()

    for k, v in metrics.items():
        if k not in stats:
            stats[k] = []
        if is_batched_env:  # concatenate by scene
            stats[k] = np.concatenate([stats[k], v], axis=0)
        else:
            stats[k].append(v)

    # remove all temporary added metrics
    for met_name in added_metrics:
        env._metrics.pop(met_name)

    if not use_gt:
        # and undo guidance setting
        policy_model.clear_guidance()

    env_info = env.get_info()
    for k, v in env_info.items():
        if k not in info:
            info[k] = []
        if is_batched_env:
            info[k].extend(v)
        else:
            info[k].append(v)

    env.reset_multi_episodes_metrics()

    return stats, info

################## HEURISTIC CONFIG UTILS #########################

from copy import deepcopy

def merge_guidance_configs(cfg1, cfg2):
    if cfg1 is None or len(cfg1) == 0:
        return cfg2
    if cfg2 is None or len(cfg2) == 0:
        return cfg1
    merge_cfg = deepcopy(cfg1)
    num_scenes = len(merge_cfg)
    for si in range(num_scenes):
        merge_cfg[si].extend(cfg2[si])
    return merge_cfg


def get_agents_future(sim_scene, fut_sec):
    '''
    Queries the sim scene for the future state traj (in global frame) of all agents.
    - sim_scene : to query
    - fut_sec : how far in the future (in sec) to get

    Returns:
    - agents_future: (N x T x 7) [pos, vel, acc, heading_angle]
    - fut_valid : (N x T) whether states are non-nan at each step
    '''
    agents_future, _, _ = sim_scene.cache.get_agents_future(sim_scene.init_scene_ts, sim_scene.agents, (fut_sec, fut_sec))
    max_fut_steps = max([fut.shape[0] for fut in agents_future])
    for fi, fut in enumerate(agents_future):
        if fut.shape[0] < max_fut_steps:
            # pad
            pad_len = max_fut_steps - fut.shape[0]
            padding = np.ones((pad_len, 7)) * np.nan
            agents_future[fi] = np.concatenate([fut, padding], axis=0)
    agents_future = np.stack(agents_future, axis=0)
    fut_valid = np.sum(np.logical_not(np.isnan(agents_future)), axis=-1) == 7
    return agents_future, fut_valid

def get_agents_curr(sim_scene):
    '''
    Queries the sim scene for current state of all agents

    returns:
    - curr_agent_state: (N x 7) [pos, vel, acc, heading_angle]
    '''
    curr_agent_state = sim_scene.cache.get_states(
            [sim_scene.agents[i].name for i in range(len(sim_scene.agents))],
            sim_scene.init_scene_ts
    )
    curr_agent_state = np.stack(curr_agent_state, axis=0)
    return curr_agent_state

def get_agent_from_world_tf(curr_state):
    pos = curr_state[:,:2]
    h = curr_state[:,-1:]
    hx, hy = np.cos(h), np.sin(h)

    last_row = np.zeros((pos.shape[0], 3))
    last_row[:,2] = 1.0
    world_from_agent_tf = np.stack([
                                np.concatenate([hx, -hy, pos[:,0:1]], axis=-1),
                                np.concatenate([hy,  hx, pos[:,1:2]], axis=-1),
                                last_row,
                                ], axis=-2)
    agent_from_world_tf = np.linalg.inv(world_from_agent_tf)

    return agent_from_world_tf

def heuristic_social_group(sim_scene, dt, group_dist_thresh, social_dist, cohesion):
    from scipy.spatial.distance import pdist, squareform
    from scipy.sparse.csgraph import csgraph_from_dense, connected_components
    import random

    curr_state = get_agents_curr(sim_scene)
    cur_pos = curr_state[:,:2]
    cur_vel = curr_state[:,2:4]

    # create graph with edges based on given distance threshold
    #   and direction of movement.
    #   create social groups from connected components
    not_moving = np.linalg.norm(cur_vel, axis=-1) < 0.9
    dir = cur_vel / (np.linalg.norm(cur_vel, axis=-1, keepdims=True) + 1e-6)
    cos_sim = np.sum(dir[:,np.newaxis] * dir[np.newaxis,:], axis=-1)
    move_sim = cos_sim >= 0
    move_sim[not_moving] = True # if they're not moving, don't care about direction
    move_sim[:,not_moving] = True
    # now distance
    dist = squareform(pdist(cur_pos))
    graph = np.logical_and(dist <= group_dist_thresh, move_sim)
    np.fill_diagonal(graph, 0)
    graph = graph.astype(int)
    graph = csgraph_from_dense(graph)

    n_comp, labels = connected_components(graph, directed=False)
    config_list = []
    for ci in range(n_comp):
        comp_mask = labels == ci
        comp_size = np.sum(comp_mask)
        # only want groups, not single agents
        if comp_size > 1:
            group_inds = np.nonzero(comp_mask)[0].tolist()
            # randomly sample leader
            leader = random.sample(group_inds, 1)[0]
            guide_config = {
                'name' : 'social_group',
                'params' : {
                            'leader_idx' : leader,
                            'social_dist' : social_dist,
                            'cohesion' : cohesion,    
                           },
                'agents' : group_inds
            }
            config_list.append(guide_config)

    if len(config_list) > 0:
        return config_list
    return None

def heuristic_global_target_pos_at_time(sim_scene, dt, target_time, urgency, pref_speed, perturb_std=None):
    '''
    Sets a global target pos and time using GT.
    '''
    fut_sec = target_time * dt
    fut_traj, fut_valid = get_agents_future(sim_scene, fut_sec)
    fut_pos = fut_traj[:,:,:2]
    agents = np.arange(fut_pos.shape[0])

    valid_agts = np.sum(fut_valid, axis=-1) > 0 # agents that show up at some point
    # valid_agts = fut_valid[:,-1] # agents that are valid at target time
    if np.sum(valid_agts) == 0:
        return None
    if np.sum(valid_agts) < fut_pos.shape[0]:
        fut_pos = fut_pos[valid_agts]
        fut_valid = fut_valid[valid_agts]
        agents = agents[valid_agts]

    # take closest time to target we can get
    N, T = fut_valid.shape
    last_valid_t = np.amax(np.repeat(np.arange(T)[np.newaxis], N, axis=0) * fut_valid, axis=-1)
    target_pos = fut_pos[np.arange(N), last_valid_t]

    # add noise if desired
    if perturb_std is not None and perturb_std > 0.0:
        target_pos = target_pos + np.random.randn(*(target_pos.shape))*perturb_std

    guide_config = {
        'name' : 'global_target_pos_at_time',
        'params' : {
                    'target_pos' : target_pos.tolist(),
                    'target_time' : last_valid_t.tolist(),
                    'urgency' : [urgency]*N,    
                    'pref_speed' : pref_speed,
                    'dt' : dt,
                   },
        'agents' : agents.tolist()
    }
    return guide_config

def heuristic_global_target_pos(sim_scene, dt, target_time, urgency, pref_speed, min_progress_dist, perturb_std=None):
    '''
    Sets a global target pos using GT.
    '''
    guide_config = heuristic_global_target_pos_at_time(sim_scene, dt, target_time, urgency, pref_speed, perturb_std)
    guide_config['name'] = 'global_target_pos'
    guide_config['params']['min_progress_dist'] = min_progress_dist
    guide_config['params'].pop('target_time', None)
    return guide_config

def heuristic_target_pos_at_time(sim_scene, dt, target_time, perturb_std=None):
    '''
    Sets a local target pos and time using GT.
    '''
    fut_sec = target_time * dt
    fut_traj, fut_valid = get_agents_future(sim_scene, fut_sec)
    fut_pos = fut_traj[:,:,:2]
    agents = np.arange(fut_pos.shape[0])

    valid_agts = np.sum(fut_valid, axis=-1) > 0 # agents that show up at some point
    # valid_agts = fut_valid[:,-1] # agents that are valid at target time
    if np.sum(valid_agts) == 0:
        return None
    if np.sum(valid_agts) < fut_pos.shape[0]:
        fut_pos = fut_pos[valid_agts]
        fut_valid = fut_valid[valid_agts]
        agents = agents[valid_agts]

    # take closest time to target we can get
    N, T = fut_valid.shape
    last_valid_t = np.amax(np.repeat(np.arange(T)[np.newaxis], N, axis=0) * fut_valid, axis=-1)
    target_pos = fut_pos[np.arange(N), last_valid_t]

    # add noise if desired
    if perturb_std is not None and perturb_std > 0.0:
        target_pos = target_pos + np.random.randn(*(target_pos.shape))*perturb_std

    # convert to local frame
    curr_state = get_agents_curr(sim_scene)[valid_agts]
    agt_from_world_tf = get_agent_from_world_tf(curr_state)
    target_pos = batch_nd_transform_points_np(target_pos, agt_from_world_tf)

    guide_config = {
        'name' : 'target_pos_at_time',
        'params' : {
                    'target_pos' : target_pos.tolist(),
                    'target_time' : last_valid_t.tolist(),
                   },
        'agents' : agents.tolist()
    }
    return guide_config

def heuristic_target_pos(sim_scene, dt, target_time, perturb_std=None):
    '''
    Sets a target pos using GT.
    '''
    guide_config = heuristic_target_pos_at_time(sim_scene, dt, target_time, perturb_std)
    guide_config['name'] = 'target_pos'
    guide_config['params'].pop('target_time', None)
    return guide_config

def heuristic_agent_collision(sim_scene, dt, num_disks, buffer_dist):
    '''
    Applies collision loss to all agents.
    '''
    guide_config = {
        'name' : 'agent_collision',
        'params' : {
                    'num_disks' : num_disks,
                    'buffer_dist' : buffer_dist,
                    },
        'agents' : None, # all agents
    }
    return guide_config

def heuristic_map_collision(sim_scene, dt, num_points_lw):
    '''
    Applies collision loss to all agents.
    '''
    guide_config = {
        'name' : 'map_collision',
        'params' : {
                    'num_points_lw' : num_points_lw,
                    },
        'agents' : None, # all agents
    }
    return guide_config

HEURISTIC_FUNC = {
    'global_target_pos_at_time' : heuristic_global_target_pos_at_time,
    'global_target_pos' : heuristic_global_target_pos,
    'target_pos_at_time' : heuristic_target_pos_at_time,
    'target_pos' : heuristic_target_pos,
    'agent_collision' : heuristic_agent_collision,
    'map_collision' : heuristic_map_collision,
    'social_group' : heuristic_social_group,
}

def compute_heuristic_guidance(heuristic_config, env, scene_indices, start_frames):
    '''
    Creates guidance configs for each scene based on the given configuration.
    '''
    env.reset(scene_indices=scene_indices, start_frame_index=start_frames)
    heuristic_guidance_cfg = []
    for i, si in enumerate(scene_indices):
        scene_guidance = []
        cur_scene = env._current_scenes[i]
        dt = cur_scene.dataset.desired_dt
        for cur_heur in heuristic_config:
            assert set(('name', 'weight', 'params')).issubset(cur_heur.keys()), "All heuristics must have these 3 fields"
            assert cur_heur['name'] in HEURISTIC_FUNC, "Unrecognized heuristic!"
            dt = cur_heur['params'].pop('dt', dt) # some already include dt, don't want to duplicate
            cur_guidance = HEURISTIC_FUNC[cur_heur['name']](cur_scene, dt, **cur_heur['params'])
            if cur_guidance is not None:
                if not isinstance(cur_guidance, list):
                    cur_guidance = [cur_guidance]
                for guide_el in cur_guidance:
                    guide_el['weight'] = cur_heur['weight']
                    scene_guidance.append(guide_el)
        heuristic_guidance_cfg.append(scene_guidance)

    return heuristic_guidance_cfg


