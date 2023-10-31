import argparse
import os
import os.path as osp
import math
import pickle
from tqdm import tqdm

import numpy as np
from shapely.geometry import Point, Polygon
import shapely.affinity

import rvo2

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mplPolygon
from matplotlib.patches import Circle as mplCircle

SIM_FPS = 60 # fps
# defines square region peds must stay in
BOUNDARY_RADIUS = 15.0 # m
# gives range to sample agent properties from
#       along with the size of the property
# some ranges given in this paper https://gamma.cs.unc.edu/CParameter/
#                                 https://gamma.cs.unc.edu/CParameter/eg2014-appendix.pdf
AGT_PROP_INFO =  { 
    'neighborDist' :    ((10.0, 20.0), 1, float),
    'maxNeighbors' :    ((10, 10), 1, int),
    'timeHorizon'  :    ((2.0, 5.0), 1, float),
    'timeHorizonObst' : ((2.0, 5.0), 1, float),
    'radius' :          ((0.4, 0.4), 1, float),
    'maxSpeed' :        ((2.0, 2.0), 1, float),
    'initVelocity' :    ((0.0, 0.0), 2, float), 
    'prefVelocity' :    ((1.0, 2.0), 1, float), # 1.42
}

RESAMPLE_GOAL_RANGE = (2.0, 4.0) # re-sample goal randomly every between 2 to 4 sec

# whether to actually limit agent motion outisde of boundary (otherwise it's just used to place obstacles and spawn agents initially)
USE_BOUNDARY = False 
# must be clockwise
BOUNDARY_OBS = [(BOUNDARY_RADIUS, BOUNDARY_RADIUS),
                (BOUNDARY_RADIUS, -BOUNDARY_RADIUS),
                (-BOUNDARY_RADIUS, -BOUNDARY_RADIUS),
                (-BOUNDARY_RADIUS, BOUNDARY_RADIUS)]

# possible obstacles
OBS_LIST = ['ellipse', 'box', 'tri']
# ranges for sampling obstacles
OBSTACLE_PROPS = {
    'width' : (3.0, 10.0),
    'height' : (3.0, 10.0),
    'rot' : (0.0, 360.)
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='./datagen_out', help='Dir to write results to')
    parser.add_argument("--no_save_maps", action="store_true", help="If given, does not save map (even if obstacles are used in simulation).")

    parser.add_argument('--num_scenes', type=int, default=10, help='Number of scenes to generate')
    parser.add_argument('--scene_len', type=float, default=10.0, help='Scene length (sec)')
    parser.add_argument('--save_rate', type=int, default=10, help='Outputs scene data at this FPS')
    parser.add_argument('--resample_goal', action="store_true", help='Resamples the preferred velocity for each agent periodically')

    parser.add_argument('--max_agents', type=int, default=20, help='Maximum number of agents that can appear in a scene.')
    parser.add_argument('--max_obstacles', type=int, default=5, help='Maximum number of obstacles that can appear in a scene. If 0, no map data will be saved!')

    parser.add_argument("--viz", action="store_true", help="Viz traj of each sim")
    parser.add_argument("--viz_vid", action="store_true", help="Viz video of each sim")

    config = parser.parse_known_args()
    config = config[0]

    return config

def init_log(num_save_steps, agents):
    '''
    - num_save_steps : the number of timesteps that will be saved during the sim
    - agents: dict of agent properties given to run sim
    '''
    nagents = agents['numAgents']
    log = {
        'trackTime' : np.ones((num_save_steps))*np.nan,
        'trackPos' : np.ones((nagents, num_save_steps, 2))*np.nan,
        'trackVel' : np.ones((nagents, num_save_steps, 2))*np.nan,
        **agents # save agent properties too
    }
    return log

def log_step(log, save_step, sim, nagents):
    cur_pos = np.array([sim.getAgentPosition(aidx) for aidx in range(nagents)])
    cur_vel = np.array([sim.getAgentVelocity(aidx) for aidx in range(nagents)])
    log['trackTime'][save_step] = sim.getGlobalTime()
    log['trackPos'][:,save_step] = cur_pos
    log['trackVel'][:,save_step] = cur_vel

def resample_goals(sim, cur_step, next_resample):
    # reset preferred velocity for each agent who is due
    needs_resample = next_resample == cur_step
    if np.sum(needs_resample) == 0:
        return next_resample
    resamp_inds = np.nonzero(needs_resample)[0]
    # perturb direction
    cur_pref_vels = np.array([sim.getAgentPrefVelocity(aidx) for aidx in resamp_inds])
    cur_dir = cur_pref_vels / np.linalg.norm(cur_pref_vels, axis=-1, keepdims=True)
    cur_theta = np.arctan2(cur_dir[:,1:2], cur_dir[:,0:1])
    cur_theta += np.pi/18 * np.random.randn(*cur_theta.shape) # std of 10 deg
    new_dir = np.concatenate([np.cos(cur_theta), np.sin(cur_theta)], axis=-1)

    prop_range, prop_size, _ = AGT_PROP_INFO['prefVelocity']
    prop_samp = np.random.uniform(prop_range[0], prop_range[1], (new_dir.shape[0], prop_size))
    prop_samp = new_dir*prop_samp

    for acount, aidx in enumerate(resamp_inds):
        sim.setAgentPrefVelocity(aidx, tuple(prop_samp[acount]))

    next_resample[resamp_inds] += (np.random.uniform(RESAMPLE_GOAL_RANGE[0], RESAMPLE_GOAL_RANGE[1], (len(resamp_inds)))*SIM_FPS).astype(int)

    return next_resample

def run_sim(agents, obstacles, scene_len, save_rate, resample_goal):
    '''
    single sim after sampling num agents, initial states, etc..
    - agents : dict properties of all agents to simulate
    - obstacles : list of polygons (list of 2d points).
    - scene_len : length of simulated scene (in sec)
    - save_rate : fps to save simulation at
    - resample_goal : whether to periodically reset the goal of each agent
    '''
    # init sim. Set defaults to min property range by default (will not be used anyway)
    sim_step = 1. / SIM_FPS
    sim = rvo2.PyRVOSimulator(sim_step, # timeStep
                            AGT_PROP_INFO['neighborDist'][0][0],   # neighborDist
                            AGT_PROP_INFO['maxNeighbors'][0][0],     # maxNeighbors
                            AGT_PROP_INFO['timeHorizon'][0][0],   # timeHorizon
                            AGT_PROP_INFO['timeHorizonObst'][0][0],     # timeHorizonObst
                            AGT_PROP_INFO['radius'][0][0],   # radius
                            AGT_PROP_INFO['maxSpeed'][0][0])     # maxSpeed

    # add obstacles
    for obs_poly in obstacles:
        sim.addObstacle(obs_poly)
    sim.processObstacles()

    # add agents
    nagents = agents['numAgents']
    for aidx in range(nagents):
        cur_agt = sim.addAgent(tuple(agents['initPos'][aidx]), # init position
                                agents['neighborDist'][aidx][0], # neighborDist
                                agents['maxNeighbors'][aidx][0],   # maxNeighbors
                                agents['timeHorizon'][aidx][0], # timeHorizon
                                agents['timeHorizonObst'][aidx][0],   # timeHorizonObst
                                agents['radius'][aidx][0], # radius
                                agents['maxSpeed'][aidx][0],   # maxSpeed
                                tuple(agents['initVelocity'][aidx])) # velocity
        sim.setAgentPrefVelocity(cur_agt, tuple(agents['prefVelocity'][aidx]))

    print('Simulation has %i agents and %i obstacle vertices in it.' %
      (sim.getNumAgents(), sim.getNumObstacleVertices()))

    # simulate
    scene_len_steps = int(scene_len * SIM_FPS)
    assert save_rate <= SIM_FPS, 'Can only save at leq the rate of simulation (%d FPS)' % (int(SIM_FPS))
    save_every = (1. / save_rate) / sim_step
    print('Running simulation for %d steps (%f sec)...' % (scene_len_steps, scene_len))
    print('Saving every %d steps...' % (save_every))
    num_save_steps = int(math.ceil(scene_len_steps / save_every))
    if scene_len_steps % save_every == 0:
        num_save_steps += 1
    next_resample = None
    if resample_goal:
        next_resample = np.random.uniform(RESAMPLE_GOAL_RANGE[0], RESAMPLE_GOAL_RANGE[1], (nagents))*SIM_FPS
        next_resample = next_resample.astype(int)

    log = init_log(num_save_steps, agents)
    save_step = 0
    for step in tqdm(range(scene_len_steps)):
        sim.doStep()
        if step % save_every == 0:
            # save current states
            log_step(log, save_step, sim, nagents)
            save_step += 1
        if next_resample is not None and step > 0:
            next_resample = resample_goals(sim, step, next_resample)

    if scene_len_steps % save_every == 0:
        log_step(log, save_step, sim, nagents)
    
    return log

def contains_py(poly, array): 
    return np.array([poly.contains(p) for p in array]) 

def intersects_py(poly, array):
    return np.array([poly.intersects(p) for p in array])

def intersects_scene(scene_poly, array):
    return np.any([intersects_py(obs_poly, array) for obs_poly in scene_poly], axis=0)

def samp_init_pos(nagents, agent_radii, bound_rad, obstacles):
    '''
    Samples initial position for all agents such that none collide
    with the polygon obstacles in the scene, are out of bounds, or collide with each other.
    - nagents: int
    - agent_radii: List[float] radius of each agent
    - bound_rad: float > 0 boundary radius (assumes square boundary)
    - obstacles: [List of 2d tuples] assumes the boundary obstacles is NOT included.
    '''
    scene_poly = [Polygon(pt_list) for pt_list in obstacles]
    # limit init pos to be in inner half of area
    #   to force more agent-agent interactions
    lim_bound_fact = 0.0
    init_pos = np.random.uniform(-bound_rad*(1.0-lim_bound_fact),
                                 bound_rad*(1.0-lim_bound_fact),
                                 (nagents, 2))
    init_pos[init_pos[:,0] < 0, 0] -= bound_rad*lim_bound_fact
    init_pos[init_pos[:,0] > 0, 0] += bound_rad*lim_bound_fact
    init_pos[init_pos[:,1] < 0, 1] -= bound_rad*lim_bound_fact
    init_pos[init_pos[:,1] > 0, 1] += bound_rad*lim_bound_fact

    # re-sample until no positions are inside obstacles
    all_valid = False
    cnt = 0
    while not all_valid:
        pos_pts = [Point(x,y).buffer(r) for x,y,r in np.concatenate([init_pos, agent_radii], axis=1).tolist()]
        in_collision = np.zeros((nagents), dtype=bool)
        # check for obstacle collisions
        if len(scene_poly) > 0:
            in_obstacle = np.array(intersects_scene(scene_poly, pos_pts))
            in_collision = np.logical_or(in_collision, in_obstacle)
        # check for agent-agent collisions
        if nagents > 1:
            pair_thresh = agent_radii + agent_radii.T
            pair_dist = np.linalg.norm(init_pos[np.newaxis] - init_pos[:,np.newaxis], axis=-1)
            pair_collide = np.logical_and(pair_dist < pair_thresh, ~np.eye(nagents, dtype=bool))
            agt_collide = np.sum(pair_collide, axis=-1) > 0
            in_collision = np.logical_or(in_collision, agt_collide)

        num_collide = np.sum(in_collision)
        if np.sum(num_collide) > 0:
            # resample the necessary indices
            resamp_pos = np.random.uniform(-bound_rad*(1.0-lim_bound_fact),
                                            bound_rad*(1.0-lim_bound_fact),
                                            (num_collide, 2))
            resamp_pos[resamp_pos[:,0] < 0, 0] -= bound_rad*lim_bound_fact
            resamp_pos[resamp_pos[:,0] > 0, 0] += bound_rad*lim_bound_fact
            resamp_pos[resamp_pos[:,1] < 0, 1] -= bound_rad*lim_bound_fact
            resamp_pos[resamp_pos[:,1] > 0, 1] += bound_rad*lim_bound_fact
            init_pos[in_collision] = resamp_pos
        else:
            all_valid = True
        cnt += 1

    return init_pos

def samp_obstacles(max_obstacles, bound_rad):
    '''
    Randomly samples a configuration of obstacles to use as a map.
    '''
    if max_obstacles == 0:
        return []

    nobs = np.random.randint(0, max_obstacles+1)
    obstacles = []
    for oi in range(nobs):
        # obstacle type
        obs_type = OBS_LIST[np.random.randint(0, len(OBS_LIST))]
        if obs_type == 'ellipse':
            obstacle = Point(0.0, 0.0).buffer(0.5)
        elif obs_type == 'box':
            obstacle = Polygon([(-0.5, 0.5), (0.5, 0.5), (0.5, -0.5), (-0.5, -0.5)])
        elif obs_type == 'tri':
            obstacle = Polygon([(-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)])

        # scale
        xfact = np.random.uniform(OBSTACLE_PROPS['width'][0], OBSTACLE_PROPS['width'][1])
        yfact = np.random.uniform(OBSTACLE_PROPS['height'][0], OBSTACLE_PROPS['height'][1])
        obstacle = shapely.affinity.scale(obstacle, xfact=xfact, yfact=yfact)
        # rotate
        angle = np.random.uniform(OBSTACLE_PROPS['rot'][0], OBSTACLE_PROPS['rot'][1])
        obstacle = shapely.affinity.rotate(obstacle, angle)
        # trans
        obs_pos = np.random.uniform(-bound_rad, bound_rad, (2,))
        obstacle = shapely.affinity.translate(obstacle, xoff=obs_pos[0], yoff=obs_pos[1])

        # need to reverse order to be counter clockwise
        cur_obs = [(x,y) for x,y in np.asarray(obstacle.boundary.xy)[:,::-1].T.tolist()]
        obstacles.append(cur_obs)

    return obstacles

def gen_scene(max_agents, max_obstacles, out_dir, scene_len, resample_goal, save_rate, 
              viz=False,
              viz_vid=False,
              no_save_maps=False):
    '''
    Gen single scene.
    - max_agents: maximum number of agents in a scene
    - out_dir : where to save scene outputs to
    - scene_len : length of simulated scene (in sec)
    - resample_goal_every : resample goal every this many secs
    - save_rate : fps to save simulation at
    - viz : whether to save visualization of simulated scene
    - no_save_maps : don't save maps even if used in sim
    '''
    env_obs = samp_obstacles(max_obstacles, BOUNDARY_RADIUS)

    if USE_BOUNDARY:
        obstacles = [BOUNDARY_OBS] + env_obs
    else:
        obstacles = env_obs

    #
    # sample pedestrian properties
    #
    nagents = np.random.randint(1, max_agents+1)
    agent_dict = {'numAgents' : nagents}
    # sample radius first since need this for pos
    prop_range, prop_size, _ = AGT_PROP_INFO['radius']
    prop_samp = np.random.uniform(prop_range[0], prop_range[1], (nagents, prop_size))
    agent_dict['radius'] = prop_samp
    # pos is based on map and boundary
    init_pos = samp_init_pos(nagents, agent_dict['radius'], BOUNDARY_RADIUS, env_obs)
    agent_dict['initPos'] = init_pos
    # other properties
    for prop_name, prop_info in AGT_PROP_INFO.items():
        if prop_name == 'radius':
            continue
        prop_range, prop_size, prop_type = prop_info
        if prop_type is int:
            prop_samp = np.random.randint(prop_range[0], prop_range[1]+1, (nagents, prop_size))
        else:
            prop_samp = np.random.uniform(prop_range[0], prop_range[1], (nagents, prop_size))
            if prop_name == 'prefVelocity':
                init_pos = agent_dict['initPos']
                dir_samp = -init_pos # point towards center
                dir_samp = dir_samp / np.linalg.norm(dir_samp, axis=-1, keepdims=True)
                theta_samp = np.arctan2(dir_samp[:,1:2], dir_samp[:,0:1])
                # add some noise so it's not exactly at center
                theta_samp += np.random.uniform(-np.pi/4, np.pi/4, theta_samp.shape)
                dir_samp = np.concatenate([np.cos(theta_samp), np.sin(theta_samp)], axis=-1)
                prop_samp = dir_samp*prop_samp
        agent_dict[prop_name] = prop_samp
    
    # simulate
    sim_data = run_sim(agent_dict, obstacles, scene_len, save_rate, resample_goal)
    # save recorded traj and agent properties
    np.savez(osp.join(out_dir, 'sim.npz'), **sim_data)

    # only save non-boundary obstacles
    if max_obstacles > 0 and not no_save_maps:
        with open(osp.join(out_dir, 'map.pkl'), 'wb') as mapf:
            pickle.dump(env_obs, mapf)

    if viz or viz_vid:
        viz_scene(obstacles, sim_data, osp.join(out_dir, 'viz_sim'),
                    vid=viz_vid,
                    includes_boundary=USE_BOUNDARY)

def create_video(img_path_form, out_path, fps):
    '''
    Creates a video from a format for frame e.g. 'data_out/frame%04d.png'.
    Saves in out_path.
    '''
    import subprocess
    subprocess.run(['ffmpeg', '-y', '-r', str(fps), '-i', img_path_form,
                    '-vcodec', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', out_path])

def plt_color(i):
    return plt.rcParams['axes.prop_cycle'].by_key()['color'][i % 9]

def plot_obstacles(obstacles, ax, includes_boundary):
    for obsi, obs_poly in enumerate(obstacles):
        polypatch = mplPolygon(obs_poly,
                                color='red' if obsi == 0 and includes_boundary else 'gray',
                                fill=False if obsi == 0 and includes_boundary else True,
                                alpha=1.0 if obsi == 0 and includes_boundary else 0.5,
                                linestyle='--' if obsi == 0 and includes_boundary else '-')
        ax.add_patch(polypatch)

def viz_scene(obstacles, sim_data, save_path, vid=False, includes_boundary=False):
    '''
    - obstacles : list of polygons. First is assumes to be the boundary.
    - sim_data: dict of saved simulated trajectories and agent properties
    '''
    print('Rendering...')

    #
    # First just trajectories
    #
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    # map
    plot_obstacles(obstacles, ax, includes_boundary)
    # trajectories
    for ai in range(sim_data['numAgents']):
        traj = sim_data['trackPos'][ai]
        plt.plot(traj[:,0], traj[:,1], '-', c=plt_color(ai))
        ax.add_patch(mplCircle(traj[0], radius=sim_data['radius'][ai][0],
                     edgecolor='k', facecolor=plt_color(ai), alpha=0.5, zorder=3))
    plt.xlim(-BOUNDARY_RADIUS-1.0, BOUNDARY_RADIUS+1.0)
    plt.ylim(-BOUNDARY_RADIUS-1.0, BOUNDARY_RADIUS+1.0)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(save_path + '.png')
    plt.close(fig)

    # 
    # then video if desired
    # 
    if vid:
        if not osp.exists(save_path):
            os.makedirs(save_path)
        track_time = sim_data['trackTime']
        for t in tqdm(range(track_time.shape[0])):
            frame_path = osp.join(save_path, 'frame_%06d.jpg' % (t))
            fig = plt.figure(figsize=(5, 5))
            ax = plt.gca()
            # map
            plot_obstacles(obstacles, ax, includes_boundary)
            # current step
            for ai in range(sim_data['numAgents']):
                traj = sim_data['trackPos'][ai]
                traj_vel = sim_data['trackVel'][ai]
                ax.add_patch(mplCircle(traj[t], radius=sim_data['radius'][ai][0],
                            edgecolor='k', facecolor=plt_color(ai), alpha=0.5, zorder=3))
                plt.arrow(traj[t,0], traj[t,1], traj_vel[t,0], traj_vel[t,1], 
                            color=plt_color(ai), alpha=1.0, zorder=3)
            plt.xlim(-BOUNDARY_RADIUS-1.0, BOUNDARY_RADIUS+1.0)
            plt.ylim(-BOUNDARY_RADIUS-1.0, BOUNDARY_RADIUS+1.0)
            ax.grid(False)
            plt.tight_layout()
            plt.savefig(frame_path)
            plt.close(fig)
        fps = round(1. / (track_time[1] - track_time[0]))
        create_video(osp.join(save_path, 'frame_%06d.jpg'),
                    save_path + '.mp4',
                    fps)

def gen_dataset(cfg):
    print(cfg)
    if not osp.exists(cfg.out):
        os.makedirs(cfg.out)

    for sidx in range(cfg.num_scenes):
        print('============ SCENE %d / %d ============ ' % (sidx, cfg.num_scenes-1))
        scene_out_pth = osp.join(cfg.out, 'scene_%06d' % (sidx))
        if not osp.exists(scene_out_pth):
            os.makedirs(scene_out_pth)

        gen_scene(cfg.max_agents, cfg.max_obstacles, scene_out_pth, args.scene_len, args.resample_goal, args.save_rate,
                  viz=args.viz,
                  viz_vid=args.viz_vid,
                  no_save_maps=args.no_save_maps)    

if __name__ == '__main__':
    args = parse_args()
    gen_dataset(args)