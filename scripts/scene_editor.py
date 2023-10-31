"""A script for evaluating closed-loop simulation"""
import argparse
import numpy as np
import json
import random
import importlib
from pprint import pprint

import os
import torch

from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env, set_global_trajdata_batch_raster_cfg
from tbsim.configs.scene_edit_config import SceneEditingConfig
from tbsim.utils.scene_edit_utils import guided_rollout, compute_heuristic_guidance, merge_guidance_configs
from tbsim.evaluation.env_builders import EnvUnifiedBuilder
from tbsim.policies.wrappers import RolloutWrapper

from tbsim.utils.tensor_utils import map_ndarray

def run_scene_editor(eval_cfg, save_cfg, data_to_disk, render_to_video, render_to_img, render_cfg,
                     use_gt=False):
    # assumes all used trajdata datasets use share same map layers
    set_global_trajdata_batch_env(eval_cfg.trajdata_source_test[0])

    print(eval_cfg)

    # for reproducibility
    np.random.seed(eval_cfg.seed)
    random.seed(eval_cfg.seed)
    torch.manual_seed(eval_cfg.seed)
    torch.cuda.manual_seed(eval_cfg.seed)

    # basic setup
    print('saving results to {}'.format(eval_cfg.results_dir))
    os.makedirs(eval_cfg.results_dir, exist_ok=True)
    if render_to_video:
        os.makedirs(os.path.join(eval_cfg.results_dir, "videos/"), exist_ok=True)
    if render_to_video or render_to_img:
        os.makedirs(os.path.join(eval_cfg.results_dir, "viz/"), exist_ok=True)
    if save_cfg:
        json.dump(eval_cfg, open(os.path.join(eval_cfg.results_dir, "config.json"), "w+"))
    if data_to_disk and os.path.exists(eval_cfg.experience_hdf5_path):
        os.remove(eval_cfg.experience_hdf5_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create policy and rollout wrapper
    policy_composers = importlib.import_module("tbsim.evaluation.policy_composers")
    composer_class = getattr(policy_composers, eval_cfg.eval_class)
    composer = composer_class(eval_cfg, device)
    policy, exp_config = composer.get_policy()
    policy_model = policy.model
    if use_gt:
        from tbsim.policies.hardcoded import GTNaNPolicy
        # overwrite policy with dummy that always returns GT
        policy = GTNaNPolicy(device=device)
        policy_model = None
        print('WARNING: Using GT data as the policy instead of the provided model!!')

    # determines cfg for rasterizing agents
    set_global_trajdata_batch_raster_cfg(exp_config.env.rasterizer)
    
    # control the full scene with same policy
    rollout_policy = RolloutWrapper(agents_policy=policy)
    
    print(exp_config.algo)

    # create env
    env_builder = EnvUnifiedBuilder(eval_config=eval_cfg, exp_config=exp_config, device=device)
    env = env_builder.get_env()

    # set up heuristic guidance config
    heuristic_config = None
    if "heuristic" in eval_cfg.edits.editing_source:
        if eval_cfg.edits.heuristic_config is not None:
            heuristic_config = eval_cfg.edits.heuristic_config
        else:
            heuristic_config = []

    render_rasterizer = None
    if render_to_video or render_to_img:
        from tbsim.utils.viz_utils import get_trajdata_renderer
        # initialize rasterizer once for all scenes
        render_rasterizer = get_trajdata_renderer(eval_cfg.trajdata_source_test,
                                                    eval_cfg.trajdata_data_dirs,
                                                    raster_size=render_cfg['size'],
                                                    px_per_m=render_cfg['px_per_m'],
                                                    rebuild_maps=False,
                                                    cache_location=eval_cfg.trajdata_cache_location)

    result_stats = None
    scene_i = 0
    eval_scenes = eval_cfg.eval_scenes
    while scene_i < eval_cfg.num_scenes_to_evaluate:
        scene_indices = eval_scenes[scene_i: scene_i + eval_cfg.num_scenes_per_batch]
        scene_i += eval_cfg.num_scenes_per_batch
        print(scene_indices)

        # check to make sure all the scenes are valid at starting step
        scenes_valid = env.reset(scene_indices=scene_indices, start_frame_index=None)
        scene_indices = [si for si, sval in zip(scene_indices, scenes_valid) if sval]
        if len(scene_indices) == 0:
            torch.cuda.empty_cache()
            continue

        # if requested, split each scene up into multiple simulations
        start_frame_index = [[exp_config.algo.history_num_frames+1]] * len(scene_indices)
        if eval_cfg.num_sim_per_scene > 1:
            start_frame_index = []
            for si in range(len(scene_indices)):
                cur_scene = env._current_scenes[si].scene
                sframe = exp_config.algo.history_num_frames+1
                # want to make sure there's GT for the full rollout
                eframe = cur_scene.length_timesteps - eval_cfg.num_simulation_steps
                scene_frame_inds = np.linspace(sframe, eframe, num=eval_cfg.num_sim_per_scene, dtype=int).tolist()
                start_frame_index.append(scene_frame_inds)

        # how many sims to run for the current batch of scenes
        print('Starting frames in current scenes:')
        print(start_frame_index)
        for ei in range(eval_cfg.num_sim_per_scene):
            guidance_config = None   # for the current batch of scenes
            
            cur_start_frames = [scene_start[ei] for scene_start in start_frame_index]
            # double check all scenes are valid at the current start step
            scenes_valid = env.reset(scene_indices=scene_indices, start_frame_index=cur_start_frames)
            sim_scene_indices = [si for si, sval in zip(scene_indices, scenes_valid) if sval]
            sim_start_frames = [sframe for sframe, sval in zip(cur_start_frames, scenes_valid) if sval]
            if len(sim_scene_indices) == 0:
                torch.cuda.empty_cache()
                continue

            # getting edits from either the config file or on-the-fly heuristics
            if "config" in eval_cfg.edits.editing_source:
                guidance_config = eval_cfg.edits.guidance_config
            if "heuristic" in eval_cfg.edits.editing_source:
                # build heuristic guidance configs for these scenes
                heuristic_guidance_cfg = compute_heuristic_guidance(heuristic_config,
                                                                    env,
                                                                    sim_scene_indices,
                                                                    sim_start_frames)
                if len(heuristic_config) > 0:
                    # we asked to apply some guidance, but if heuristic determined there was no valid
                    #       guidance to apply (e.g. no social groups), we should skip these scenes.
                    valid_scene_inds = []
                    for sci, sc_cfg in enumerate(heuristic_guidance_cfg):
                        if len(sc_cfg) > 0:
                            valid_scene_inds.append(sci)

                    # collect only valid scenes under the given heuristic config
                    heuristic_guidance_cfg = [heuristic_guidance_cfg[vi] for vi in valid_scene_inds]
                    sim_scene_indices = [sim_scene_indices[vi] for vi in valid_scene_inds]
                    sim_start_frames = [sim_start_frames[vi] for vi in valid_scene_inds]
                    # skip if no valid...
                    if len(sim_scene_indices) == 0:
                        print('No scenes with valid heuristic configs in this batch, skipping...')
                        torch.cuda.empty_cache()
                        continue

                # add to the current guidance config
                guidance_config = merge_guidance_configs(guidance_config, heuristic_guidance_cfg)

            stats, info = guided_rollout(
                env,
                rollout_policy,
                policy_model,
                n_step_action=eval_cfg.n_step_action,
                guidance_config=guidance_config,
                scene_indices=sim_scene_indices,
                obs_to_torch=True,
                horizon=eval_cfg.num_simulation_steps,
                use_gt=use_gt,
                start_frames=sim_start_frames,
            )    

            print(info["scene_index"])
            print(sim_start_frames)
            pprint(stats)

            # aggregate stats from the same class of guidance within each scene
            #       this helps parse_scene_edit_results
            guide_agg_dict = {}
            pop_list = []
            for k,v in stats.items():
                if k.split('_')[0] == 'guide':
                    guide_name = '_'.join(k.split('_')[:-1])
                    guide_scene_tag = k.split('_')[-1][:2]
                    canon_name = guide_name + '_%sg0' % (guide_scene_tag)
                    if canon_name not in guide_agg_dict:
                        guide_agg_dict[canon_name] = []
                    guide_agg_dict[canon_name].append(v)
                    # remove from stats
                    pop_list.append(k)
            for k in pop_list:
                stats.pop(k, None)
            # average over all of the same guide stats in each scene
            for k,v in guide_agg_dict.items():
                scene_stats = np.stack(v, axis=0) # guide_per_scenes x num_scenes (all are nan except 1)
                stats[k] = np.mean(scene_stats, axis=0)

            # aggregate metrics stats
            if result_stats is None:
                result_stats = stats
                result_stats["scene_index"] = np.array(info["scene_index"])
            else:
                for k in stats:
                    if k not in result_stats:
                        result_stats[k] = stats[k]
                    else:
                        result_stats[k] = np.concatenate([result_stats[k], stats[k]], axis=0)
                result_stats["scene_index"] = np.concatenate([result_stats["scene_index"], np.array(info["scene_index"])])

            # write stats to disk
            with open(os.path.join(eval_cfg.results_dir, "stats.json"), "w+") as fp:
                stats_to_write = map_ndarray(result_stats, lambda x: x.tolist())
                json.dump(stats_to_write, fp)

            if render_to_video or render_to_img:
                # high quality
                from tbsim.utils.viz_utils import visualize_guided_rollout
                scene_cnt = 0
                for si, scene_buffer in zip(info["scene_index"], info["buffer"]):
                    viz_dir = os.path.join(eval_cfg.results_dir, "viz/")
                    invalid_guidance = guidance_config is None or len(guidance_config) == 0
                    visualize_guided_rollout(viz_dir, render_rasterizer, si, scene_buffer,
                                                guidance_config=None if invalid_guidance else guidance_config[scene_cnt],
                                                fps=(1.0 / exp_config.algo.step_time),
                                                n_step_action=eval_cfg.n_step_action,
                                                viz_diffusion_steps=False,
                                                first_frame_only=render_to_img,
                                                viz_traj=True, #(not render_to_video),
                                                sim_num=sim_start_frames[scene_cnt])
                    scene_cnt += 1

            if data_to_disk and "buffer" in info:
                dump_episode_buffer(
                    info["buffer"],
                    info["scene_index"],
                    sim_start_frames,
                    h5_path=eval_cfg.experience_hdf5_path
                )
            torch.cuda.empty_cache()


def dump_episode_buffer(buffer, scene_index, start_frames, h5_path):
    import h5py
    h5_file = h5py.File(h5_path, "a")

    for ei, si, scene_buffer in zip(start_frames, scene_index, buffer):
        for mk in scene_buffer:
            h5key = "/{}_{}/{}".format(si, ei, mk)
            h5_file.create_dataset(h5key, data=scene_buffer[mk])
    h5_file.close()
    print("scene {} written to {}".format(scene_index, h5_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="A json file containing evaluation configs"
    )

    parser.add_argument(
        "--eval_class",
        type=str,
        default=None,
        help="Optionally specify the evaluation class through argparse"
    )

    parser.add_argument(
        "--policy_ckpt_dir",
        type=str,
        default=None,
        help="Directory to look for saved checkpoints"
    )

    parser.add_argument(
        "--policy_ckpt_key",
        type=str,
        default=None,
        help="A string that uniquely identifies a checkpoint file within a directory, e.g., iter50000"
    )

    parser.add_argument(
        "--results_root_dir",
        type=str,
        default=None,
        help="Directory to save results and videos"
    )

    parser.add_argument(
        "--num_scenes_per_batch",
        type=int,
        default=None,
        help="Number of scenes to run concurrently (to accelerate eval)"
    )

    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="whether to render videos"
    )

    parser.add_argument(
        "--render_img",
        action="store_true",
        default=False,
        help="whether to render first frame of rollout"
    )

    parser.add_argument(
        "--render_size",
        type=int,
        default=400,
        help="width and height of the rendered image size in pixels"
    )

    parser.add_argument(
        "--render_px_per_m",
        type=float,
        default=12.0,
        help="resolution of rendering"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--ground_truth",
        action="store_true",
        default=False,
        help="outputs the ground truth rollout instead of predictions"
    )

    #
    # Editing options
    #
    parser.add_argument(
        "--editing_source",
        type=str,
        choices=["config", "heuristic", "none"],
        nargs="+",
        help="Which edits to use. config is directly from the configuration file. heuristic will \
              set edits automatically based on heuristics parameterized in config file. \
              config and heuristic may be used together. If none, does not use edits",
    )

    args = parser.parse_args()

    cfg = SceneEditingConfig()

    if args.config_file is not None:
        external_cfg = json.load(open(args.config_file, "r"))
        cfg.update(**external_cfg)

    if args.eval_class is not None:
        cfg.eval_class = args.eval_class

    if args.policy_ckpt_dir is not None:
        assert args.policy_ckpt_key is not None, "Please specify a key to look for the checkpoint, e.g., 'iter50000'"
        cfg.ckpt.policy.ckpt_dir = args.policy_ckpt_dir
        cfg.ckpt.policy.ckpt_key = args.policy_ckpt_key

    if args.num_scenes_per_batch is not None:
        cfg.num_scenes_per_batch = args.num_scenes_per_batch

    if cfg.name is None:
        cfg.name = cfg.eval_class

    if args.prefix is not None:
        cfg.name = args.prefix + cfg.name

    if args.seed is not None:
        cfg.seed = args.seed
    if args.results_root_dir is not None:
        cfg.results_dir = os.path.join(args.results_root_dir, cfg.name)
    else:
        cfg.results_dir = os.path.join(cfg.results_dir, cfg.name)

    assert cfg.env == "trajdata", "trajdata is the only supported environment"

    if args.editing_source is not None:
        cfg.edits.editing_source = args.editing_source
    if not isinstance(cfg.edits.editing_source, list):
        cfg.edits.editing_source = [cfg.edits.editing_source]

    cfg.experience_hdf5_path = os.path.join(cfg.results_dir, "data.hdf5")

    for k in cfg[cfg.env]:  # copy env-specific config to the global-level
        cfg[k] = cfg[cfg.env][k]
    cfg.pop("trajdata")

    render_cfg = {
        'size' : args.render_size,
        'px_per_m' : args.render_px_per_m,
    }
    
    cfg.lock()
    run_scene_editor(
        cfg,
        save_cfg=True,
        data_to_disk=True,
        render_to_video=args.render,
        render_to_img=args.render_img,
        render_cfg=render_cfg,
        use_gt=args.ground_truth,
    )
