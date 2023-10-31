import json
import argparse
import math
import numpy as np
import os
from copy import copy
import torch
import h5py
from trajdata.simulation.sim_stats import calc_stats
import tbsim.utils.tensor_utils as TensorUtils
import glob
import csv

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


SIM_DT = 0.1
STAT_DT = 0.5
HIST_NUM_BINS = 40
HIST_VEL_MAX = 2
HIST_ACCEL_MAX = 1
HIST_JERK_MAX = 1

SAVE_STAT_NAMES = ['ade', 'fde', 'all_disk_off_road_rate_rate', 'all_disk_collision_rate_coll_any', \
                    'guide_agent_collision_disk', 'guide_social_dist', \
                    'guide_map_collision_disk', 'guide_target_pos', 'guide_target_pos_at_time', \
                    'guide_global_target_pos', 'guide_global_target_pos_at_time', 'guide_social_group']
# stats that are variations on the same root, will be collected and saved
COLLECT_SAVE_STAT_NAMES = ['all_sem_layer_rate_', 'all_comfort_', 'emd_']

def parse_single_result(results_dir, gt_hist_path=None):
    rjson = json.load(open(os.path.join(results_dir, "stats.json"), "r"))
    cfg = json.load(open(os.path.join(results_dir, "config.json"), "r"))

    eval_out_dir = os.path.join(results_dir, 'eval_out')
    if not os.path.exists(eval_out_dir):
        os.makedirs(eval_out_dir)

    agg_results = dict()
    scene_results = dict()
    guide_mets = []
    for k in rjson:
        if k == "scene_index":
            scene_results["names"] = rjson[k]
        else:
            if k.split('_')[0] == 'guide':
                # guide metrics must be handled specially (many nans from scene masking)
                met_name = '_'.join(k.split('_')[:-1])
                if met_name in scene_results:
                    # scene_results[met_name] = np.stack([scene_results[met_name], rjson[k]], axis=-1)
                    scene_results[met_name].append(rjson[k])
                else:
                    scene_results[met_name] = [rjson[k]]
                    guide_mets.append(met_name)
            else:
                per_scene = rjson[k]
                scene_results[k] = rjson[k]
                agg_results[k] = per_scene
                if np.sum(np.isnan(rjson[k])) > 0:
                    print("WARNING: metric %s has some nan values! Ignoring them..." % (k))
                rnum = np.nanmean(rjson[k])
                agg_results[k] = rnum
                print("{} = {}".format(k, np.nanmean(rjson[k])))

    num_scenes = len(scene_results["names"])

    for guide_met in guide_mets:
        met_arr = scene_results[guide_met]

        # collect stats across all "scene" versions of guide metrics
        out_met_arr = []
        sc_cnt = [np.argmin(np.isnan(met_arr[si])) for si in range(len(met_arr))]
        while len(out_met_arr) < num_scenes:
            # s0 always valid
            out_met_arr.append(met_arr[0][sc_cnt[0]])
            # find how many scenes were valid in this batch (next non-nan index)
            if sc_cnt[0] + 1 >= len(met_arr[0]):
                # we're at the end
                continue
            nan_mask = np.isnan(met_arr[0][sc_cnt[0]+1:])
            if np.sum(nan_mask) == len(nan_mask):
                # all valid
                num_valid = len(nan_mask)
            else:
                num_valid = np.argmin(nan_mask)
            sc_cnt[0] += num_valid + 1
            # collect other valid in batch
            for si in range(num_valid):
                cur_s = si + 1
                out_met_arr.append(met_arr[cur_s][sc_cnt[cur_s]])
                sc_cnt[cur_s] += num_valid + 1 - si

        assert len(out_met_arr) == num_scenes

        scene_results[guide_met] = out_met_arr
        agg_results[guide_met] = np.mean(out_met_arr)
        print("{} = {}".format(guide_met, agg_results[guide_met]))

    print("num_scenes: {}".format(num_scenes))

    # histogram of trajectory stats like vel, accel, jerk
    compute_and_save_hist(os.path.join(results_dir, "data.hdf5"), eval_out_dir)
    # compute hist dist to GT if given
    if gt_hist_path is not None:
        hjson = json.load(open(os.path.join(eval_out_dir, 'hist_stats.json'), "r"))
        gt_hjson = json.load(open(gt_hist_path, "r"))
        print('Computing EMD stats...')
        for k in gt_hjson["stats"]:
            agg_results["emd_{}".format(k)] = calc_hist_distance(
                hist1=np.array(gt_hjson["stats"][k]),
                hist2=np.array(hjson["stats"][k]),
                bin_edges=np.array(gt_hjson["ticks"][k][1:])
            )

    # stats to save
    all_stat_names = copy(SAVE_STAT_NAMES)
    if len(COLLECT_SAVE_STAT_NAMES) > 0:
        for collect_name in COLLECT_SAVE_STAT_NAMES:
            add_stat_names = [k for k in agg_results.keys() if collect_name in k]
            all_stat_names += add_stat_names

    # save csv of per_scene
    with open(os.path.join(eval_out_dir, 'results_per_scene.csv'), 'w') as f:
        csvwrite = csv.writer(f)
        csvwrite.writerow(['scene'] + all_stat_names)
        for sidx, scene in enumerate(scene_results["names"]):
            currow = [scene] + [scene_results[k][sidx] if k in scene_results else np.nan for k in all_stat_names]
            csvwrite.writerow(currow)

    # save agg csv
    with open(os.path.join(eval_out_dir, 'results_agg.csv'), 'w') as f:
        csvwrite = csv.writer(f)
        csvwrite.writerow(all_stat_names)
        currow = [agg_results[k] if k in agg_results else np.nan for k in all_stat_names]
        csvwrite.writerow(currow)

    return agg_results

def compute_and_save_hist(h5_path, out_path):
    """Compute histogram statistics for a run"""
    h5f = h5py.File(h5_path, "r")
    bins = {
        "velocity": torch.linspace(0, HIST_VEL_MAX, HIST_NUM_BINS+1),
        "lon_accel": torch.linspace(-HIST_ACCEL_MAX, HIST_ACCEL_MAX, HIST_NUM_BINS+1),
        "lat_accel": torch.linspace(-HIST_ACCEL_MAX, HIST_ACCEL_MAX, HIST_NUM_BINS+1),
        "jerk": torch.linspace(-HIST_JERK_MAX, HIST_JERK_MAX, HIST_NUM_BINS+1),
    }

    sim_stats = dict()
    ticks = None

    dt_ratio = int(math.ceil(STAT_DT / SIM_DT))
    for i, scene_index in enumerate(h5f.keys()):
        scene_data = h5f[scene_index]
        sim_pos = scene_data["centroid"]
        sim_yaw = scene_data["yaw"][:][:, :, None]
        sim_pos = sim_pos[:,::dt_ratio]
        sim_yaw = sim_yaw[:,::dt_ratio]

        num_agents = sim_pos.shape[0]
        sim = calc_stats(positions=torch.Tensor(sim_pos),
                         heading=torch.Tensor(sim_yaw),
                         dt=STAT_DT,
                         bins=bins)

        for k in sim:
            if k not in sim_stats:
                sim_stats[k] = sim[k].hist.long()
            else:
                sim_stats[k] += sim[k].hist.long()

        if ticks is None or k not in ticks:
            if ticks is None:
                ticks = dict()
            for k in sim:
                ticks[k] = sim[k].bin_edges

    for k in sim_stats:
        # normalize by total count to proper distrib
        sim_stats[k] = TensorUtils.to_numpy(sim_stats[k] / torch.sum(sim_stats[k])).tolist()
    for k in ticks:
        ticks[k] = TensorUtils.to_numpy(ticks[k]).tolist()

    results_path = out_path
    output_file = os.path.join(results_path, "hist_stats.json")
    json.dump({"stats": sim_stats, "ticks": ticks}, open(output_file, "w+"), indent=4)
    print("results dumped to {}".format(output_file))

    # visualize
    viz_path = os.path.join(results_path, "hist_viz")
    if not os.path.exists(viz_path):
        os.makedirs(viz_path)
    for k in sim_stats:
        fig = plt.figure()
        plt.bar(ticks[k][:-1], sim_stats[k], width=np.diff(ticks[k]).mean(), align='edge')
        plt.title(k)
        plt.ylabel('count')
        plt.xlabel(k)
        plt.savefig(os.path.join(viz_path, k + '.jpg'))
        plt.close(fig)

def calc_hist_distance(hist1, hist2, bin_edges):
    from pyemd import emd
    bins = np.array(bin_edges)
    bins_dist = np.abs(bins[:, None] - bins[None, :])
    hist_dist = emd(hist1, hist2, bins_dist)
    return hist_dist

def parse(args):
    all_results = []
    if args.results_set is not None:
        all_results = sorted(glob.glob(os.path.join(args.results_set, '*')))
    elif args.results_dir is not None:
        all_results = [args.results_dir]
    else:
        raise
    print(all_results)

    result_names = [cur_res.split('/')[-1] for cur_res in all_results]
    print(result_names)

    agg_res_list = []
    for result_path in all_results:
        agg_res = parse_single_result(result_path, args.gt_hist)
        agg_res_list.append(agg_res)
        
    if args.results_set is not None:
        all_stat_names = copy(SAVE_STAT_NAMES)
        if len(COLLECT_SAVE_STAT_NAMES) > 0:
            for collect_name in COLLECT_SAVE_STAT_NAMES:
                add_stat_names = [k for k in agg_res_list[0].keys() if collect_name in k]
                all_stat_names += add_stat_names
        # save agg csv
        with open(os.path.join(args.results_set, 'results_agg_set.csv'), 'w') as f:
            csvwrite = csv.writer(f)
            csvwrite.writerow(['eval_name'] + all_stat_names)
            for res_name, res_agg in zip(result_names, agg_res_list):
                currow = [res_agg[k] if k in res_agg else np.nan for k in all_stat_names]
                csvwrite.writerow([res_name] + currow)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="A directory of results files (including config.json and stats.json)"
    )

    parser.add_argument(
        "--results_set",
        type=str,
        default=None,
        help="A directory of directories where each contained directory is a results_dir to evaluate"
    )

    parser.add_argument(
        "--gt_hist",
        type=str,
        default=None,
        help="Path to histogram stats for GT data if wanting to compute EMD metrics"
    )

    args = parser.parse_args()

    parse(args)