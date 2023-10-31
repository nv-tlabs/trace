from collections import defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm

from trajdata import AgentBatch, AgentType, UnifiedDataset

import numpy as np
import torch
from torch import Tensor

def main():

    dt = 0.1
    dataset = UnifiedDataset(
        
        # NOTE: uncomment to compute stats for ORCA data
        # desired_data=["orca_maps-train", "orca_no_maps-train"],
        # NOTE: uncomment to compute stats for mixed ETH/UCY + nuScenes
        desired_data=["nusc_trainval-train", "eupeds_eth-train", "eupeds_hotel-train", "eupeds_univ-train", "eupeds_zara1-train", "eupeds_zara2-train"],
        # NOTE: uncomment to compute stats for ETH/UCY data only
        # desired_data=["eupeds_eth-train", "eupeds_hotel-train", "eupeds_univ-train", "eupeds_zara1-train", "eupeds_zara2-train"],
        # NOTE: uncomment to compute stats for nuScenes only
        # desired_data=["nusc_trainval-train"],

        centric="agent",
        desired_dt=dt,
        history_sec=(3.0, 3.0),
        future_sec=(5.2, 5.2),
        only_types=[AgentType.PEDESTRIAN],
        # only_types=[AgentType.PEDESTRIAN, AgentType.VEHICLE, AgentType.BICYCLE, AgentType.MOTORCYCLE],
        only_predict=[AgentType.PEDESTRIAN],
        agent_interaction_distances=defaultdict(lambda: 15.0),
        incl_robot_future=False,
        incl_map=False,
        map_params={
            "px_per_m": 12, 
            "map_size_px": 224,
            "offset_frac_xy": (-0.5, 0.0),
            "return_rgb" : False,
            "no_map_fill_value" : -1.0,
        },
        num_workers=0,
        verbose=True,
        data_dirs={
            "orca_maps" : "../datasets/orca_sim",
            "orca_no_maps" : "../datasets/orca_sim",
            "nusc_mini" : "../datasets/nuscenes",
            "nusc_trainval": "../datasets/nuscenes",
            "eupeds_eth" : "../datasets/eth_ucy",
            "eupeds_hotel" : "../datasets/eth_ucy",
            "eupeds_univ" : "../datasets/eth_ucy",
            "eupeds_zara1" : "../datasets/eth_ucy",
            "eupeds_zara2" : "../datasets/eth_ucy",
        },
        cache_location="~/.unified_data_cache",
        rebuild_cache=False,
        rebuild_maps=False,
        standardize_data=True,
    )

    print(f"# Data Samples: {len(dataset):,}")

    dataloader = DataLoader(
        dataset,
        batch_size=200,
        shuffle=False,
        collate_fn=dataset.get_collate_fn(),
        num_workers=8,
    )

    batch: AgentBatch
    compile_data = {
        'ego_fut' : [],
        'ego_hist' : [],
        'neighbor_hist' : []
    }
    for batch in tqdm(dataloader):
        # normalize over future traj
        past_traj: Tensor = batch.agent_hist.cuda()
        future_traj: Tensor = batch.agent_fut.cuda()

        hist_pos, hist_yaw, hist_speed, _ = trajdata2posyawspeed(past_traj, nan_to_zero=False)
        curr_speed = hist_speed[..., -1]

        fut_pos, fut_yaw, _, fut_mask = trajdata2posyawspeed(future_traj, nan_to_zero=False)

        traj_state = torch.cat(
                (fut_pos, fut_yaw), dim=2)

        traj_state_and_action = convert_state_to_state_and_action(traj_state, curr_speed, dt).reshape((-1, 6))

        # B*T x 6 where (x, y, vel, yaw, acc, yawvel)
        # print(traj_state_and_action.size())
        compile_data['ego_fut'].append(traj_state_and_action.cpu().numpy())

        # ego history
        ego_lw = batch.agent_hist_extent[:,:,:2].cuda()
        ego_hist_state = torch.cat((hist_pos, hist_speed.unsqueeze(-1), ego_lw), dim=-1).reshape((-1, 5))
        compile_data['ego_hist'].append(ego_hist_state.cpu().numpy())

        # neighbor history
        neigh_hist_pos, _, neigh_hist_speed, neigh_mask = trajdata2posyawspeed(batch.neigh_hist.cuda(), nan_to_zero=False)
        neigh_lw = batch.neigh_hist_extents[...,:2].cuda()
        neigh_state = torch.cat((neigh_hist_pos, neigh_hist_speed.unsqueeze(-1), neigh_lw), dim=-1)
        # only want steps from neighbors that are valid
        neigh_state = neigh_state[neigh_mask]
        compile_data['neighbor_hist'].append(neigh_state.cpu().numpy())


    val_labels = {
        'ego_fut' : [    'x',       ' y',       'vel',      'yaw',     'acc',    'yawvel' ],
        'ego_hist' : [    'x',        'y',       'vel',      'len',     'width'    ],
        'neighbor_hist' : [    'x',        'y',       'vel',      'len',     'width'    ]
    }
    for state_name, state_list in compile_data.items():
        print(state_name)
        all_states = np.concatenate(state_list, axis=0)
        print(all_states.shape)
        print(np.sum(np.isnan(all_states)))

        # import matplotlib
        # import matplotlib.pyplot as plt
        # for di, dname in enumerate(['x', 'y', 'vel', 'yaw', 'acc', 'yawvel']):
        #     fig = plt.figure()
        #     plt.hist((all_state_and_action[:,di] - np_mean[di]) / np_std[di], bins=100)
        #     plt.title(dname)
        #     plt.show()
        #     plt.close(fig)

        # remove outliers before computing final statistics
        print('Removing outliers...')
        print(np.median(all_states, axis=0, keepdims=True))
        d = np.abs(all_states - np.median(all_states, axis=0, keepdims=True))
        mdev = np.std(all_states, axis=0, keepdims=True, dtype=np.float64)
        print(mdev)
        s = d / mdev

        dev_thresh = 4.0
        all_states[s > dev_thresh] = np.nan # reject outide of N deviations from median
        print('after outlier removal:')
        print(np.sum(s > dev_thresh))
        print(np.sum(s > dev_thresh, axis=0))
        print(np.sum(s > dev_thresh) / (s.shape[0]*s.shape[1])) # removal rate

        out_mean = np.nanmean(all_states, axis=0, dtype=np.float64)
        out_std = np.nanstd(all_states, axis=0, dtype=np.float64)
        out_max = np.nanmax(all_states, axis=0)
        out_min = np.nanmin(all_states, axis=0)

        print('    '.join(val_labels[state_name]))
        out_fmt = ['( '] + ['%05f, ' for _ in val_labels[state_name]] + [' )']
        out_fmt = ''.join(out_fmt)
        print('out-mean')
        print(out_fmt % tuple(out_mean.tolist()))
        print('out-std')
        print(out_fmt % tuple(out_std.tolist()))
        print('out-max')
        print(out_fmt % tuple(out_max.tolist()))
        print('out-min')
        print(out_fmt % tuple(out_min.tolist()))

        # for di, dname in enumerate(['x', 'y', 'vel', 'yaw', 'acc', 'yawvel']):
        #     fig = plt.figure()
        #     plt.hist(s[:,di], bins=100)
        #     plt.title(dname)
        #     plt.show()
        #     plt.close(fig)

        # import matplotlib
        # import matplotlib.pyplot as plt
        # for di, dname in enumerate(['x', 'y', 'vel', 'yaw', 'acc', 'yawvel']):
        #     fig = plt.figure()
        #     plt.hist((all_states[:,di] - out_mean[di]) / out_std[di], bins=100)
        #     plt.title(dname)
        #     plt.show()
        #     plt.close(fig)

def trajdata2posyawspeed(state, nan_to_zero=True):
    """Converts trajdata's state format to pos, yaw, and speed. Set Nans to 0s"""
    
    if state.shape[-1] == 7:  # x, y, vx, vy, ax, ay, sin(heading), cos(heading)
        state = torch.cat((state[...,:6],torch.sin(state[...,6:7]),torch.cos(state[...,6:7])),-1)
    else:
        assert state.shape[-1] == 8
    pos = state[..., :2]
    yaw = torch.atan2(state[..., [-2]], state[..., [-1]])
    speed = torch.norm(state[..., 2:4], dim=-1)
    mask = torch.bitwise_not(torch.max(torch.isnan(state), dim=-1)[0])
    if nan_to_zero:
        pos[torch.bitwise_not(mask)] = 0.
        yaw[torch.bitwise_not(mask)] = 0.
        speed[torch.bitwise_not(mask)] = 0.
    return pos, yaw, speed, mask

def angle_diff(theta1, theta2):
    '''
    :param theta1: angle 1 (..., 1)
    :param theta2: angle 2 (..., 1)
    :return diff: smallest angle difference between angles (..., 1)
    '''
    period = 2*np.pi
    diff = (theta1 - theta2 + period / 2) % period - period / 2
    diff[diff > np.pi] = diff[diff > np.pi] - (2 * np.pi)
    return diff

def convert_state_to_state_and_action(traj_state, vel_init, dt):
    '''
    Infer vel and action (acc, yawvel) from state (x, y, yaw).
    Input:
        traj_state: (batch_size, num_steps, 3)
        vel_init: (batch_size,)
        dt: float
    Output:
        traj_state_and_action: (batch_size, num_steps, 6)
    '''
    target_pos = traj_state[:, :, :2]
    traj_yaw = traj_state[:, :, 2:]
    
    b = target_pos.size()[0]
    device = target_pos.get_device()

    # pre-pad with zero pos
    pos_init = torch.zeros(b, 1, 2, device=device)
    pos = torch.cat((pos_init, target_pos), dim=1)
    
    # pre-pad with zero pos
    yaw_init = torch.zeros(b, 1, 1, device=device) # data_batch["yaw"][:, None, None]
    yaw = torch.cat((yaw_init, traj_yaw), dim=1)

    # estimate speed from position and orientation
    vel_init = vel_init[:, None, None]
    vel = (pos[..., 1:, 0:1] - pos[..., :-1, 0:1]) / dt * torch.cos(
        yaw[..., 1:, :]
    ) + (pos[..., 1:, 1:2] - pos[..., :-1, 1:2]) / dt * torch.sin(
        yaw[..., 1:, :]
    )
    vel = torch.cat((vel_init, vel), dim=1)
    
    # m/s^2
    acc = (vel[..., 1:, :] - vel[..., :-1, :]) / dt
    # rad/s
    yawdiff = angle_diff(yaw[..., 1:, :], yaw[..., :-1, :])
    yawvel = yawdiff / dt

    pos, yaw, vel = pos[..., 1:, :], yaw[..., 1:, :], vel[..., 1:, :]

    traj_state_and_action = torch.cat((pos, vel, yaw, acc, yawvel), dim=2)

    return traj_state_and_action

if __name__ == "__main__":
    main()
