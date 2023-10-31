from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from trajdata import AgentBatch, AgentType, UnifiedDataset
from trajdata.visualization.vis import plot_agent_batch


def main():

    dataset = UnifiedDataset(
        desired_data=["orca_maps-train"], #, "orca_no_maps-train"],
        centric="agent",
        desired_dt=0.1,
        history_sec=(3.0, 3.0),
        future_sec=(5.0, 5.0),
        only_types=[AgentType.PEDESTRIAN],
        agent_interaction_distances=defaultdict(lambda: 15.0),
        incl_robot_future=False,
        incl_map=True,
        map_params={
            "px_per_m": 12, 
            "map_size_px": 224,
            "offset_frac_xy": (-0.5, 0.0),
            "return_rgb" : False,
            "no_map_fill_value" : -1.0
        },
        num_workers=0,
        verbose=True,
        data_dirs={
            "orca_maps" : "../datasets/orca_sim",
            "orca_no_maps" : "../datasets/orca_sim",
        },
        cache_location="~/.unified_data_cache",
        rebuild_cache=False,
        rebuild_maps=False,
        standardize_data=True,
    )

    # # NOTE: uncomment to run through nuscenes training data
    # dataset = UnifiedDataset(
    #     desired_data=["nusc_trainval-train"],
    #     centric="agent",
    #     desired_dt=0.1,
    #     history_sec=(3.0, 3.0),
    #     future_sec=(5.2, 5.2),
    #     only_types=[AgentType.PEDESTRIAN, AgentType.VEHICLE, AgentType.MOTORCYCLE, AgentType.BICYCLE],
    #     only_predict=[AgentType.PEDESTRIAN],
    #     agent_interaction_distances=defaultdict(lambda: 15.0),
    #     incl_robot_future=False,
    #     incl_map=True,
    #     map_params={
    #         "px_per_m": 12, 
    #         "map_size_px": 224,
    #         "offset_frac_xy": (-0.5, 0.0),
    #         "return_rgb" : False,
    #         "no_map_fill_value" : -1.0
    #     },
    #     num_workers=3,
    #     verbose=True,
    #     data_dirs={ 
    #         "nusc_mini" : "../datasets/nuscenes",
    #         "nusc_trainval": "../datasets/nuscenes",
    #         "nusc_test" : "../datasets/nuscenes",
    #     },
    #     cache_location="~/.unified_data_cache",
    #     rebuild_cache=False,
    #     rebuild_maps=False,
    # )

    print(f"# Data Samples: {len(dataset):,}")

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=dataset.get_collate_fn(),
        num_workers=4,
    )

    batch: AgentBatch
    for batch in tqdm(dataloader):
        print(batch.scene_ids)

        plot_agent_batch(batch, batch_idx=0, rgb_idx_groups=([1], [0], [1])) # NOTE: for ORCA
        # plot_agent_batch(batch, batch_idx=0, rgb_idx_groups=([0, 1, 2], [3, 4], [5, 6])) # NOTE: for nuscenes


if __name__ == "__main__":
    main()
