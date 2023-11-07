import os
import numpy as np
from collections import defaultdict
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tbsim.configs.base import TrainConfig
from tbsim.utils.trajdata_utils import TRAJDATA_AGENT_TYPE_MAP

from trajdata import UnifiedDataset

import gc

class PassUnifiedDataModule(pl.LightningDataModule):
    """
    Pass-through config options to unified data loader.
    This is a more general version of the above UnifiedDataModule which 
    only supports any dataset available through trajdata.
    """
    def __init__(self, data_config, train_config: TrainConfig):
        super(PassUnifiedDataModule, self).__init__()
        self._data_config = data_config
        self._train_config = train_config
        self.train_dataset = None
        self.valid_dataset = None
        self.num_sem_layers = None

    @property
    def modality_shapes(self):
        """
        Returns the expected shape of combined rasterized layers
        (semantic + traj history + current)
        """
        # num_history + current
        hist_layer_size = self._data_config.history_num_frames + 1 if self._data_config.raster_include_hist \
                            else 0
        return dict(
            image=(self.num_sem_layers + hist_layer_size,  # semantic map
                   self._data_config.raster_size,
                   self._data_config.raster_size)
        )

    def setup(self, stage=None):
        data_cfg = self._data_config
        future_sec = data_cfg.future_num_frames * data_cfg.step_time
        history_sec = data_cfg.history_num_frames * data_cfg.step_time
        neighbor_distance = data_cfg.max_agents_distance
        agent_only_types = [TRAJDATA_AGENT_TYPE_MAP[cur_type] for cur_type in data_cfg.trajdata_only_types]
        agent_predict_types = None
        if data_cfg.trajdata_predict_types is not None:
            agent_predict_types = [TRAJDATA_AGENT_TYPE_MAP[cur_type] for cur_type in data_cfg.trajdata_predict_types]

        kwargs = dict(
            cache_location=data_cfg.trajdata_cache_location,
            desired_data=data_cfg.trajdata_source_train,
            desired_dt=data_cfg.step_time,
            future_sec=(future_sec, future_sec),
            history_sec=(history_sec, history_sec),
            data_dirs=data_cfg.trajdata_data_dirs,
            only_types=agent_only_types,
            only_predict=agent_predict_types,
            agent_interaction_distances=defaultdict(lambda: neighbor_distance),
            incl_map=data_cfg.trajdata_incl_map,
            map_params={
                "px_per_m": int(1 / data_cfg.pixel_size),
                "map_size_px": data_cfg.raster_size,
                "return_rgb": False,
                "offset_frac_xy": data_cfg.raster_center,
                "no_map_fill_value": data_cfg.no_map_fill_value,
            },
            centric=data_cfg.trajdata_centric,
            scene_description_contains=data_cfg.trajdata_scene_desc_contains,
            standardize_data=data_cfg.trajdata_standardize_data,
            verbose=True,
            num_workers=os.cpu_count(),
            rebuild_cache=data_cfg.trajdata_rebuild_cache,
            rebuild_maps=data_cfg.trajdata_rebuild_cache,
        )
        print(kwargs)
        self.train_dataset = UnifiedDataset(**kwargs)

        kwargs["desired_data"] = data_cfg.trajdata_source_valid
        self.valid_dataset = UnifiedDataset(**kwargs)

        # set modality shape based on input
        self.num_sem_layers = 0 if not data_cfg.trajdata_incl_map else data_cfg.num_sem_layers

        gc.collect()

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self._train_config.training.batch_size,
            num_workers=self._train_config.training.num_data_workers,
            drop_last=True,
            collate_fn=self.train_dataset.get_collate_fn(return_dict=True),
            persistent_workers=False
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            shuffle=True, # since pytorch lightning only evals a subset of val on each epoch, shuffle
            batch_size=self._train_config.validation.batch_size,
            num_workers=self._train_config.validation.num_data_workers,
            drop_last=True,
            collate_fn=self.valid_dataset.get_collate_fn(return_dict=True),
            persistent_workers=False
        )

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass