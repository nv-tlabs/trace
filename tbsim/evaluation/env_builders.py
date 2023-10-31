from collections import defaultdict

from trajdata import UnifiedDataset

from tbsim.configs.eval_config import EvaluationConfig
from tbsim.configs.base import ExperimentConfig
from tbsim.envs.env_trajdata import EnvUnifiedSimulation
from tbsim.utils.config_utils import translate_pass_trajdata_cfg
from tbsim.utils.trajdata_utils import TRAJDATA_AGENT_TYPE_MAP
import tbsim.envs.env_metrics as EnvMetrics

class EnvironmentBuilder(object):
    """Builds an simulation environment for evaluation."""
    def __init__(self, eval_config: EvaluationConfig, exp_config: ExperimentConfig, device):
        self.eval_cfg = eval_config
        self.exp_cfg = exp_config
        self.device = device

    def _get_analytical_metrics(self):
        metrics = dict(
            all_off_road_rate=EnvMetrics.OffRoadRate(),
            all_disk_off_road_rate=EnvMetrics.DiskOffRoadRate(),
            all_sem_layer_rate=EnvMetrics.SemLayerRate(),
            all_collision_rate=EnvMetrics.CollisionRate(),
            all_disk_collision_rate=EnvMetrics.DiskCollisionRate(),
            agents_collision_rate=EnvMetrics.CollisionRate(),
            all_failure=EnvMetrics.CriticalFailure(num_offroad_frames=2),
            all_comfort=EnvMetrics.Comfort(sim_dt=self.exp_cfg.algo.step_time, stat_dt=0.5),
        )
        return metrics

    def get_env(self):
        raise NotImplementedError

class EnvUnifiedBuilder(EnvironmentBuilder):
    def get_env(self):
        exp_cfg = self.exp_cfg.clone()
        exp_cfg.unlock()
        exp_cfg.env.simulation.num_simulation_steps = self.eval_cfg.num_simulation_steps
        exp_cfg.env.simulation.start_frame_index = exp_cfg.algo.history_num_frames + 1
        exp_cfg.lock()

        # the config used at training time
        data_cfg = translate_pass_trajdata_cfg(exp_cfg)

        future_sec = data_cfg.future_num_frames * data_cfg.step_time
        history_sec = data_cfg.history_num_frames * data_cfg.step_time
        neighbor_distance = data_cfg.max_agents_distance
        agent_only_types = [TRAJDATA_AGENT_TYPE_MAP[cur_type] for cur_type in data_cfg.trajdata_only_types]
        agent_predict_types = None
        if data_cfg.trajdata_predict_types is not None:
            agent_predict_types = [TRAJDATA_AGENT_TYPE_MAP[cur_type] for cur_type in data_cfg.trajdata_predict_types]

        kwargs = dict(
            cache_location=self.eval_cfg.trajdata_cache_location,
            desired_data=self.eval_cfg.trajdata_source_test,
            desired_dt=data_cfg.step_time,
            future_sec=(future_sec, future_sec),
            history_sec=(history_sec, history_sec),
            data_dirs=self.eval_cfg.trajdata_data_dirs,
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
            num_workers=0, #os.cpu_count(),
            rebuild_cache=self.eval_cfg.trajdata_rebuild_cache,
            rebuild_maps=self.eval_cfg.trajdata_rebuild_cache,
        )

        env_dataset = UnifiedDataset(**kwargs)

        metrics = dict()
        if self.eval_cfg.metrics.compute_analytical_metrics:
            metrics.update(self._get_analytical_metrics())

        # if we don't have a map, can't compute map-based metrics
        if not data_cfg.trajdata_incl_map:
            metrics.pop("all_off_road_rate", None)
            metrics.pop("all_sem_layer_rate", None)
            metrics.pop("all_failure", None)

        env = EnvUnifiedSimulation(
            exp_cfg.env,
            dataset=env_dataset,
            seed=self.eval_cfg.seed,
            num_scenes=self.eval_cfg.num_scenes_per_batch,
            prediction_only=False,
            metrics=metrics
        )

        return env