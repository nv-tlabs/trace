from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from trajdata.utils import arr_utils


class SimStatistic:
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, scene_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()


class VelocityHistogram(SimStatistic):
    def __init__(self, bins: List[int]) -> None:
        super().__init__("vel_hist")
        self.bins = bins

    def __call__(self, scene_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        velocities: np.ndarray = np.linalg.norm(scene_df[["vx", "vy"]], axis=1)

        return np.histogram(velocities, bins=self.bins)


class LongitudinalAccHistogram(SimStatistic):
    def __init__(self, bins: List[int]) -> None:
        super().__init__("lon_acc_hist")
        self.bins = bins

    def __call__(self, scene_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        accels: np.ndarray = np.linalg.norm(scene_df[["ax", "ay"]], axis=1)
        lon_accels: np.ndarray = accels * np.cos(scene_df["heading"])

        return np.histogram(lon_accels, bins=self.bins)


class LateralAccHistogram(SimStatistic):
    def __init__(self, bins: List[int]) -> None:
        super().__init__("lat_acc_hist")
        self.bins = bins

    def __call__(self, scene_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        accels: np.ndarray = np.linalg.norm(scene_df[["ax", "ay"]], axis=1)
        lat_accels: np.ndarray = accels * np.sin(scene_df["heading"])

        return np.histogram(lat_accels, bins=self.bins)


class JerkHistogram(SimStatistic):
    def __init__(self, bins: List[int], dt: float) -> None:
        super().__init__("jerk_hist")
        self.bins = bins
        self.dt = dt

    def __call__(self, scene_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        accels: np.ndarray = np.linalg.norm(scene_df[["ax", "ay"]], axis=1)
        jerk: np.ndarray = (
            arr_utils.agent_aware_diff(accels, scene_df.index.get_level_values(0))
            / self.dt
        )

        return np.histogram(jerk, bins=self.bins)


def calc_stats(
    positions: Tensor, heading: Tensor, dt: float, bins: Dict[str, Tensor]
) -> Dict[str, Tensor]:
    """Calculate scene statistics for a simulated scene.

    Args:
        positions (Tensor): N x T x 2 tensor of agent positions (in world coordinates).
        heading (Tensor): N x T x 1 tensor of agent headings (in world coordinates).
        dt (float): The data's delta timestep.
        bins (Dict[str, Tensor]): A mapping from statistic name to a Tensor of bin edges.

    Returns:
        Dict[str, Tensor]: A mapping of value names to histograms.
    """

    out_dict = {}

    velocity: Tensor = (
        torch.diff(positions, dim=1) / dt
    )
    velocity_norm: Tensor = torch.linalg.vector_norm(velocity, dim=-1)

    if torch.sum(~torch.isnan(velocity_norm)) > 0:
        out_dict["velocity"] = torch.histogram(velocity_norm[~torch.isnan(velocity_norm)], bins["velocity"])

    if velocity.size(1) < 2:
        return out_dict

    accel: Tensor = (
        torch.diff(velocity, dim=1) / dt
    )
    accel_norm: Tensor = torch.linalg.vector_norm(accel, dim=-1)
    T = accel_norm.size(1)
    lon_acc: Tensor = accel_norm * torch.cos(heading[:,:T].squeeze(-1))
    lat_acc: Tensor = accel_norm * torch.sin(heading[:,:T].squeeze(-1))

    if torch.sum(~torch.isnan(lon_acc)) > 0:
        out_dict["lon_accel"] = torch.histogram(lon_acc[~torch.isnan(lon_acc)], bins["lon_accel"])
        out_dict["lat_accel"] = torch.histogram(lat_acc[~torch.isnan(lat_acc)], bins["lat_accel"])

    if accel_norm.size(1) < 2:
        return out_dict

    jerk: Tensor = (
        torch.diff(accel_norm, dim=1)
        / dt
    )

    if torch.sum(~torch.isnan(jerk)) > 0:
        out_dict["jerk"] = torch.histogram(jerk[~torch.isnan(jerk)], bins["jerk"])

    return out_dict
