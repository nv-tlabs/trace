"""
Adapted from https://github.com/lyft/l5kit/blob/master/l5kit/l5kit/evaluation/metrics.py
"""


from typing import Callable
import torch
import numpy as np

from tbsim.utils.geometry_utils import (
    get_box_world_coords,
)

metric_signature = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
]


def _assert_shapes(
    ground_truth: np.ndarray,
    pred: np.ndarray,
    confidences: np.ndarray,
    avails: np.ndarray,
) -> None:
    """
    Check the shapes of args required by metrics
    Args:
        ground_truth (np.ndarray): array of shape (batch)x(timesteps)x(2D coords)
        pred (np.ndarray): array of shape (batch)x(modes)x(timesteps)x(2D coords)
        confidences (np.ndarray): array of shape (batch)x(modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (batch)x(timesteps) with the availability for each gt timesteps
    Returns:
    """
    assert (
        len(pred.shape) == 4
    ), f"expected 3D (BxMxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert ground_truth.shape == (
        batch_size,
        future_len,
        num_coords,
    ), f"expected 2D (Batch x Time x Coords) array for gt, got {ground_truth.shape}"
    assert confidences.shape == (
        batch_size,
        num_modes,
    ), f"expected 2D (Batch x Modes) array for confidences, got {confidences.shape}"

    assert np.allclose(np.sum(confidences, axis=1), 1), "confidences should sum to 1"
    assert avails.shape == (
        batch_size,
        future_len,
    ), f"expected 1D (Time) array for avails, got {avails.shape}"
    # assert all data are valid
    assert np.isfinite(pred).all(), "invalid value found in pred"
    assert np.isfinite(ground_truth).all(), "invalid value found in gt"
    assert np.isfinite(confidences).all(), "invalid value found in confidences"
    assert np.isfinite(avails).all(), "invalid value found in avails"

def batch_average_displacement_error(
    ground_truth: np.ndarray,
    pred: np.ndarray,
    confidences: np.ndarray,
    avails: np.ndarray,
    mode: str = "mean",
) -> np.ndarray:
    """
    Returns the average displacement error (ADE), which is the average displacement over all timesteps.
    During calculation, confidences are ignored, and two modes are available:
        - oracle: only consider the best hypothesis
        - mean: average over all hypotheses
    Args:
        ground_truth (np.ndarray): array of shape (batch)x(time)x(2D coords)
        pred (np.ndarray): array of shape (batch)x(modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (batch)x(modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (batch)x(time) with the availability for each gt timestep
        mode (str): calculation mode - options are 'mean' (average over hypotheses) and 'oracle' (use best hypotheses)
    Returns:
        np.ndarray: average displacement error (ADE) of the batch, an array of float numbers
    """
    _assert_shapes(ground_truth, pred, confidences, avails)

    ground_truth = np.expand_dims(ground_truth, 1)  # add modes
    avails = avails[:, np.newaxis, :, np.newaxis]  # add modes and cords

    error = np.sum(
        ((ground_truth - pred) * avails) ** 2, axis=-1
    )  # reduce coords and use availability
    error = error ** 0.5  # calculate root of error (= L2 norm)
    error = np.mean(error, axis=-1)  # average over timesteps
    if mode == "oracle":
        error = np.min(error, axis=1)  # use best hypothesis
    elif mode == "mean":
        error = np.sum(error*confidences, axis=1).mean()  # average over hypotheses
    else:
        raise ValueError(f"mode: {mode} not valid")

    return error

def batch_final_displacement_error(
    ground_truth: np.ndarray,
    pred: np.ndarray,
    confidences: np.ndarray,
    avails: np.ndarray,
    mode: str = "mean",
) -> np.ndarray:
    """
    Returns the final displacement error (FDE), which is the displacement calculated at the last timestep.
    During calculation, confidences are ignored, and two modes are available:
        - oracle: only consider the best hypothesis
        - mean: average over all hypotheses
    Args:
        ground_truth (np.ndarray): array of shape (batch)x(time)x(2D coords)
        pred (np.ndarray): array of shape (batch)x(modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (batch)x(modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (batch)x(time) with the availability for each gt timestep
        mode (str): calculation mode - options are 'mean' (average over hypotheses) and 'oracle' (use best hypotheses)
    Returns:
        np.ndarray: final displacement error (FDE) of the batch, an array of float numbers
    """
    _assert_shapes(ground_truth, pred, confidences, avails)
    inds = np.arange(0, pred.shape[2])
    inds = (avails > 0) * inds  # [B, (A), T] arange indices with unavailable indices set to 0
    last_inds = inds.max(axis=-1)
    last_inds = np.tile(last_inds[:, np.newaxis, np.newaxis],(1,pred.shape[1],1))
    ground_truth = np.expand_dims(ground_truth, 1)  # add modes
    avails = avails[:, np.newaxis, :, np.newaxis]  # add modes and cords
    
    
    error = np.sum(
        ((ground_truth - pred) * avails) ** 2, axis=-1
    )  # reduce coords and use availability
    error = error ** 0.5  # calculate root of error (= L2 norm)

    # error = error[:, :, -1]  # use last timestep
    error = np.take_along_axis(error,last_inds,axis=2).squeeze(-1)
    if mode == "oracle":
        error = np.min(error, axis=-1)  # use best hypothesis
    elif mode == "mean":
        error = np.mean(error, axis=-1)  # average over hypotheses
    else:
        raise ValueError(f"mode: {mode} not valid")

    return error

def batch_average_diversity(
    ground_truth: np.ndarray,
    pred: np.ndarray,
    confidences: np.ndarray,
    avails: np.ndarray,
    mode: str = "max",
) -> np.ndarray:
    """
    Compute the distance among trajectory samples averaged across time steps
    Args:
        ground_truth (np.ndarray): array of shape (batch)x(time)x(2D coords)
        pred (np.ndarray): array of shape (batch)x(modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (batch)x(modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (batch)x(time) with the availability for each gt timestep
        mode (str): calculation mode: option are "mean" (average distance) and "max" (distance between
            the two most distinctive samples).
    Returns:
        np.ndarray: average displacement error (ADE) of the batch, an array of float numbers
    """
    _assert_shapes(ground_truth, pred, confidences, avails)
    # compute pairwise distances
    error = np.linalg.norm(
        pred[:, np.newaxis, :] - pred[:, :, np.newaxis], axis=-1
    )  # [B, M, M, T]
    error = np.mean(error, axis=-1)  # average over timesteps
    error = error.reshape([error.shape[0], -1])  # [B, M * M]
    if mode == "max":
        error = np.max(error, axis=-1)
    elif mode == "mean":
        error = np.mean(error, axis=-1)
    else:
        raise ValueError(f"mode: {mode} not valid")

    return error

def batch_final_diversity(
    ground_truth: np.ndarray,
    pred: np.ndarray,
    confidences: np.ndarray,
    avails: np.ndarray,
    mode: str = "max",
) -> np.ndarray:
    """
    Compute the distance among trajectory samples at the last timestep
    Args:
        ground_truth (np.ndarray): array of shape (batch)x(time)x(2D coords)
        pred (np.ndarray): array of shape (batch)x(modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (batch)x(modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (batch)x(time) with the availability for each gt timestep
        mode (str): calculation mode: option are "mean" (average distance) and "max" (distance between
            the two most distinctive samples).
    Returns:
        np.ndarray: average displacement error (ADE) of the batch, an array of float numbers
    """
    _assert_shapes(ground_truth, pred, confidences, avails)
    # compute pairwise distances at the last time step
    pred = pred[..., -1]
    error = np.linalg.norm(
        pred[:, np.newaxis, :] - pred[:, :, np.newaxis], axis=-1
    )  # [B, M, M]
    error = error.reshape([error.shape[0], -1])  # [B, M * M]
    if mode == "max":
        error = np.max(error, axis=-1)
    elif mode == "mean":
        error = np.mean(error, axis=-1)
    else:
        raise ValueError(f"mode: {mode} not valid")

    return error

def batch_detect_off_road(positions, drivable_region_map):
    """
    Compute whether the given positions are out of drivable region
    Args:
        positions (torch.Tensor): a position (x, y) in rasterized frame [B, ..., 2]
        drivable_region_map (torch.Tensor): binary drivable region maps [B, H, W]

    Returns:
        off_road (torch.Tensor): whether each given position is off-road [B, ...]
    """
    assert positions.shape[0] == drivable_region_map.shape[0]
    assert drivable_region_map.ndim == 3
    b, h, w = drivable_region_map.shape
    # project any off-map pixels to edges so valid
    positions[..., 0] = torch.clamp(positions[..., 0], 0, w-1)
    positions[..., 1] = torch.clamp(positions[..., 1], 0, h-1)
    positions_flat = positions[..., 1].long() * w + positions[..., 0].long()
    if positions_flat.ndim == 1:
        positions_flat = positions_flat[:, None]
    drivable = torch.gather(
        drivable_region_map.flatten(start_dim=1),  # [B, H * W]
        dim=1,
        index=positions_flat.long().flatten(start_dim=1),  # [B, (all trailing dim flattened)]
    ).reshape(*positions.shape[:-1])
    return 1 - drivable.float()

def batch_detect_off_road_boxes(positions, yaws, extents, drivable_region_map):
    """
    Compute whether boxes specified by (@positions, @yaws, and @extents) are out of drivable region.
    A box is considered off-road if at least one of its corners are out of drivable region
    Args:
        positions (torch.Tensor): agent centroid (x, y) in rasterized frame [B, ..., 2]
        yaws (torch.Tensor): agent yaws in rasterized frame [B, ..., 1]
        extents (torch.Tensor): agent extents in RASTERIZED FRAME scale [B, ..., 2]
        drivable_region_map (torch.Tensor): binary drivable region maps [B, H, W]

    Returns:
        box_off_road (torch.Tensor): whether each given box is off-road [B, ...]
    """
    B, H, W = drivable_region_map.size()
    boxes = get_box_world_coords(positions, yaws, extents)  # [B, ..., 4, 2]
    # project any off-map pixels to edges so valid
    boxes[..., 0] = torch.clamp(boxes[..., 0], 0, W-1)
    boxes[..., 1] = torch.clamp(boxes[..., 1], 0, H-1)
    off_road = batch_detect_off_road(boxes, drivable_region_map)  # [B, ..., 4]
    box_off_road = off_road.sum(dim=-1) > 0.5
    return box_off_road.float()


def batch_detect_off_road_disk(positions, extents, drivable_region_map):
    """
    Compute whether disks specified by (@positions, and @extents) are out of drivable region.
    Args:
        positions (torch.Tensor): agent centroid (x, y) in rasterized frame [B, 2]
        extents (torch.Tensor): agent extents in RASTERIZED FRAME scale [B, 2]
        drivable_region_map (torch.Tensor): binary drivable region maps [B, H, W]

    Returns:
        box_off_road (torch.Tensor): whether each given box is off-road [B, ...]
    """
    B, H, W = drivable_region_map.size()
    
    # sample along radii to extent for each agent
    ntheta_samp = 13 # sample at 13 angles
    nrad_samp = 4 # sample at 4 radii
    disk_samp = torch.linspace(0, 2*np.pi, ntheta_samp)
    disk_samp = torch.stack([torch.cos(disk_samp), torch.sin(disk_samp)], dim=1) # 13 x 2
    disk_samp = disk_samp[None,None].expand((1, nrad_samp, ntheta_samp, 2))
    
    agt_rad = torch.amin(extents, dim=-1) / 2.0
    rad_len = torch.stack([torch.linspace(0, agt_rad[ai], nrad_samp+1)[1:] for ai in range(agt_rad.size(0))], dim=0)[:,:,None,None]
    
    disk_samp = disk_samp * rad_len
    disk_samp = disk_samp.reshape((rad_len.size(0), -1, 2)) # B x 52 x 2
    # move to positions
    disk_samp = positions.unsqueeze(1) + disk_samp

    # project any off-map pixels to edges so valid
    disk_samp[..., 0] = torch.clamp(disk_samp[..., 0], 0, W-1)
    disk_samp[..., 1] = torch.clamp(disk_samp[..., 1], 0, H-1)
    off_road = batch_detect_off_road(disk_samp, drivable_region_map)  # [B, ..., 52]
    box_off_road = off_road.sum(dim=-1) > 0.5
    return box_off_road.float()
