from typing import Tuple

import numpy as np
import torch

from shapely.geometry import LineString, Polygon

from enum import IntEnum

def transform_yaw(yaw, tf_mat):
    '''
    - yaw : (B)
    - tf_mat : (B, 3, 3) matrix to transform yaw by
    '''
    yaw = yaw[:,None]
    hvec = torch.cat([torch.cos(yaw), torch.sin(yaw)], dim=-1) # B x 2
    rot_mat = tf_mat[:,:2,:2].clone() # B x 2 x 2
    # rot part of mat may have scaling too
    # print(rot_mat)
    rot_mat[:,:,0] = rot_mat[:,:,0] / torch.norm(rot_mat[:,:,0], dim=-1, keepdim=True)
    rot_mat[:,:,1] = rot_mat[:,:,1] / torch.norm(rot_mat[:,:,1], dim=-1, keepdim=True)
    # print(rot_mat)
    rot_hvec = torch.matmul(rot_mat, hvec.unsqueeze(-1))[:,:,0] # B x 2
    # rot_hvec = rot_hvec / torch.norm(rot_hvec, dim=-1, keepdim=True)
    # print(rot_hvec)
    tf_yaw = torch.atan2(rot_hvec[:,1], rot_hvec[:,0]) # rot part of mat may have scaling too
    return tf_yaw


def get_box_agent_coords(pos, yaw, extent):
    corners = (torch.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * 0.5).to(pos.device) * (
        extent.unsqueeze(-2)
    )
    s = torch.sin(yaw).unsqueeze(-1)
    c = torch.cos(yaw).unsqueeze(-1)
    rotM = torch.cat((torch.cat((c, s), dim=-1), torch.cat((-s, c), dim=-1)), dim=-2)
    rotated_corners = (corners + pos.unsqueeze(-2)) @ rotM
    return rotated_corners


def get_box_world_coords(pos, yaw, extent):
    corners = (torch.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * 0.5).to(pos.device) * (
        extent.unsqueeze(-2)
    )
    s = torch.sin(yaw).unsqueeze(-1)
    c = torch.cos(yaw).unsqueeze(-1)
    rotM = torch.cat((torch.cat((c, s), dim=-1), torch.cat((-s, c), dim=-1)), dim=-2)
    rotated_corners = corners @ rotM + pos.unsqueeze(-2)
    return rotated_corners


def get_box_agent_coords_np(pos, yaw, extent):
    corners = (np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * 0.5) * (
        extent[..., None, :]
    )
    s = np.sin(yaw)[..., None]
    c = np.cos(yaw)[..., None]
    rotM = np.concatenate((np.concatenate((c, s), axis=-1), np.concatenate((-s, c), axis=-1)), axis=-2)
    rotated_corners = (corners + pos[..., None, :]) @ rotM
    return rotated_corners


def get_box_world_coords_np(pos, yaw, extent):
    corners = (np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * 0.5) * (
        extent[..., None, :]
    )
    s = np.sin(yaw)[..., None]
    c = np.cos(yaw)[..., None]
    rotM = np.concatenate((np.concatenate((c, s), axis=-1), np.concatenate((-s, c), axis=-1)), axis=-2)
    rotated_corners = corners @ rotM + pos[..., None, :]
    return rotated_corners


def get_upright_box(pos, extent):
    yaws = torch.zeros(*pos.shape[:-1], 1).to(pos.device)
    boxes = get_box_world_coords(pos, yaws, extent)
    upright_boxes = boxes[..., [0, 2], :]
    return upright_boxes


def batch_nd_transform_points(points, Mat):
    ndim = Mat.shape[-1] - 1
    Mat = Mat.transpose(-1, -2)
    return (points.unsqueeze(-2) @ Mat[..., :ndim, :ndim]).squeeze(-2) + Mat[
        ..., -1:, :ndim
    ].squeeze(-2)

def batch_nd_transform_points_np(points, Mat):
    ndim = Mat.shape[-1] - 1
    batch = list(range(Mat.ndim-2))+[Mat.ndim-1]+[Mat.ndim-2]
    Mat = np.transpose(Mat,batch)
    if points.ndim==Mat.ndim-1:
        return (points[...,np.newaxis,:] @ Mat[..., :ndim, :ndim]).squeeze(-2) + Mat[
            ..., -1:, :ndim
        ].squeeze(-2)
    elif points.ndim==Mat.ndim:
        return ((points[...,np.newaxis,:] @ Mat[...,np.newaxis, :ndim, :ndim]) + Mat[
            ...,np.newaxis, -1:, :ndim]).squeeze(-2)
    else:
        raise Exception("wrong shape")

#
# from https://github.com/woven-planet/l5kit/blob/master/l5kit/l5kit/geometry/transform.py#L73
#
def transform_points_tensor(
    points: torch.Tensor, transf_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Transform a set of 2D/3D points using the given transformation matrix.
    Assumes row major ordering of the input points. The transform function has 3 modes:
    - points (N, F), transf_matrix (F+1, F+1)
    all points are transformed using the matrix and the output points have shape (N, F).
    - points (B, N, F), transf_matrix (F+1, F+1)
    all sequences of points are transformed using the same matrix and the output points have shape (B, N, F).
    transf_matrix is broadcasted.
    - points (B, N, F), transf_matrix (B, F+1, F+1)
    each sequence of points is transformed using its own matrix and the output points have shape (B, N, F).
    Note this function assumes points.shape[-1] == matrix.shape[-1] - 1, which means that last
    rows in the matrices do not influence the final results.
    For 2D points only the first 2x3 parts of the matrices will be used.

    :param points: Input points of shape (N, F) or (B, N, F)
        with F = 2 or 3 depending on input points are 2D or 3D points.
    :param transf_matrix: Transformation matrix of shape (F+1, F+1) or (B, F+1, F+1) with F = 2 or 3.
    :return: Transformed points of shape (N, F) or (B, N, F) depending on the dimensions of the input points.
    """
    points_log = f" received points with shape {points.shape} "
    matrix_log = f" received matrices with shape {transf_matrix.shape} "

    assert points.ndim in [2, 3], f"points should have ndim in [2,3],{points_log}"
    assert transf_matrix.ndim in [
        2,
        3,
    ], f"matrix should have ndim in [2,3],{matrix_log}"
    assert (
        points.ndim >= transf_matrix.ndim
    ), f"points ndim should be >= than matrix,{points_log},{matrix_log}"

    points_feat = points.shape[-1]
    assert points_feat in [2, 3], f"last points dimension must be 2 or 3,{points_log}"
    assert (
        transf_matrix.shape[-1] == transf_matrix.shape[-2]
    ), f"matrix should be a square matrix,{matrix_log}"

    matrix_feat = transf_matrix.shape[-1]
    assert matrix_feat in [3, 4], f"last matrix dimension must be 3 or 4,{matrix_log}"
    assert (
        points_feat == matrix_feat - 1
    ), f"points last dim should be one less than matrix,{points_log},{matrix_log}"

    def _transform(points: torch.Tensor, transf_matrix: torch.Tensor) -> torch.Tensor:
        num_dims = transf_matrix.shape[-1] - 1
        transf_matrix = torch.permute(transf_matrix, (0, 2, 1))
        return (
            points @ transf_matrix[:, :num_dims, :num_dims]
            + transf_matrix[:, -1:, :num_dims]
        )

    if points.ndim == transf_matrix.ndim == 2:
        points = torch.unsqueeze(points, 0)
        transf_matrix = torch.unsqueeze(transf_matrix, 0)
        return _transform(points, transf_matrix)[0]

    elif points.ndim == transf_matrix.ndim == 3:
        return _transform(points, transf_matrix)

    elif points.ndim == 3 and transf_matrix.ndim == 2:
        transf_matrix = torch.unsqueeze(transf_matrix, 0)
        return _transform(points, transf_matrix)
    else:
        raise NotImplementedError(f"unsupported case!{points_log},{matrix_log}")
    
def transform_points(points: np.ndarray, transf_matrix: np.ndarray) -> np.ndarray:
    """
    Transform a set of 2D/3D points using the given transformation matrix.
    Assumes row major ordering of the input points. The transform function has 3 modes:
    - points (N, F), transf_matrix (F+1, F+1)
    all points are transformed using the matrix and the output points have shape (N, F).
    - points (B, N, F), transf_matrix (F+1, F+1)
    all sequences of points are transformed using the same matrix and the output points have shape (B, N, F).
    transf_matrix is broadcasted.
    - points (B, N, F), transf_matrix (B, F+1, F+1)
    each sequence of points is transformed using its own matrix and the output points have shape (B, N, F).
    Note this function assumes points.shape[-1] == matrix.shape[-1] - 1, which means that last
    rows in the matrices do not influence the final results.
    For 2D points only the first 2x3 parts of the matrices will be used.

    :param points: Input points of shape (N, F) or (B, N, F)
        with F = 2 or 3 depending on input points are 2D or 3D points.
    :param transf_matrix: Transformation matrix of shape (F+1, F+1) or (B, F+1, F+1) with F = 2 or 3.
    :return: Transformed points of shape (N, F) or (B, N, F) depending on the dimensions of the input points.
    """
    points_log = f" received points with shape {points.shape} "
    matrix_log = f" received matrices with shape {transf_matrix.shape} "

    assert points.ndim in [2, 3], f"points should have ndim in [2,3],{points_log}"
    assert transf_matrix.ndim in [2, 3], f"matrix should have ndim in [2,3],{matrix_log}"
    assert points.ndim >= transf_matrix.ndim, f"points ndim should be >= than matrix,{points_log},{matrix_log}"

    points_feat = points.shape[-1]
    assert points_feat in [2, 3], f"last points dimension must be 2 or 3,{points_log}"
    assert transf_matrix.shape[-1] == transf_matrix.shape[-2], f"matrix should be a square matrix,{matrix_log}"

    matrix_feat = transf_matrix.shape[-1]
    assert matrix_feat in [3, 4], f"last matrix dimension must be 3 or 4,{matrix_log}"
    assert points_feat == matrix_feat - 1, f"points last dim should be one less than matrix,{points_log},{matrix_log}"

    def _transform(points: np.ndarray, transf_matrix: np.ndarray) -> np.ndarray:
        num_dims = transf_matrix.shape[-1] - 1
        transf_matrix = np.transpose(transf_matrix, (0, 2, 1))
        return points @ transf_matrix[:, :num_dims, :num_dims] + transf_matrix[:, -1:, :num_dims]

    if points.ndim == transf_matrix.ndim == 2:
        points = np.expand_dims(points, 0)
        transf_matrix = np.expand_dims(transf_matrix, 0)
        return _transform(points, transf_matrix)[0]

    elif points.ndim == transf_matrix.ndim == 3:
        return _transform(points, transf_matrix)

    elif points.ndim == 3 and transf_matrix.ndim == 2:
        transf_matrix = np.expand_dims(transf_matrix, 0)
        return _transform(points, transf_matrix)
    else:
        raise NotImplementedError(f"unsupported case!{points_log},{matrix_log}")
    
# from https://github.com/woven-planet/l5kit/blob/master/l5kit/l5kit/planning/utils.py#L18C30-L18C30
def _get_bounding_box(centroid: np.ndarray, yaw: np.ndarray,
                      extent: np.ndarray,) -> Polygon:
    """This function will get a shapely Polygon representing the bounding box
    with an optional buffer around it.

    :param centroid: centroid of the agent
    :param yaw: the yaw of the agent
    :param extent: the extent of the agent
    :return: a shapely Polygon
    """
    x, y = centroid[0], centroid[1]
    sin, cos = np.sin(yaw), np.cos(yaw)
    width, length = extent[0] / 2, extent[1] / 2

    x1, y1 = (x + width * cos - length * sin, y + width * sin + length * cos)
    x2, y2 = (x + width * cos + length * sin, y + width * sin - length * cos)
    x3, y3 = (x - width * cos + length * sin, y - width * sin - length * cos)
    x4, y4 = (x - width * cos - length * sin, y - width * sin + length * cos)
    return Polygon([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

def _get_sides(bbox: Polygon) -> Tuple[LineString, LineString, LineString, LineString]:
    """This function will get the sides of a bounding box.

    :param bbox: the bounding box
    :return: a tuple with the four sides of the bounding box as LineString,
             representing front/rear/left/right.
    """
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox.exterior.coords[:-1]
    return (
        LineString([(x1, y1), (x2, y2)]),
        LineString([(x3, y3), (x4, y4)]),
        LineString([(x1, y1), (x4, y4)]),
        LineString([(x2, y2), (x3, y3)]),
    )

class CollisionType(IntEnum):
    """This enum defines the three types of collisions: front, rear and side."""
    FRONT = 0
    REAR = 1
    SIDE = 2


def detect_collision(
        ego_pos: np.ndarray,
        ego_yaw: np.ndarray,
        ego_extent: np.ndarray,
        other_pos: np.ndarray,
        other_yaw: np.ndarray,
        other_extent: np.ndarray,
):
    """
    Computes whether a collision occured between ego and any another agent.
    Also computes the type of collision: rear, front, or side.
    For this, we compute the intersection of ego's four sides with a target
    agent and measure the length of this intersection. A collision
    is classified into a class, if the corresponding length is maximal,
    i.e. a front collision exhibits the longest intersection with
    egos front edge.

    .. note:: please note that this funciton will stop upon finding the first
              colision, so it won't return all collisions but only the first
              one found.

    :param ego_pos: predicted centroid
    :param ego_yaw: predicted yaw
    :param ego_extent: predicted extent
    :param other_pos: target agents
    :return: None if not collision was found, and a tuple with the
             collision type and the agent track_id
    """
    ego_bbox = _get_bounding_box(centroid=ego_pos, yaw=ego_yaw, extent=ego_extent)
    
    for i in range(other_pos.shape[0]):
        agent_bbox = _get_bounding_box(other_pos[i], other_yaw[i], other_extent[i])
        if ego_bbox.intersects(agent_bbox):
            front_side, rear_side, left_side, right_side = _get_sides(ego_bbox)

            intersection_length_per_side = np.asarray(
                [
                    agent_bbox.intersection(front_side).length,
                    agent_bbox.intersection(rear_side).length,
                    agent_bbox.intersection(left_side).length,
                    agent_bbox.intersection(right_side).length,
                ]
            )
            argmax_side = np.argmax(intersection_length_per_side)

            # Remap here is needed because there are two sides that are
            # mapped to the same collision type CollisionType.SIDE
            max_collision_types = max(CollisionType).value
            remap_argmax = min(argmax_side, max_collision_types)
            collision_type = CollisionType(remap_argmax)
            return collision_type, i
    return None
