# Utility functions to visualize the end effector as polygons
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils import pytorch3d_transforms


matplotlib.rcParams['figure.dpi'] = 256


GRIPPER_DELTAS_FOR_VIS = torch.tensor([
    [0, 0, 0,],
    [0, -0.08, 0.08],
    [0, 0.08, 0.08],
])


def normalize_vector(v, return_mag=False):
    device = v.device
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).to(device)))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag

    if return_mag:
        return v, v_mag[:, 0]
    else:
        return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)
    return out  # batch*3


def build_rectangle_points(center, axis_h, axis_w, axis_d, h, w, d):
    def _helper(cur_points, axis, size):
        points = []
        for p in cur_points:
            points.append(p + axis * size / 2)
        for p in cur_points:
            points.append(p - axis * size / 2)
        return points

    points = _helper([center], axis_h, h)
    points = _helper(points, axis_w, w)
    points = _helper(points, axis_d, d)

    return points


def make_polygons(points):
    """Make polygons from 8 side points of a rectangle
    """
    def _helper(four_points):
        center = four_points.mean(axis=0, keepdims=True)
        five_points = np.concatenate([four_points, center], axis=0)
        return [five_points[[0,1,-1]],
                five_points[[0,2,-1]],
                five_points[[0,3,-1]],
                five_points[[1,2,-1]],
                five_points[[1,3,-1]],
                five_points[[2,3,-1]]]

    polygons = (
        _helper(points[:4])
        + _helper(points[-4:])
        + _helper(points[[0,1,4,5]])
        + _helper(points[[2,3,6,7]])
        + _helper(points[[0,2,4,6]])
        + _helper(points[[1,3,5,7]])
    )
    return polygons


def compute_rectangle_polygons(points):
    p1, p2, p3 = points.chunk(3, 0)

    line12 = p2 - p1
    line13 = p3 - p1

    axis_d = F.normalize(cross_product(line12, line13))
    axis_w = F.normalize(p3 - p2)
    axis_h = F.normalize(cross_product(axis_d, axis_w))
    
    length23 = torch.norm(p3 - p2, dim=-1)
    length13 = (line13 * axis_h).sum(-1).abs()
    rectangle1 = build_rectangle_points(p1, axis_d, axis_w, axis_h,
                                        0.03, length23, length13 / 2)
    rectangle2 = build_rectangle_points(p2, axis_d, axis_w, axis_h,
                                        0.03, length23 / 4, length13 * 2)
    rectangle3 = build_rectangle_points(p3, axis_d, axis_w, axis_h,
                                        0.03, length23 / 4 , length13 * 2)

    rectangle1 = torch.cat(rectangle1, dim=0).data.cpu().numpy()
    rectangle2 = torch.cat(rectangle2, dim=0).data.cpu().numpy()
    rectangle3 = torch.cat(rectangle3, dim=0).data.cpu().numpy()

    polygon1 = make_polygons(rectangle1)
    polygon2 = make_polygons(rectangle2)
    polygon3 = make_polygons(rectangle3)

    return polygon1, polygon2, polygon3


def get_gripper_matrix_from_action(action: torch.Tensor,
                                   rotation_param="quat_from_query"):
    """Converts an action to a transformation matrix.

    Args:
        action: A N-D tensor of shape (batch_size, ..., 8) if rotation is
                parameterized as quaternion.  Otherwise, we assume to have
                a 9D rotation vector (3x3 flattened).

    """
    dtype = action.dtype
    device = action.device

    position = action[..., :3]

    if "quat" in rotation_param:
        quaternion = action[..., 3:7]
        rotation = pytorch3d_transforms.quaternion_to_matrix(quaternion)
    else:
        raise NotImplementedError

    shape = list(action.shape[:-1]) + [4, 4]
    gripper_matrix = torch.zeros(shape, dtype=dtype, device=device)
    gripper_matrix[..., :3, :3] = rotation
    gripper_matrix[..., :3, 3] = position
    gripper_matrix[..., 3, 3] = 1

    return gripper_matrix


def get_three_points_from_curr_action(gripper: torch.Tensor,
                                      rotation_param="quat_from_query"):
    gripper_matrices = get_gripper_matrix_from_action(gripper, rotation_param)
    bs = gripper.shape[0]
    pcd = GRIPPER_DELTAS_FOR_VIS.unsqueeze(0).repeat(bs, 1, 1).to(gripper.device)

    pcd = torch.cat([pcd, torch.ones_like(pcd[..., :1])], dim=-1)
    pcd = pcd.permute(0, 2, 1)

    pcd = (gripper_matrices @ pcd).permute(0, 2, 1)
    pcd = pcd[..., :3]

    return pcd


def visualize_ee_pose_and_pcd(pcd, rgb, ee_pose):
    """Use Matplotlib 3D plot to visualize the end effector pose and the point
    cloud.

    Args:
        pcd: a Numpy array of shape (N, 3) representing the point cloud.
        rgb: a Numpy array of shape (N, 3) representing the color of each point.
        ee_pose: a Numpy array of shape (1, 8) representing the end effector pose.
    """
    fig = plt.figure()
    canvas = fig.canvas
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    # Plot the scene point cloud
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c=rgb, s=1)

    # Plot the robot end effector
    ee_pcd = get_three_points_from_curr_action(
        ee_pose, rotation_param="quat_from_query"
    )

    ax.plot(ee_pcd[0, [1, 0, 2], 0],
            ee_pcd[0, [1, 0, 2], 1],
            ee_pcd[0, [1, 0, 2], 2],
            c='r',
            markersize=10, marker='o',
            linestyle='--', linewidth=10)

    polygons = compute_rectangle_polygons(ee_pcd[0])
    for poly_ind, polygon in enumerate(polygons):
        polygon = Poly3DCollection(polygon, facecolors='r')
        ax.add_collection3d(polygon)

    # Show the 3D plot in 10 different views
    ax.legend(loc="lower center")
    images = []
    for elev, azim in zip([10, 15, 20, 25, 30, 25, 20, 15, 45, 90],
                          [0, 45, 90, 135, 180, 225, 270, 315, 360, 360]):
        ax.view_init(elev=elev, azim=azim, roll=0)
        fig.canvas.draw()
        image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)
        image = cv2.resize(image, dsize=None, fx=0.75, fy=0.75)
        images.append(image)

    images = np.concatenate([
        np.concatenate(images[:5], axis=1),
        np.concatenate(images[5:10], axis=1)
    ], axis=0)

    plt.close()

    return images
