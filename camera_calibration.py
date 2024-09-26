import numpy as np
import torch
import PIL.Image as Image
from scipy.spatial.transform import Rotation

from utils.vis_utils import visualize_ee_pose_and_pcd


APRILTAG_R = np.array(
    [
        [1., 0., 0.],
        [0.,-1., 0.],
        [0., 0.,-1.],
    ]
)
APRILTAG_SIZE = 0.06
CAMERA_K = np.array(
    [
        [734.1779174804688, 0.,                 993.6226806640625],
        [0.,                734.1779174804688,  551.8895874023438],
        [0.,                0.,                 1.               ],
    ]
)
SCALE = 0.30769231


class DirectLinearTransformationSolver:

    def __init__(self):
        self.H = np.zeros((3, 3))

    @staticmethod
    def expand_vector_dim(points):
        """
        Expands a vector from n dimensions to n+1 dimensions.
        """
        return np.hstack([points, np.ones((points.shape[0], 1))])

    @staticmethod
    def reduce_vector_dim(points):
        """
        Reduces a vector from n+1 dimensions to n dimensions.
        """
        EPSILON = 1e-8
        dim = points.shape[-1]
        points = points / (np.expand_dims(points[:, dim-1], axis=1) + EPSILON)
        return points[:, :dim-1]
    
    def solve(self, points_3d, points_2d):
        # TODO: Implement the Direct Linear Transformation (DLT) algorithm 
        # to estimate the homography matrix
        # self.H = ...
        pass

    def transform(self, points):
        return (self.H @ points.T).T

    def cal_reprojection_error(self, soruce_points, target_points):
        diff = self.reduce_vector_dim(self.transform(soruce_points)) - target_points
        return np.sqrt((diff ** 2).sum(axis=1)).mean()


class CameraToWorldTransformationSolver:

    def __init__(self):
        self.T = np.eye(4)

    def solve_rigid_transformation(self, sources, targets):
        # TODO: Implement the rigid transformation solver
        # The rigid transformation matrix T transforms a point from the source
        # coordinate system to the target coordinate system.
        # It is defined as:
        # T = [R t]
        #     [0 1]
        # where R is a 3x3 rotation matrix and t is a 3x1 translation vector.

        assert sources.shape == targets.shape
        
        # T = ...
        # return T
        pass

    def calculate_reprojection_error(self, sources, targets):
        errors = []
        for source, target in zip(sources, targets):
            # Transform target pose using matrix T
            transformed_target = self.T @ target
            transformed_pos = transformed_target[:3, 3]

            # Compare with tag pos
            tag_pos = source[:3, 3]
            error = np.linalg.norm(tag_pos - transformed_pos)
            errors.append(error)

        return np.mean(errors)

    def solve_extrinsic(self, tag_poses, target_poses_in_camera):
        """
        Solve the extrinsic calibration between the camera and the base.
        """
        tag_pos = np.array([pose[:3, 3] for pose in tag_poses])
        target_pos = np.array([pose[:3, 3] for pose in target_poses_in_camera])
        T = self.solve_rigid_transformation(target_pos, tag_pos)
        print(f"Transformation matrix T:\n{T}")

        rot = Rotation.from_matrix(T[:3, :3])
        quat = rot.as_quat()
        trans = T[:3, 3]
        avg_error = self.calculate_reprojection_error(tag_poses, target_poses_in_camera)
        print(f"Average reprojection error: {avg_error}")
        return T

    def get_matrix(self, transf):
        T = np.eye(4)
        T[:3, 3] = transf[:3]
        T[:3,:3] = Rotation.from_quat(transf[3:7]).as_matrix()[0]
        return T

    def solve(self):
        data = np.load("./data/1a/points.npy", allow_pickle=True)
        base_tags = []
        cam_tags = []

        for point in data:
            base_tag = point["base_tag"]
            base_tag_transform = self.get_matrix(base_tag)
            base_tags.append(base_tag_transform)

            cam_tag = point["camera_tag"]
            cam_tag_transform = self.get_matrix(cam_tag)
            cam_tags.append(cam_tag_transform)

        self.T = self.solve_extrinsic(base_tags, cam_tags)

    def transform(self, point):
        point = np.concatenate([point, np.ones((point.shape[0], 1))], axis=1).T
        transformed_point = (self.T @ point).T
        return transformed_point[:, :3]


class EndEffectorPointCloudAligner:
    def __init__(self):
        # TODO: Load the ZoeDepth model
        # self.model = ...

    def generate_depth_map(self, rgb_image):
        # TODO: Implement the depth map generation
        pass

    def depth_map_to_point_cloud(self, depth_map):
        # TODO: Implement the depth map to point cloud conversion
        # Note that the point cloud should be multiplied by the SCALE factor
        # X = ... * SCALE
        # Y = ... * SCALE
        # Z = ... * SCALE
        # return np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        pass

    def generate_scene_point_cloud(self, pcd, rgb, ee_pose):
        return visualize_ee_pose_and_pcd(pcd, rgb, ee_pose)


if __name__ == "__main__":
    # Part 1: Direct Linear Transform
    points_3d = np.array(
        [
            [-50, -50, 100],
            [ 50, -50, 100],
            [ 50,  50, 100],
            [-50,  50, 100],
            [-50, -50, 200],
            [ 50, -50, 200],
            [ 50,  50, 200],
            [-50,  50, 200],
        ]
    )

    points_2d = np.array(
        [
            [0.03, 1.29],
            [0.35, 1.52],
            [0.41, 0.56],
            [0.34, 0.40],
            [0.26, 0.82],
            [0.34, 0.95],
            [0.38, 0.60],
            [0.34, 0.50],
        ]
    )

    dlt_solver = DirectLinearTransformationSolver()
    dlt_solver.estimate_homography_matrix(points_3d, points_2d)
    error = dlt_solver.cal_reprojection_error(points_3d, points_2d)
    print("Homography Matrix:\n", dlt_solver.H)
    print(error)


    # Part 2: Camera Calibration
    # Load the calibration data
    image1 = Image.open("./data/1a/1.png").convert("RGB")
    image2 = Image.open("./data/1a/2.png").convert("RGB")
    image3 = Image.open("./data/1a/3.png").convert("RGB")

    data1 = np.load("./data/1a/tag_1.npy", allow_pickle=True)
    data2 = np.load("./data/1a/tag_2.npy", allow_pickle=True)
    data3 = np.load("./data/1a/tag_3.npy", allow_pickle=True)

    ee_pose1 = torch.tensor(data1[0]['base_tag'][[0,1,2,4,5,6,3]].astype(np.float32)).unsqueeze(0)
    ee_pose2 = torch.tensor(data2[0]['base_tag'][[0,1,2,4,5,6,3]].astype(np.float32)).unsqueeze(0)
    ee_pose3 = torch.tensor(data3[0]['base_tag'][[0,1,2,4,5,6,3]].astype(np.float32)).unsqueeze(0)


    # Step 1: Solving camera-to-world transformation
    c2w_solver = CameraToWorldTransformationSolver()
    c2w_solver.solve()


    # Step 2: Align the scene point cloud and the end-effector pose
    aligner = EndEffectorPointCloudAligner()
    depth_map1 = aligner.generate_depth_map(image1)
    depth_map2 = aligner.generate_depth_map(image2)
    depth_map3 = aligner.generate_depth_map(image3)

    # Generate point cloud in the camera frame
    pcd1_c = aligner.depth_map_to_point_cloud(depth_map1)
    pcd2_c = aligner.depth_map_to_point_cloud(depth_map2)
    pcd3_c = aligner.depth_map_to_point_cloud(depth_map3)

    rgb1 = np.array(image1).reshape(-1, 3) / 255
    rgb2 = np.array(image2).reshape(-1, 3) / 255
    rgb3 = np.array(image3).reshape(-1, 3) / 255

    # Transform the point cloud to the world frame
    pcd1_w = c2w_solver.transform(pcd1_c)
    pcd2_w = c2w_solver.transform(pcd2_c)
    pcd3_w = c2w_solver.transform(pcd3_c)

    image1 = aligner.generate_scene_point_cloud(pcd1_w, rgb1, ee_pose1)
    image2 = aligner.generate_scene_point_cloud(pcd2_w, rgb2, ee_pose2)
    image3 = aligner.generate_scene_point_cloud(pcd3_w, rgb3, ee_pose3)

    Image.fromarray(image1).save("generated_scene_point_cloud_1.png")
    Image.fromarray(image2).save("generated_scene_point_cloud_2.png")
    Image.fromarray(image3).save("generated_scene_point_cloud_3.png")
