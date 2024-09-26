import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import torch

from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images


class Dust3rPlayGround:

    def __init__(self):
        self.device = 'cuda'
        self.batch_size = 1
        model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        # you can put the path to a local checkpoint in model_name if needed
        self.model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(self.device)

        # Parameters for global alignment
        self._schedule = 'cosine'
        self._lr = 0.01
        self._niter = 300

    def get_global_aligned_pointmaps(self, image_paths, gui=True):
        """Given a list of image paths, return the global aligned pointmaps

        Check https://github.com/naver/dust3r#usage for more details

        Args:
            image_paths: a list of image paths
        """
        # Predict 3d pointmaps with DUSt3R
        images = load_images(image_paths, size=512)
        pairs = make_pairs(
            images, scene_graph='complete', prefilter=None, symmetrize=True
        )
        output = inference(
            pairs, self.model, self.device, batch_size=self.batch_size
        )

        # Perform global alignment
        scene = global_aligner(
            output, device=self.device, mode=GlobalAlignerMode.PointCloudOptimizer
        )
        loss = scene.compute_global_alignment(
            init="mst", niter=self._niter, schedule=self._schedule, lr=self._lr)

        # Visualize globally aligned pointmaps
        pts3d = scene.get_pts3d(raw=False)  # A list of HxWx3 numpy arrays
        rgb = scene.imgs

        pts3d = torch.stack(pts3d, dim=0).reshape(-1, 3).data.cpu().numpy()
        rgb = np.stack(rgb, axis=0).reshape(-1, 3)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], c=rgb, s=1)

        if gui:
            plt.show()
        else:
            # Visualize point cloud from different views
            fig.tight_layout()
            canvas = fig.canvas
            vizs = []
            for elev, azim in zip([10, 15, 20, 25, 30, 25, 20, 15, 45, 90],
                                [0, 45, 90, 135, 180, 225, 270, 315, 360, 360]):
                ax.view_init(elev=elev, azim=azim, roll=0)
                canvas.draw()
                viz = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
                viz = viz.reshape(*reversed(canvas.get_width_height()), 3)
                vizs.append(viz)
            vizs = np.concatenate([
                np.concatenate(vizs[:5], axis=1),
                np.concatenate(vizs[5:10], axis=1)
            ], axis=0)

            Image.fromarray(vizs, mode='RGB').save('global_aligned_pointmaps.png')

    def nearest_neighbor_search(self, arr1, arr2):
        """You will implement nearest neighbor search to find the correspondence
        between two arrays.  Use Euclidean distance as the metric to search the
        nearest neighbors.

        Args:
            arr1: A torch.Tensor of shape (num_arr1, C)
            arr2: A torch.Tensor of shape (num_arr2, C)
        
        Return:
            nn_inds: A tensor of shape (num_arr1) indicates the nearest-neighbor
                indices of arr1 in arr2
        """
        # TODO: compute pairwise Euclidean distance between arr1 and arr2
        # dists = ...
        # TODO: compute nearest neighbor of arr1 in arr2
        # nn_inds = ...

        # return nn_inds
        pass

    def search_correspondence(self, image_paths):
        """Given a list of image paths, detect interesting points with SIFT
        features, and find their correspondence in another view

        Args:
            image_paths: a list of image paths
        """
        # Predict 3d pointmaps with DUSt3R
        images = load_images(image_paths, size=512)
        pairs = make_pairs(
            images, scene_graph='complete', prefilter=None, symmetrize=True
        )
        output = inference(
            pairs, self.model, self.device, batch_size=self.batch_size
        )

        # Detect keypoints with SIFT
        view1 = output['view1']['img'][0].permute(1, 2, 0).add(1).mul(0.5 * 255)
        view1 = view1.data.cpu().numpy().astype(np.uint8)
        sift = cv2.SIFT_create()
        key_pts, des = sift.detectAndCompute(view1, None)
        key_pts = [(round(key_pt.pt[0]), round(key_pt.pt[1])) for key_pt in key_pts]

        viz = view1.copy()
        for (x, y) in key_pts:
            viz[y-2:y+2, x-2:x+2] = (255, 0, 0)
        Image.fromarray(viz, mode='RGB').save('kp.png')

        # Find the corresponded pixels in another view
        view2 = output['view2']['img'][0].permute(1, 2, 0).add(1).mul(0.5 * 255)
        view2 = view2.data.cpu().numpy().astype(np.uint8)
        pts3d_view1 = output['pred1']['pts3d'][0]
        pts3d_view2 = output['pred2']['pts3d_in_other_view'][0]

        key_pts3d_view1 = [pts3d_view1[y, x] for x, y in key_pts]
        pts3d_view2 = pts3d_view2.reshape(-1, 3)
        key_pts3d_view1 = torch.stack(key_pts3d_view1, dim=0)
        nn_inds_view2 = self.nearest_neighbor_search(key_pts3d_view1, pts3d_view2)
        nn_inds_view2 = nn_inds_view2.tolist()


        h, w = pts3d_view1.shape[:2]
        grid_xs, grid_ys = np.meshgrid(np.arange(w), np.arange(h))
        grid_xs = grid_xs.reshape(-1)
        grid_ys = grid_ys.reshape(-1)

        # Visualize in another view
        viz = np.concatenate([view1, view2], axis=1)
        n_viz = len(key_pts)
        plt.figure()
        plt.imshow(viz)
        cmap = plt.get_cmap('jet')
        for i in range(len(key_pts)):
            x1, y1 = key_pts[i]
            nn_ind = nn_inds_view2[i]
            x2, y2 = grid_xs[nn_ind], grid_ys[nn_ind]
            plt.plot([x1, x2 + w], [y1, y2], '-+', color=cmap(i / (n_viz - 1)),
                    scalex=False, scaley=False)
        plt.show(block=True)
        plt.savefig('nn_match.png', dpi=128)


if __name__ == "__main__":
    dustr_playground = Dust3rPlayGround()
    
    # TODO: Provide your image paths here (image_paths is the str list)
    image_paths = []
    dustr_playground.get_global_aligned_pointmaps(image_paths, gui=False)

    image_paths = glob.glob('./data/1b/*.png')
    dustr_playground.search_correspondence(image_paths)
