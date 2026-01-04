import cv2
import numpy as np
import torch
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.image import load_images

class ImageMatcher:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def get_correspondences(self, image_0, image_1, unprocessed_img0, rgb_18, depth, T, R):
        """Get 3D-2D correspondences between two images"""
        # Format images for DUST3R inference
        if isinstance(image_0, np.ndarray):
            image_0 = {'img': torch.from_numpy(image_0).permute(2, 0, 1).float() / 255.0}
        if isinstance(image_1, np.ndarray):
            image_1 = {'img': torch.from_numpy(image_1).permute(2, 0, 1).float() / 255.0}
            
        # Run DUST3R inference
        output = inference([tuple([image_0, image_1])], self.model, self.device, batch_size=1, verbose=False)
        
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']
        
        desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()
        
        # Find 2D-2D matches
        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1, desc2, 
            subsample_or_initxy1=8,
            device=self.device, 
            dist='dot', 
            block_size=2**13
        )
        
        # Filter matches near image borders
        H0, W0 = view1['true_shape'][0]
        valid_matches_im0 = (
            (matches_im0[:, 0] >= 3) & 
            (matches_im0[:, 0] < int(W0) - 3) & 
            (matches_im0[:, 1] >= 3) & 
            (matches_im0[:, 1] < int(H0) - 3)
        )
        
        H1, W1 = view2['true_shape'][0]
        valid_matches_im1 = (
            (matches_im1[:, 0] >= 3) & 
            (matches_im1[:, 0] < int(W1) - 3) & 
            (matches_im1[:, 1] >= 3) & 
            (matches_im1[:, 1] < int(H1) - 3)
        )
        
        valid_matches = valid_matches_im0 & valid_matches_im1
        matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]
        
        # Process images and convert matches
        rgb_18 = cv2.cvtColor(rgb_18, cv2.COLOR_BGR2RGB)
        rgb_18_processed = cv2.resize(rgb_18, (512, 512))
        cx, cy = rgb_18_processed.shape[1]//2, rgb_18_processed.shape[0]//2
        halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
        halfh = int(3*halfw/4)
        
        unprocessed_matches1 = (matches_im1 + np.array([0, cy-halfh]))
        unprocessed_matches0 = (matches_im0 + np.array([0, cy-halfh]))
        
        unprocessed_img0 = cv2.cvtColor(unprocessed_img0, cv2.COLOR_BGR2RGB)
        
        # Filter matches with invalid depth
        depth_of_matches1 = depth[unprocessed_matches1[:, 1].astype(int), unprocessed_matches1[:, 0].astype(int)]
        valid_matches = (depth_of_matches1 != -1)
        unprocessed_matches0 = unprocessed_matches0[valid_matches]
        unprocessed_matches1 = unprocessed_matches1[valid_matches]
        depth_of_matches1 = depth_of_matches1[valid_matches]
        
        # Convert 2D points to 3D points
        fx, fy, cx, cy = 560.44, 560.44, 256, 256
        u = 512 - unprocessed_matches1[:, 0]
        v = 512 - unprocessed_matches1[:, 1]
        
        x = (u - cx) * depth_of_matches1 / fx
        y = (v - cy) * depth_of_matches1 / fy
        z = depth_of_matches1
        
        points_3d = np.stack((x, y, z), axis=-1)
        
        # Convert to world coordinates
        T = T.reshape(3, 1)
        R = R.squeeze(0)
        points_world = np.matmul(R, (points_3d.T - T)).T
        
        return points_world, unprocessed_matches0 