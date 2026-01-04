import cv2
import numpy as np
import torch
from pytorch3d.transforms import so3_exp_map
from pytorch3d.utils import cameras_from_opencv_projection

class PoseEstimator:
    def __init__(self, device):
        self.device = device
        
    def estimate_pose_pnp(self, object_points, image_points, camera_matrix, dist_coeffs):
        """
        Estimate object pose using solvePnPRANSAC
        
        Parameters:
        object_points: 3D point coordinates, shape=(N,3)
        image_points: Corresponding 2D image points, shape=(N,2)
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        
        Returns:
        success: Whether estimation succeeded
        rotation_vec: Rotation vector
        translation_vec: Translation vector
        inliers: Inliers selected by RANSAC
        error: Mean reprojection error
        """
        # Ensure correct input data types
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)
        
        # RANSAC parameters
        iterationsCount = 1000
        reprojectionError = 20.0
        confidence = 0.99
        
        # Call solvePnPRANSAC
        success, rotation_vec, translation_vec, inliers = cv2.solvePnPRansac(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
            iterationsCount=iterationsCount,
            reprojectionError=reprojectionError,
            confidence=confidence,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            # Calculate reprojection error
            projected_points, _ = cv2.projectPoints(
                object_points,
                rotation_vec,
                translation_vec,
                camera_matrix,
                dist_coeffs
            )
            
            error = cv2.norm(image_points, projected_points.reshape(-1,2), cv2.NORM_L2)
            error = error/len(object_points)
            
            return success, rotation_vec, translation_vec, inliers, error, projected_points
        
        return success, None, None, None, None, None
    
    def create_camera_from_pose(self, rvec, tvec, camera_matrix, image_size):
        """Create PyTorch3D camera from OpenCV pose parameters"""
        R = so3_exp_map(torch.tensor(rvec.reshape(1,3))).to(torch.float32)
        tvec = torch.tensor(tvec.reshape(1,3)).to(torch.float32)
        camera_matrix = torch.tensor(camera_matrix).unsqueeze(0).to(torch.float32)
        image_size = torch.tensor([image_size]).to(torch.float32)
        
        cameras = cameras_from_opencv_projection(
            R=R,
            tvec=tvec,
            camera_matrix=camera_matrix,
            image_size=image_size
        )
        
        return cameras.to(self.device) 