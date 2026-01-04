import os
import cv2
import numpy as np
import torch
import pickle
from pytorch3d.io import IO
from pytorch3d.io.experimental_gltf_io import MeshGlbFormat
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRendererWithFragments,
    MeshRasterizer,
    HardPhongShader,
    PointLights,
    BlendParams,
    look_at_view_transform
)

class GLBRenderer:
    def __init__(self, device):
        self.device = device
        self.io = IO()
        self.io.register_meshes_format(MeshGlbFormat())
        
    def load_mesh(self, glb_path):
        """Load GLB model from file"""
        mesh = self.io.load_mesh(glb_path, include_textures=True)
        return mesh.to(self.device)
    
    def setup_camera(self, distance=1.5, elevation=0.0, azimuth=0.0):
        """Setup camera parameters"""
        R, T = look_at_view_transform(distance, elevation, azimuth, device=self.device)
        
        cameras = PerspectiveCameras(
            focal_length=((560.44, 560.44),),
            principal_point=((256, 256),),
            in_ndc=False,
            image_size=[[512,512]],
            device=self.device
        )
        
        return cameras, R, T
    
    def setup_renderer(self, cameras, image_size=512):
        """Setup renderer with given camera parameters"""
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        
        lights = PointLights(
            device=self.device,
            location=((0.0000e+00, 2.9802e-08, -1.0000e+00),),
            ambient_color=((1.0, 1.0, 1.0),),
            diffuse_color=((0.0, 0.0, 0.0),),
            specular_color=((0.0, 0.0, 0.0),),
        )
        
        renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights
            )
        )
        
        return renderer
    
    def render_mesh(self, mesh, cameras, R, T, image_size=512):
        """Render mesh with given camera parameters"""
        renderer = self.setup_renderer(cameras, image_size)
        if R is None:
            image_ref, fragment = renderer(meshes_world=mesh)
        else:
            image_ref, fragment = renderer(meshes_world=mesh, R=R, T=T)
        return image_ref.cpu().numpy().squeeze(), fragment.zbuf.cpu().numpy().squeeze()
    
    def render_multiple_views(self, mesh, output_dir, elevations, azimuths):
        """Render mesh from multiple viewpoints"""
        os.makedirs(output_dir, exist_ok=True)
        
        rgbs = []
        depths = []
        Rs = []
        Ts = []
        
        for elevation, azimuth in zip(elevations, azimuths):
            cameras, R, T = self.setup_camera(distance=1.5, elevation=elevation, azimuth=azimuth)
            rgb, depth = self.render_mesh(mesh, cameras, R, T)
            
            rgbs.append(rgb)
            depths.append(depth)
            Rs.append(R)
            Ts.append(T)
            
            # Save rendered images and depth maps
            cv2.imwrite(f'{output_dir}/rgb_{len(rgbs)-1}.png', rgb[..., [2,1,0]] * 255)
            pickle.dump(depth, open(f'{output_dir}/depth_{len(depths)-1}.pkl', 'wb'))
            pickle.dump(R, open(f'{output_dir}/R_{len(Rs)-1}.pkl', 'wb'))
            pickle.dump(T, open(f'{output_dir}/T_{len(Ts)-1}.pkl', 'wb'))
        
        return rgbs, depths, Rs, Ts 