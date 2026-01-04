import json
import os
import bpy
import mathutils
import numpy as np
from pathlib import Path
import sys
from contextlib import redirect_stdout, redirect_stderr
import argparse
import trimesh

"""
Render PLY point clouds with 3D bbox overlay and camera trajectory animation.
See README.md for usage details.
"""

# ============================================================================
# BBox JSON to PLY Conversion Functions
# ============================================================================

def create_thick_line(start, end, thickness, color=[255, 0, 0, 255]):
    """Create a colored box (cuboid) between two 3D points."""
    direction = end - start
    length = np.linalg.norm(direction)
    if length == 0:
        return None
    direction /= length

    z = direction
    up = np.array([0, 1, 0]) if abs(z[1]) < 0.99 else np.array([1, 0, 0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    rot = np.vstack((x, y, z)).T
    box = trimesh.creation.box(extents=[thickness, thickness, length])
    box.apply_translation([0, 0, length / 2])
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = start
    box.apply_transform(transform)
    box.visual.vertex_colors = np.tile(np.array(color), (len(box.vertices), 1))
    return box


def compute_adaptive_thickness(data, ratio=0.02):
    """Compute adaptive line thickness based on bbox sizes."""
    box_sizes = []
    for box_data in data:
        bbox = np.array(box_data['bbox3D_cam'], dtype=np.float32)
        w = np.linalg.norm(bbox[1] - bbox[0])
        h = np.linalg.norm(bbox[4] - bbox[0])
        d = np.linalg.norm(bbox[3] - bbox[0])
        box_sizes.append(np.mean([w, h, d]))
    return np.median(box_sizes) * ratio


def convert_bbox_json_to_ply(json_path, ply_path, thickness=None, ratio=0.04):
    """Convert bbox JSON file to PLY with colored edges."""
    edges = [
        [0,1],[1,2],[2,3],[3,0],  # bottom
        [4,5],[5,6],[6,7],[7,4],  # top
        [0,4],[1,5],[2,6],[3,7]   # verticals
    ]
    color_palette = [
        [255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255],
        [255, 255, 0, 255], [255, 0, 255, 255], [0, 255, 255, 255],
        [255, 127, 0, 255], [127, 0, 255, 255], [0, 127, 255, 255]
    ]

    with open(json_path, 'r') as f:
        data = json.load(f)

    if len(data) == 0:
        return False

    if thickness is None:
        thickness = compute_adaptive_thickness(data, ratio=ratio)

    meshes = []
    for i, box_data in enumerate(data):
        bbox = np.array(box_data['bbox3D_cam'], dtype=np.float32)
        color = color_palette[i % len(color_palette)]
        for i0, i1 in edges:
            bar = create_thick_line(bbox[i0], bbox[i1], thickness, color=color)
            if bar:
                meshes.append(bar)

    full_mesh = trimesh.util.concatenate(meshes)
    full_mesh.export(ply_path)
    return True


# ============================================================================
# Blender Rendering Functions
# ============================================================================

def cleanup():
    for key in bpy.data.objects.keys():
        bpy.data.objects.remove(bpy.data.objects[key], do_unlink=True)

def suppress_blender_output():
    """Context manager to suppress Blender output more effectively"""
    class SuppressOutput:
        def __init__(self):
            self.original_stdout = None
            self.original_stderr = None
            self.devnull = None
            self.original_stdout_fd = None
            self.original_stderr_fd = None
            self.devnull_fd = None
        
        def __enter__(self):
            # Redirect Python stdout/stderr
            self.original_stdout = sys.stdout
            self.original_stderr = sys.stderr
            self.devnull = open(os.devnull, 'w')
            sys.stdout = self.devnull
            sys.stderr = self.devnull
            
            # Also redirect file descriptors (for C-level output)
            try:
                self.original_stdout_fd = os.dup(1)
                self.original_stderr_fd = os.dup(2)
                self.devnull_fd = os.open(os.devnull, os.O_WRONLY)
                os.dup2(self.devnull_fd, 1)
                os.dup2(self.devnull_fd, 2)
            except:
                pass
            
            # Set environment to suppress output
            os.environ['BLENDER_USER_RESOURCES'] = os.devnull
            
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            # Restore Python stdout/stderr
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            if self.devnull:
                self.devnull.close()
            
            # Restore file descriptors
            try:
                if self.original_stdout_fd is not None:
                    os.dup2(self.original_stdout_fd, 1)
                    os.close(self.original_stdout_fd)
                if self.original_stderr_fd is not None:
                    os.dup2(self.original_stderr_fd, 2)
                    os.close(self.original_stderr_fd)
                if self.devnull_fd is not None:
                    os.close(self.devnull_fd)
            except:
                pass
    
    return SuppressOutput()

def render_ply_with_bbox_trajectory(ply_file_path, bbox_file_path, cam_params_path, output_dir,
                                   output_name="camera_trajectory", verbose=False, camera_offset_ratio=0.8):
    """
    Render PLY file with bbox overlay and create camera trajectory animation.

    Args:
        ply_file_path (str or Path): Path to the main PLY file
        bbox_file_path (str or Path): Path to the bbox PLY file
        cam_params_path (str or Path): Path to the camera parameters JSON file
        output_dir (str or Path): Directory to save output files
        output_name (str): Base name for output files (without extension)
        verbose (bool): Whether to show Blender rendering output
        camera_offset_ratio (float): Ratio of camera movement relative to bbox max dimension (default: 0.8)
    """
    # Convert to Path objects
    ply_file_path = Path(ply_file_path)
    bbox_file_path = Path(bbox_file_path)
    cam_params_path = Path(cam_params_path)
    output_dir = Path(output_dir)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Processing: {ply_file_path.name}")
    else:
        # Set Blender to minimal output mode
        try:
            # Disable various output options
            bpy.context.scene.render.use_stamp = False
            bpy.context.scene.render.use_stamp_time = False
            bpy.context.scene.render.use_stamp_date = False
            bpy.context.scene.render.use_stamp_frame = False
            bpy.context.scene.render.use_stamp_scene = False
            bpy.context.scene.render.use_stamp_camera = False
            bpy.context.scene.render.use_stamp_lens = False
            bpy.context.scene.render.use_stamp_filename = False
            bpy.context.scene.render.use_stamp_marker = False
            bpy.context.scene.render.use_stamp_sequencer_strip = False
        except:
            pass
    
    # Clean up existing objects
    cleanup()

    # Load the main PLY file
    bpy.ops.import_mesh.ply(filepath=str(ply_file_path))
    ply_object = bpy.context.selected_objects[0]

    # Set the PLY object's position and rotation matrix
    ply_object.matrix_world = mathutils.Matrix(
        ((1, 0, 0, 0),
         (0, -1, 0, 0),
         (0, 0, -1, 0),
         (0, 0, 0, 1))
    )

    # Load the bbox PLY file
    bpy.ops.import_mesh.ply(filepath=str(bbox_file_path))
    bbox_object = bpy.context.selected_objects[0]

    # Apply the same transformation matrix to bbox object
    bbox_object.matrix_world = mathutils.Matrix(
        ((1, 0, 0, 0),
         (0, -1, 0, 0),
         (0, 0, -1, 0),
         (0, 0, 0, 1))
    )

    # Calculate scale factor with better normalization for outdoor scenes
    # Option 1: Base scaling on bbox range (better for outdoor scenes)
    bbox_bbox = bbox_object.bound_box
    world_bbox_corners = [bbox_object.matrix_world @ mathutils.Vector(corner) for corner in bbox_bbox]
    
    # Find bbox X-axis range
    bbox_x_coords = [corner.x for corner in world_bbox_corners]
    bbox_x_min = min(bbox_x_coords)
    bbox_x_max = max(bbox_x_coords)
    bbox_x_range = bbox_x_max - bbox_x_min

    # Find bbox Y-axis range
    bbox_y_coords = [corner.y for corner in world_bbox_corners]
    bbox_y_min = min(bbox_y_coords)
    bbox_y_max = max(bbox_y_coords)
    bbox_y_range = bbox_y_max - bbox_y_min

    # Find bbox Z-axis range
    bbox_z_coords = [corner.z for corner in world_bbox_corners]
    bbox_z_min = min(bbox_z_coords)
    bbox_z_max = max(bbox_z_coords)
    bbox_z_range = bbox_z_max - bbox_z_min

    # Get maximum dimension of bbox
    bbox_max_dimension = max(bbox_x_range, bbox_y_range, bbox_z_range)

    # Also get PLY range for comparison
    ply_bbox = ply_object.bound_box
    world_ply_corners = [ply_object.matrix_world @ mathutils.Vector(corner) for corner in ply_bbox]
    ply_x_coords = [corner.x for corner in world_ply_corners]
    ply_x_min = min(ply_x_coords)
    ply_x_max = max(ply_x_coords)
    ply_x_range = ply_x_max - ply_x_min
    
    # Choose scaling strategy based on scene characteristics
    if bbox_x_range > 0 and ply_x_range > 0:
        # If bbox is much smaller than scene, use bbox-based scaling
        bbox_to_scene_ratio = bbox_x_range / ply_x_range

        if bbox_to_scene_ratio < 0.1:  # bbox is less than 10% of scene width
            # Use bbox range for scaling (better for outdoor scenes)
            bbox_scale = 0.8 / bbox_x_range  # Scale bbox to unit size
            min_scale_outdoor = 0.1  # Minimum scale factor for outdoor scenes
            scale_factor = max(bbox_scale, min_scale_outdoor)
            print(f"Using bbox-based scaling (bbox ratio: {bbox_to_scene_ratio:.3f}, raw: {bbox_scale:.3f}, clamped: {scale_factor:.3f})")
        else:
            # Use scene range but with minimum scale limit
            scene_scale = 2.0 / ply_x_range
            min_scale_indoor = 0.2  # Minimum scale factor for indoor scenes (increased from 0.1)
            scale_factor = max(scene_scale, min_scale_indoor)
            print(f"Using scene-based scaling with limits (bbox ratio: {bbox_to_scene_ratio:.3f}, raw: {scene_scale:.3f}, clamped: {scale_factor:.3f})")
    else:
        scale_factor = 1.0
    
    print(f"PLY X range: {ply_x_min:.3f} to {ply_x_max:.3f} (range: {ply_x_range:.3f})")
    print(f"BBox ranges - X: {bbox_x_range:.3f}, Y: {bbox_y_range:.3f}, Z: {bbox_z_range:.3f}")
    print(f"BBox max dimension: {bbox_max_dimension:.3f}")
    print(f"Final scale factor: {scale_factor:.3f}")

    # Apply uniform scaling to both objects
    ply_object.scale = (scale_factor, scale_factor, scale_factor)
    bbox_object.scale = (scale_factor, scale_factor, scale_factor)

    # Force update to apply transformations
    bpy.context.view_layer.update()

    print(f"Applied uniform scaling factor {scale_factor:.3f} to both PLY and bbox objects")

    # Calculate camera offset distance based on scaled bbox dimensions
    scaled_bbox_max_dimension = bbox_max_dimension * scale_factor
    calculated_offset_distance = scaled_bbox_max_dimension * camera_offset_ratio

    print(f"Scaled bbox max dimension: {scaled_bbox_max_dimension:.3f}")
    print(f"Calculated camera offset distance: {calculated_offset_distance:.3f} (ratio: {camera_offset_ratio})")

    # Set up material for bbox object (different from main PLY)
    if bbox_object.data.materials:
        bbox_mat = bbox_object.data.materials[0]
    else:
        bbox_mat = bpy.data.materials.new(name="BBox_Material")
        bbox_object.data.materials.append(bbox_mat)

    # Use nodes to set bbox material
    bbox_mat.use_nodes = True

    # Get node tree for bbox
    bbox_nodes = bbox_mat.node_tree.nodes
    bbox_links = bbox_mat.node_tree.links

    # Clear default nodes
    for node in bbox_nodes:
        bbox_nodes.remove(node)

    # Create nodes for bbox
    bbox_output = bbox_nodes.new(type='ShaderNodeOutputMaterial')
    bbox_bsdf = bbox_nodes.new(type='ShaderNodeBsdfPrincipled')
    bbox_vc_node = bbox_nodes.new(type='ShaderNodeVertexColor')
    bbox_emission = bbox_nodes.new(type='ShaderNodeEmission')
    bbox_mix = bbox_nodes.new(type='ShaderNodeMixShader')
    bbox_coloramp = bbox_nodes.new(type='ShaderNodeValToRGB')

    # Automatically detect vertex color layer names for bbox
    if bbox_object.data.color_attributes:
        bbox_vc_node.layer_name = bbox_object.data.color_attributes[0].name  # Usually "Col"
    else:
        print("Warning: No color attributes found in bbox PLY file")

    # Set up color ramp to enhance colors
    bbox_coloramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)  # Black at 0
    bbox_coloramp.color_ramp.elements[1].color = (2.0, 2.0, 2.0, 1.0)  # Brightened white at 1

    # Connect vertex color through color ramp for enhancement
    bbox_links.new(bbox_vc_node.outputs['Color'], bbox_coloramp.inputs['Fac'])
    bbox_links.new(bbox_vc_node.outputs['Color'], bbox_bsdf.inputs['Base Color'])

    # Set up emission for glow effect
    bbox_links.new(bbox_vc_node.outputs['Color'], bbox_emission.inputs['Color'])
    bbox_emission.inputs['Strength'].default_value = 0.5  # Emission strength

    # Mix BSDF and emission
    bbox_links.new(bbox_bsdf.outputs['BSDF'], bbox_mix.inputs[1])
    bbox_links.new(bbox_emission.outputs['Emission'], bbox_mix.inputs[2])
    bbox_mix.inputs['Fac'].default_value = 0.3  # 30% emission, 70% BSDF

    # Connect to output
    bbox_links.new(bbox_mix.outputs['Shader'], bbox_output.inputs['Surface'])

    # Set material parameters for bbox (more metallic for better visibility)
    bbox_bsdf.inputs['Roughness'].default_value = 0.3  # More reflective
    bbox_bsdf.inputs['Metallic'].default_value = 0.8   # More metallic

    # If you need to customize the material, you can configure it as before
    if ply_object.data.materials:
        mat = ply_object.data.materials[0]
    else:
        mat = bpy.data.materials.new(name="PLY_Material")
        ply_object.data.materials.append(mat)

    # Use nodes to set materials
    mat.use_nodes = True

    # Get node tree
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes (prevent conflicts)
    for node in nodes:
        nodes.remove(node)

    # Create a node
    output = nodes.new(type='ShaderNodeOutputMaterial')
    emission = nodes.new(type='ShaderNodeEmission')
    vc_node = nodes.new(type='ShaderNodeVertexColor')

    # Automatically detects vertex color layer names
    if ply_object.data.color_attributes:
        vc_node.layer_name = ply_object.data.color_attributes[0].name  # 通常是 "Col"
    else:
        print("Warning: There is no color attribute in the PLY file")

    # connect nodes - use emission for unlit rendering (preserves original colors)
    links.new(vc_node.outputs['Color'], emission.inputs['Color'])
    links.new(emission.outputs['Emission'], output.inputs['Surface'])

    # Set emission strength to 1.0 for original brightness
    emission.inputs['Strength'].default_value = 1.0

    with open(cam_params_path, 'r') as fp:
        data = json.load(fp)

    K = np.array(data['K'])
    W = data['W']
    H = data['H']
    c2w = np.array(data['c2w'])

    new_camera_data = bpy.data.cameras.new(name="New Camera")
    new_camera_object = bpy.data.objects.new(name="New Camera", object_data=new_camera_data)

    # Link the camera object to the scene
    bpy.context.collection.objects.link(new_camera_object)

    # Set the new camera as the active camera in the scene
    bpy.context.scene.camera = new_camera_object

    # Set camera intrinsics
    camera = bpy.context.scene.camera.data

    # Convert focal length from pixels to millimeters
    # Assuming sensor width is 36mm (default for Blender cameras)
    sensor_width = 36  # in mm
    focal_length_mm = (K[0, 0] * sensor_width) / W  # Focal length in mm
    camera.lens = focal_length_mm

    # Set the sensor width and height
    camera.sensor_width = sensor_width
    camera.sensor_height = sensor_width * (H / W)  # Maintain aspect ratio

    # Set the render resolution to be divisible by 2
    bpy.context.scene.render.resolution_x = W - (W % 2)  # Make sure width is even
    bpy.context.scene.render.resolution_y = H - (H % 2)  # Make sure height is even

    # Set camera extrinsics - keep original c2w orientation
    new_camera_object.matrix_world = mathutils.Matrix(c2w.tolist())

    # Force update the scene
    bpy.context.view_layer.update()

    # Print camera position and direction
    camera_matrix = new_camera_object.matrix_world
    camera_location = camera_matrix.translation
    camera_rotation = camera_matrix.to_euler()

    print("=== Initial Camera Information ===")
    print(f"Camera Position (X, Y, Z): ({camera_location.x:.3f}, {camera_location.y:.3f}, {camera_location.z:.3f})")
    print(f"Camera Location directly: ({new_camera_object.location.x:.3f}, {new_camera_object.location.y:.3f}, {new_camera_object.location.z:.3f})")
    print(f"Camera Rotation (X, Y, Z) in radians: ({camera_rotation.x:.3f}, {camera_rotation.y:.3f}, {camera_rotation.z:.3f})")
    print(f"Camera Rotation (X, Y, Z) in degrees: ({np.degrees(camera_rotation.x):.1f}, {np.degrees(camera_rotation.y):.1f}, {np.degrees(camera_rotation.z):.1f})")

    # Get camera forward direction (negative Z in camera space becomes world direction)
    camera_direction = camera_matrix @ mathutils.Vector((0, 0, -1, 0))
    print(f"Camera Forward Direction: ({camera_direction.x:.3f}, {camera_direction.y:.3f}, {camera_direction.z:.3f})")
    print("=========================")

    light_type = 'AREA' 

    # Create a new light data block
    new_light_data = bpy.data.lights.new(name="New Light", type=light_type)

    # Create a new light object
    new_light_object = bpy.data.objects.new(name="New Light", object_data=new_light_data)

    # Link the light object to the scene
    bpy.context.collection.objects.link(new_light_object)

    # Set the new light as the active light in the scene
    bpy.context.view_layer.objects.active = new_light_object

    bpy.ops.object.light_add(type='POINT', radius=1, align='WORLD', location=(0, 4, 2))

    # Get the newly created light object
    light_object = bpy.context.object

    # Set light parameters
    light_data = light_object.data
    light_data.energy = 1000  # Adjust light intensity as needed
    light_data.shadow_soft_size = 0.8
    bpy.context.scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 1.0
    bpy.context.scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)

    # Set white background instead of transparent
    bpy.context.scene.render.film_transparent = False

    # Function to set camera position and look at center
    def set_camera_position_and_look_at(camera, position, center):
        camera.location = position
        
        # Calculate look-at rotation
        direction = center - camera.location
        direction_normalized = direction.normalized()
        
        # Get the original rotation to preserve the roll angle
        original_roll = camera.rotation_euler.z
        
        # Calculate new angles
        yaw = np.arctan2(-direction_normalized.x, -direction_normalized.z)
        pitch = np.arctan2(direction_normalized.y, 
                          np.sqrt(direction_normalized.x**2 + direction_normalized.z**2))
        
        # Set rotation
        camera.rotation_euler = mathutils.Euler((pitch, yaw, original_roll), 'XYZ')
        
        # Force update
        bpy.context.view_layer.update()

    # Get the original camera position from c2w matrix
    original_camera_pos = mathutils.Vector(new_camera_object.location)
    print(f"Original camera position: ({original_camera_pos.x:.3f}, {original_camera_pos.y:.3f}, {original_camera_pos.z:.3f})")

    # Calculate the original camera's forward direction
    original_camera_matrix = mathutils.Matrix(c2w.tolist())
    original_forward = original_camera_matrix @ mathutils.Vector((0, 0, -1, 0))
    original_forward_3d = original_forward.xyz.normalized()

    print(f"Original camera forward direction: ({original_forward_3d.x:.3f}, {original_forward_3d.y:.3f}, {original_forward_3d.z:.3f})")

    # Cast a ray from camera position in the forward direction to find intersection with PLY object
    ray_origin = original_camera_pos
    ray_direction = original_forward_3d
    max_distance = 100.0  # Maximum ray cast distance

    # Perform ray casting
    hit_result, hit_location, hit_normal, hit_face = ply_object.ray_cast(ray_origin, ray_direction, distance=max_distance)

    if hit_result:
        # Use the intersection point as the look-at target
        look_at_target = hit_location
        print(f"Ray hit at: ({look_at_target.x:.3f}, {look_at_target.y:.3f}, {look_at_target.z:.3f})")
    else:
        # Fallback: use average depth of bbox points
        bbox_mesh = bbox_object.data
        bbox_vertices = [bbox_object.matrix_world @ v.co for v in bbox_mesh.vertices]
        
        # Calculate average Z coordinate (depth) of bbox points
        avg_z = sum(v.z for v in bbox_vertices) / len(bbox_vertices)
        
        # Use average bbox depth, keeping original camera's X and Y
        look_at_target = mathutils.Vector((original_camera_pos.x, original_camera_pos.y, avg_z))
        print("No intersection found, using bbox average depth")
        print(f"BBox vertices count: {len(bbox_vertices)}")
        print(f"BBox average depth (Z): {avg_z:.3f}")
        print(f"Fallback target: ({look_at_target.x:.3f}, {look_at_target.y:.3f}, {look_at_target.z:.3f})")

    # Define camera positions relative to original position
    # Use calculated_offset_distance based on scaled bbox dimensions
    positions = [
        original_camera_pos,  # Start from original position
        original_camera_pos + mathutils.Vector((-calculated_offset_distance, calculated_offset_distance, calculated_offset_distance * 0.75)),  # Left-up
        original_camera_pos + mathutils.Vector((calculated_offset_distance, calculated_offset_distance, calculated_offset_distance * 0.75)),   # Right-up
        original_camera_pos   # Back to original position
    ]

    # Set up animation parameters
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 90  # 4 seconds at 30fps for smoother motion
    frames_per_position = 30  # Longer transition time for smoother movement

    # Create animation with smoother transitions
    for i, pos in enumerate(positions):
        # Set frame
        frame = i * frames_per_position
        bpy.context.scene.frame_set(frame)

        # Move camera
        if i == 0 or i == 3:
            # First and last frame: keep original c2w orientation (don't look-at)
            new_camera_object.location = pos
            new_camera_object.matrix_world = mathutils.Matrix(c2w.tolist())
        else:
            # Other frames: look at target point
            set_camera_position_and_look_at(new_camera_object, pos, look_at_target)

        # Add keyframe for location and rotation
        new_camera_object.keyframe_insert(data_path="location")
        new_camera_object.keyframe_insert(data_path="rotation_euler")

    # Set interpolation for very smooth camera movement
    if new_camera_object.animation_data and new_camera_object.animation_data.action:
        for fcurve in new_camera_object.animation_data.action.fcurves:
            for kf in fcurve.keyframe_points:
                kf.interpolation = 'BEZIER'
                kf.handle_left_type = 'AUTO_CLAMPED'  # Better than AUTO for smooth curves
                kf.handle_right_type = 'AUTO_CLAMPED'
                # Reduce easing for smoother motion
                kf.easing = 'EASE_IN_OUT'

    # Optional: Add more intermediate keyframes for ultra-smooth motion
    # Uncomment the following if you want even smoother motion

    # import bpy
    # Add intermediate keyframes between each main position
    # for i in range(len(positions) - 1):
    #     start_frame = i * frames_per_position
    #     end_frame = (i + 1) * frames_per_position
    #     mid_frame = (start_frame + end_frame) // 2
        
    #     # Calculate intermediate position
    #     start_pos = positions[i]
    #     end_pos = positions[i + 1]
    #     mid_pos = start_pos.lerp(end_pos, 0.5)  # Linear interpolation
        
    #     bpy.context.scene.frame_set(mid_frame)
    #     set_camera_position_and_look_at(new_camera_object, mid_pos, look_at_target)
    #     new_camera_object.keyframe_insert(data_path="location")
    #     new_camera_object.keyframe_insert(data_path="rotation_euler")

    # Set the output format for video
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
    bpy.context.scene.render.ffmpeg.format = 'MPEG4'
    bpy.context.scene.render.ffmpeg.codec = 'H264'
    bpy.context.scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'  # Better quality
    bpy.context.scene.render.ffmpeg.ffmpeg_preset = 'GOOD'  # Better compression
    bpy.context.scene.render.filepath = str(output_dir / f"{output_name}.mp4")

    # Render animation
    if verbose:
        bpy.ops.render.render(animation=True)
    else:
        with suppress_blender_output():
            bpy.ops.render.render(animation=True)

# Example function call - uncomment and modify paths as needed
"""
render_ply_with_bbox_trajectory(
    ply_file_path="experimental_results/COCO/val/000000000632/depth_scene_no_edge.ply",
    bbox_file_path="experimental_results/COCO/val/000000000632/bbox.ply",
    cam_params_path="experimental_results/COCO/val/000000000632/cam_params.json",
    output_dir="experimental_results/COCO/val/000000000632/output",
    output_name="camera_trajectory",
    camera_offset_ratio=0.8  # Adjust ratio for more/less camera movement
)
"""

# Parse command line arguments
parser = argparse.ArgumentParser(description='Render PLY files with bbox overlay and camera trajectory')
parser.add_argument('--root', type=str, required=True,
                   help='Root directory containing scene folders (each with depth_scene_no_edge.ply, bbox.ply, cam_params.json)')
parser.add_argument('--start_idx', type=int, default=None,
                   help='Start directory index to process (0 to total_dirs-1). If not specified, start from 0.')
parser.add_argument('--end_idx', type=int, default=None,
                   help='End directory index to process (0 to total_dirs-1). If not specified, process to the end.')
parser.add_argument('--verbose', action='store_true',
                   help='Show verbose rendering output')

# Handle Blender's argument parsing quirks
if '--' in sys.argv:
    # Everything after '--' are script arguments
    script_args = sys.argv[sys.argv.index('--') + 1:]
else:
    # No '--' separator, use all arguments after the script name
    script_args = sys.argv[1:] if len(sys.argv) > 1 else []

args = parser.parse_args(script_args)
root = Path(args.root)
start_idx = args.start_idx
end_idx = args.end_idx
verbose_global = args.verbose

# Get all directories and count them for progress tracking
all_dirs = sorted(list(root.glob("*")))
total_dirs = len(all_dirs)

print(f"Found {total_dirs} total directories")

# Set default values if not specified
if start_idx is None:
    start_idx = 0
if end_idx is None:
    end_idx = total_dirs - 1

# Validate index ranges
if start_idx < 0 or start_idx >= total_dirs:
    print(f"Error: start_idx {start_idx} is out of range (0-{total_dirs-1})")
    sys.exit(1)
if end_idx < 0 or end_idx >= total_dirs:
    print(f"Error: end_idx {end_idx} is out of range (0-{total_dirs-1})")
    sys.exit(1)
if start_idx > end_idx:
    print(f"Error: start_idx ({start_idx}) cannot be greater than end_idx ({end_idx})")
    sys.exit(1)

# Select directories based on range
selected_dirs = all_dirs[start_idx:end_idx+1]
print(f"Processing directories {start_idx} to {end_idx} ({len(selected_dirs)} directories)")

for i, image_dir in enumerate(selected_dirs, 1):
    progress_text = f"[{i}/{len(selected_dirs)}]" if len(selected_dirs) > 1 else "[1/1]"
    print(f"{progress_text} Processing {image_dir.name}...")

    ply_file_path = image_dir / "depth_scene_no_edge.ply"
    bbox_json_path = image_dir / "3dbbox.json"
    bbox_ply_path = image_dir / "bbox.ply"
    cam_params_path = image_dir / "cam_params.json"
    output_dir = image_dir
    output_name = "scene_bbox"
    verbose = verbose_global

    # Check if required files exist
    if not ply_file_path.exists():
        print(f"  Warning: PLY file not found: {ply_file_path}")
        continue
    if not bbox_json_path.exists():
        print(f"  Warning: bbox JSON file not found: {bbox_json_path}")
        continue
    if not cam_params_path.exists():
        print(f"  Warning: cam_params file not found: {cam_params_path}")
        continue
    if (output_dir / f"{output_name}.mp4").exists():
        print(f"  Skipping: output file already exists: {output_dir / f'{output_name}.mp4'}")
        continue

    # Convert bbox JSON to PLY
    if not convert_bbox_json_to_ply(bbox_json_path, bbox_ply_path):
        print(f"  Warning: Failed to convert bbox JSON to PLY")
        continue

    try:
        render_ply_with_bbox_trajectory(
            ply_file_path=ply_file_path,
            bbox_file_path=bbox_ply_path,
            cam_params_path=cam_params_path,
            output_dir=output_dir,
            output_name=output_name,
            verbose=verbose
        )
        print(f"  Completed {image_dir.name}")
    except Exception as e:
        print(f"  Error processing {image_dir.name}: {str(e)}")
    finally:
        # Clean up intermediate bbox.ply file
        if bbox_ply_path.exists():
            bbox_ply_path.unlink()

print("All processing completed!")

# # Actual function call with current paths - uncomment to run
# render_ply_with_bbox_trajectory(
#     ply_file_path="experimental_results/COCO/val/000000000632/depth_scene_no_edge.ply",
#     bbox_file_path="experimental_results/COCO/val/000000000632/bbox.ply",
#     cam_params_path="experimental_results/COCO/val/000000000632/cam_params.json",
#     output_dir="experimental_results/COCO/val/000000000632",
#     output_name="camera_trajectory",
#     verbose=False,  # Set to True if you want to see rendering progress
#     camera_offset_ratio=0.8  # Adjust ratio for more/less camera movement
# )