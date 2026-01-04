import os
import warnings
import argparse
import time
from PIL import Image
import torch

warnings.simplefilter('ignore', category=UserWarning)
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)

from infer import Text2Image, Removebg, Image2Views, Views2Mesh, GifRenderer
from third_party.mesh_baker import MeshBaker
from third_party.check import check_bake_available

st = time.time()

rembg_model = Removebg()
image_to_views_model = Image2Views(
    device='cuda:0', 
    use_lite=False,
    save_memory=False,
    std_pretrain='../third_party/Hunyuan3D-1/weights/mvd_std',
)
# from third_party.mesh_baker import MeshBaker

# mesh_baker = MeshBaker(
#     device = 'cuda:0',
#     align_times = 3
# )
  
views_to_mesh_model = Views2Mesh(
    '../third_party/Hunyuan3D-1/svrm/configs/svrm.yaml', 
    '../third_party/Hunyuan3D-1/weights/svrm/svrm.safetensors', 
    'cuda:0', 
    use_lite=False,
    save_memory=False
)
print(f"Init Models cost {time.time()-st}s")


def infer_on_image(input_path, out_path, gen_seed = 0, gen_steps = 50, max_faces_num = 90000, do_texture_mapping = True):

    print("arigattoooo")
    
    os.makedirs(out_path, exist_ok=True)
    res_rgb_pil = Image.open(input_path)

    # stage, remove back ground
    # res_rgba_pil = rembg_model(res_rgb_pil)
    res_rgb_pil.save(os.path.join(out_path, "img_nobg.png"))

    # stage, image to views
    (views_grid_pil, cond_img), view_pil_list = image_to_views_model(
        res_rgb_pil,
        seed = gen_seed,
        steps = gen_steps
    )
    views_grid_pil.save(os.path.join(out_path, "views.jpg"))

    # stage , views to mesh
    views_to_mesh_model(
        views_grid_pil, 
        cond_img, 
        seed = gen_seed,
        target_face_count = max_faces_num,
        save_folder = out_path,
        do_texture_mapping = do_texture_mapping
    )




# if __name__ == "__main__":

#     infer('./demos/bus.png', './outputs/test/bus2/')
