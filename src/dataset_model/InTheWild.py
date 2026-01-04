from PIL import Image
import numpy as np
from dataset_model.BaseScene import BaseScene


class InTheWild(BaseScene):
    def __init__(self, img_path):
        super().__init__(Image.open(img_path))