import importlib
from dataset_model.BaseScene import BaseScene


def get_scene(name, attributes) -> BaseScene:
    module = importlib.import_module('dataset_model.' + name)
    return getattr(module, name)(**attributes)