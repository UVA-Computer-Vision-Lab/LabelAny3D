"""RoMa matching backend.

Install: pip install romatch
"""

import numpy as np


class RomaBackend:

    def __init__(self, device, num_matches=5000, weights='outdoor'):
        from romatch import roma_outdoor, roma_indoor

        self.device = device
        self.num_matches = num_matches
        builder = roma_outdoor if weights == 'outdoor' else roma_indoor
        self.model = builder(device=device)

        # The fused local-correlation CUDA extension is an optional RoMa extra
        # that needs nvcc; fall back to the pure-torch path without it.
        if not _fused_local_corr_available():
            for module in self.model.modules():
                if hasattr(module, 'use_custom_corr'):
                    module.use_custom_corr = False

    def match_paths(self, path_0, path_1, size=512):
        warp, certainty = self.model.match(path_0, path_1, device=self.device)
        matches, _ = self.model.sample(warp, certainty, num=self.num_matches)
        kpts0, kpts1 = self.model.to_pixel_coordinates(matches, size, size, size, size)
        return kpts0.cpu().numpy().astype(np.float64), kpts1.cpu().numpy().astype(np.float64)


def _fused_local_corr_available():
    import importlib.util
    return importlib.util.find_spec('local_corr') is not None
