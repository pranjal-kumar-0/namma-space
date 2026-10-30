import cv2
import numpy as np

from .config import FUSE_VOXEL
from .frame import Frame

_fp_cache: dict[int, np.ndarray] = {}


def reset_caches():
    _fp_cache.clear()


def flying_pixel_mask(depth: np.ndarray) -> np.ndarray:
    key = id(depth)
    if key not in _fp_cache:
        gx = np.abs(cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3))
        rel = (gx + gy) / np.maximum(depth, 1e-3)
        _fp_cache[key] = rel < 0.25
    return _fp_cache[key]


def backproject(f: Frame, mask: np.ndarray | None = None, step: int = 1,
                max_depth: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """World points (metres) and BGR colours for a frame's valid pixels."""
    h, w = f.depth.shape
    vs, us = np.mgrid[0:h:step, 0:w:step]
    d = f.depth[vs, us]
    valid = (d > 0.1) & (d < max_depth) & flying_pixel_mask(f.depth)[vs, us]
    if mask is not None:
        valid &= mask[vs, us]
    us, vs, d = us[valid], vs[valid], d[valid]

    K = f.K
    x = (us - K[0, 2]) / K[0, 0] * d
    y = (vs - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], 1)
    pw = (pc - f.t * f.sfm_scale) @ f.R          # R^T (pc - t)
    return pw, f.image[vs, us]


def voxel_down(p: np.ndarray, c: np.ndarray | None, vox: float):
    keys = np.floor(p / vox).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return p[idx], (c[idx] if c is not None else None)


def fuse(frames: list[Frame]) -> tuple[np.ndarray, np.ndarray]:
    pts, cols = [], []
    for f in frames:
        p, c = backproject(f, step=3)
        pts.append(p)
        cols.append(c)
    return voxel_down(np.concatenate(pts), np.concatenate(cols), FUSE_VOXEL)
