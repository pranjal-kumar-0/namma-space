"""The per-image record that flows through the pipeline."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Frame:
    name: str
    image: np.ndarray            # BGR
    K: np.ndarray
    R: np.ndarray                # world->cam
    t: np.ndarray
    depth: np.ndarray | None = None
    sfm_scale: float = 1.0       # metres per SfM unit
    sparse_uv: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    sparse_z: np.ndarray = field(default_factory=lambda: np.empty((0,), np.float32))
    dets: list = field(default_factory=list)

    def center(self) -> np.ndarray:
        return -(self.R.T @ (self.t * self.sfm_scale))
