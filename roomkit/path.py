"""Camera trajectory -> evenly spaced, smoothed walk path for the viewer's guided tour."""

import numpy as np
from scipy.ndimage import gaussian_filter1d

STEP = 0.1       # metres between path samples
SMOOTH = 2.0     # gaussian sigma, in frames; removes hand shake without cutting corners


def build_walk_path(cams: list[dict]) -> dict:
    """cams: [{pos, dir}] per posed frame, in scene coordinates (metres, Y-up).

    Returns {"length": metres, "points": [{pos, dir}, ...]} sampled every STEP metres.
    """
    if len(cams) < 2:
        return {"length": 0.0, "points": []}
    pos = gaussian_filter1d(np.array([c["pos"] for c in cams], float), SMOOTH, axis=0, mode="nearest")
    look = gaussian_filter1d(np.array([c["dir"] for c in cams], float), SMOOTH, axis=0, mode="nearest")

    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])
    if s[-1] < STEP:
        return {"length": float(s[-1]), "points": []}
    u = np.append(np.arange(0.0, s[-1], STEP), s[-1])
    P = np.stack([np.interp(u, s, pos[:, k]) for k in range(3)], 1)
    D = np.stack([np.interp(u, s, look[:, k]) for k in range(3)], 1)
    D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)

    return {"length": float(s[-1]),
            "points": [{"pos": np.round(p, 3).tolist(), "dir": np.round(d, 3).tolist()}
                       for p, d in zip(P, D)]}
