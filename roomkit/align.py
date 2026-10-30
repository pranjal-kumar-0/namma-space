import numpy as np

from .frame import Frame


def apply_T(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    return P @ T[:3, :3].T + T[:3, 3]


def fit_plane_ransac(P, iters=400, thresh=0.03, normal_hint=None, max_angle_deg=25,
                     rng=np.random.default_rng(0)):
    best, best_n = None, 0
    cos_lim = np.cos(np.deg2rad(max_angle_deg))
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if normal_hint is not None and abs(n @ normal_hint) < cos_lim:
            continue
        inl = np.abs(P @ n + (-n @ s[0])) < thresh
        if inl.sum() > best_n:
            best_n, best = inl.sum(), (n, -n @ s[0], inl)
    return best


def _floor_height(hy: np.ndarray, cam_y: float) -> float:
    """Lowest dense horizontal slab below the cameras."""
    lo = np.percentile(hy, 0.5)
    cand = hy[(hy < cam_y - 0.5) & (hy >= lo)]
    if len(cand) < 200:
        return float(np.percentile(hy, 2))
    hist, edges = np.histogram(cand, bins=np.arange(cand.min(), cand.max() + 0.04, 0.02))
    i = next(i for i in range(len(hist)) if hist[i] >= hist.max() * 0.3)
    return float((edges[i] + edges[i + 1]) / 2)


def gravity_align(P: np.ndarray, frames: list[Frame]) -> np.ndarray:
    """4x4 transform: world metres -> Y-up with the floor at y=0."""
    # Camera up in world is -R^T[:,1] (image +y points down).
    up = np.mean([-f.R[1] for f in frames], 0)
    up /= np.linalg.norm(up)

    cam_h = np.array([f.center() @ up for f in frames])
    sub = P[np.random.default_rng(0).choice(len(P), min(len(P), 60000), replace=False)]
    below = sub[sub @ up < np.median(cam_h) - 0.5]
    if len(below) > 500:
        res = fit_plane_ransac(below, normal_hint=up, thresh=0.04)
        if res is not None and res[2].sum() > 300:
            n = res[0]
            up = n if n @ up > 0 else -n
            Q = below[res[2]]
            _, _, vt = np.linalg.svd(Q - Q.mean(0))
            n2 = vt[2] if vt[2] @ up > 0 else -vt[2]
            if n2 @ up > np.cos(np.deg2rad(5)):
                up = n2

    y = up
    x = np.cross(y, [0, 0, 1.0])
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(y, [1.0, 0, 0])
    x /= np.linalg.norm(x)
    R = np.stack([x, y, np.cross(x, y)])

    T = np.eye(4)
    T[:3, :3] = R
    cam_y = np.array([(R @ f.center())[1] for f in frames])
    T[1, 3] = -_floor_height((P @ R.T)[:, 1], np.median(cam_y))
    return T
