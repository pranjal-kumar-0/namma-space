import cv2
import numpy as np


def rasterize(xz, cell):
    x0, z0 = xz.min(0) - cell
    ij = np.floor((xz - [x0, z0]) / cell).astype(int)
    H = ij.max(0) + 2
    img = np.zeros((H[1], H[0]), np.float32)
    np.add.at(img, (ij[:, 1], ij[:, 0]), 1)
    return img


def dominant_yaw(xz: np.ndarray) -> float:
    """Manhattan orientation in [0, pi/2) from floor-plan gradients."""
    img = rasterize(xz, 0.05)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5)
    gz = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5)
    mag = np.hypot(gx, gz)
    strong = mag > mag.max() * 0.1
    ang = np.arctan2(gz, gx)[strong] % (np.pi / 2)
    w = mag[strong]
    if len(ang) == 0:
        return 0.0
    s = (w * np.sin(4 * ang)).sum()
    c = (w * np.cos(4 * ang)).sum()
    return float((np.arctan2(s, c) / 4) % (np.pi / 2))


def ceiling_height(P: np.ndarray) -> float:
    hy = P[:, 1]
    top = hy[hy > 1.8]
    if len(top) > 300:
        hist, edges = np.histogram(top, bins=np.arange(1.8, top.max() + 0.04, 0.03))
        if hist.max() > 0.02 * len(top) and hist.max() > 150:
            i = len(hist) - 1 - np.argmax(hist[::-1] >= hist.max() * 0.4)
            return float(np.clip((edges[i] + edges[i + 1]) / 2, 2.1, 4.0))
    return float(np.clip(np.percentile(hy, 99.5), 2.4, 3.2))


def detect_walls(P: np.ndarray, ceiling: float) -> tuple[list[dict], float]:
    """Tall vertical structure, fitted as Manhattan-snapped 2D segments."""
    band = P[(P[:, 1] > 0.25) & (P[:, 1] < min(ceiling - 0.2, 2.3))]
    if len(band) < 500:
        return [], 0.0
    xz = band[:, [0, 2]]
    hbin = np.floor(band[:, 1] / 0.15).astype(int)

    cell = 0.06
    origin = xz.min(0)
    ij = np.floor((xz - origin) / cell).astype(int)
    key = ij[:, 0].astype(np.int64) * 100000 + ij[:, 1]
    uk = np.unique(np.stack([key, hbin], 1), axis=0)
    cells, span = np.unique(uk[:, 0], return_counts=True)
    nbins = hbin.max() - hbin.min() + 1
    tall = cells[span >= max(4, 0.55 * nbins)]       # a wall cell spans many heights
    if len(tall) < 20:
        return [], 0.0
    pts = np.stack([(tall // 100000) * cell + origin[0] + cell / 2,
                    (tall % 100000) * cell + origin[1] + cell / 2], 1)

    yaw = dominant_yaw(pts)
    dirs = [np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])]
    walls = []
    remaining = pts.copy()
    rng = np.random.default_rng(1)
    for _ in range(12):
        if len(remaining) < 15:
            break
        best = None
        for _ in range(300):
            a, b = remaining[rng.choice(len(remaining), 2, replace=False)]
            v = b - a
            if np.linalg.norm(v) < 0.3:
                continue
            v /= np.linalg.norm(v)
            for dvec in dirs:
                if abs(v @ dvec) > np.cos(np.deg2rad(12)):
                    v = dvec
                    break
            nrm = np.array([-v[1], v[0]])
            off = (remaining - a) @ nrm
            inl = np.abs(off) < 0.08
            if best is None or inl.sum() > best[2].sum():
                best = (a + nrm * np.median(off[inl]), v, inl)
        if best is None or best[2].sum() < 15:
            break
        a, v, inl = best
        proj = np.sort((remaining[inl] - a) @ v)
        cuts = np.where(np.diff(proj) > 0.6)[0]       # split at door/occlusion gaps
        for s, e in zip(np.r_[0, cuts + 1], np.r_[cuts, len(proj) - 1]):
            if proj[e] - proj[s] >= 0.5 and e - s >= 8:
                p1, p2 = a + v * proj[s], a + v * proj[e]
                walls.append({"x1": float(p1[0]), "z1": float(p1[1]),
                              "x2": float(p2[0]), "z2": float(p2[1]), "height": float(ceiling)})
        remaining = remaining[~inl]
    return walls, yaw
