"""Render a synthetic room walkthrough video with known geometry (for pipeline smoke tests).

Room: 5.0 x 4.0 m, 2.6 m high, x in [0,5], z in [0,4], floor y=0.
Boxes: "bed" 2.0x0.5x1.6 at x[0.3,2.3] z[0.3,1.9]; "table" 1.2x0.75x0.8 at x[3.2,4.4] z[2.6,3.4].
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

W, H, F = 480, 360, 300.0
ROOM = (np.array([0, 0, 0.0]), np.array([5, 2.6, 4.0]))
BOXES = [(np.array([0.3, 0, 0.3]), np.array([2.3, 0.5, 1.9])),
         (np.array([3.2, 0, 2.6]), np.array([4.4, 0.75, 3.4]))]


def tex(p, n, seed):
    """Procedural, view-consistent texture: multi-scale value noise on the surface."""
    out = np.zeros(len(p))
    for k, sc in enumerate([0.07, 0.2, 0.6]):
        q = np.floor(p / sc).astype(np.int64)
        h = (q[:, 0] * 73856093 ^ q[:, 1] * 19349663 ^ q[:, 2] * 83492791 ^ (seed + k) * 2654435761)
        out += ((h % 1000) / 1000.0) / (k + 1)
    return out / 1.83


def render(cam, R):
    vs, us = np.mgrid[0:H, 0:W]
    d = np.stack([(us - W / 2) / F, (vs - H / 2) / F, np.ones_like(us, float)], -1).reshape(-1, 3)
    d = d @ R  # cam->world (R is world->cam)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    o = cam
    tbest = np.full(len(d), np.inf)
    col = np.zeros((len(d), 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1 / d
        # room interior: exit distance
        t1 = (ROOM[0] - o) * inv
        t2 = (ROOM[1] - o) * inv
        tf = np.maximum(t1, t2)
        t_exit = tf.min(1)
        axis = tf.argmin(1)
        tbest = t_exit
        hit = o + d * tbest[:, None]
        base = np.array([[0.75, 0.72, 0.68], [0.55, 0.45, 0.35], [0.7, 0.75, 0.8]])
        face = np.where((axis == 1) & (d[:, 1] < 0), 1, np.where(axis == 1, 2, 0))
        col = base[face] * (0.55 + 0.45 * tex(hit, axis, 1))[:, None]
        for bi, (lo, hi) in enumerate(BOXES):
            t1 = (lo - o) * inv
            t2 = (hi - o) * inv
            tn = np.minimum(t1, t2).max(1)
            tx = np.maximum(t1, t2).min(1)
            m = (tn <= tx) & (tn > 0) & (tn < tbest)
            tbest[m] = tn[m]
            hp = o + d[m] * tn[m, None]
            c = np.array([[0.8, 0.3, 0.3], [0.3, 0.5, 0.8]])[bi]
            col[m] = c * (0.5 + 0.5 * tex(hp, None, 7 + bi))[:, None]
    shade = 1 / (1 + 0.05 * tbest)
    img = np.clip(col * shade[:, None] * 255, 0, 255).astype(np.uint8).reshape(H, W, 3)
    return img


def look(cam, target):
    z = target - cam
    z /= np.linalg.norm(z)
    x = np.cross([0, -1, 0], z)  # image +y is world down
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def main(out):
    out = Path(out)
    proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", "10", "-i", "-", "-pix_fmt", "yuv420p", str(out)],
                            stdin=subprocess.PIPE)
    n = 120
    for i in range(n):
        a = 2 * np.pi * i / n
        cam = np.array([2.5 + 1.0 * np.cos(a), 1.45, 2.0 + 0.7 * np.sin(a)])
        # look outward-ish toward the walls, slightly down, sweeping around
        tgt = np.array([2.5 + 3 * np.cos(a + 0.9), 0.9, 2.0 + 3 * np.sin(a + 0.9)])
        proc.stdin.write(render(cam, look(cam, tgt)).tobytes())
    proc.stdin.close()
    proc.wait()


if __name__ == "__main__":
    main(sys.argv[1])
