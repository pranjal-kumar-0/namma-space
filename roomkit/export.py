import json
from pathlib import Path

import numpy as np

from .walls import find_openings


def write_ply(path: Path, P: np.ndarray, rgb: np.ndarray):
    with open(path, "wb") as fh:
        fh.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
                  ).encode())
        rec = np.zeros(len(P), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        rec["p"], rec["c"] = P, rgb
        fh.write(rec.tobytes())


def write_outputs(out: Path, P, C, walls, ceiling, boxes, cams, room_yaw) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    keep = (P[:, 1] > -0.3) & (P[:, 1] < ceiling + 0.3)
    P, C = P[keep], C[keep]
    if len(P) > 400_000:
        idx = np.random.default_rng(0).choice(len(P), 400_000, replace=False)
        P, C = P[idx], C[idx]
    rgb = C[:, ::-1].astype(np.uint8)

    with open(out / "points.bin", "wb") as fh:
        fh.write(np.uint32(len(P)).tobytes())
        fh.write(P.astype(np.float32).tobytes())
        fh.write(rgb.tobytes())
    write_ply(out / "points.ply", P, rgb)

    find_openings(P, walls, ceiling)       # doorways: walk-mode lets you through these
    xz = P[:, [0, 2]]
    scene = {
        "units": "meters",
        "up": "+y",
        "ceiling_height": ceiling,
        "room_yaw": room_yaw,
        "bounds": {"min": P.min(0).tolist(), "max": P.max(0).tolist()},
        "floor_extent": {"min": np.percentile(xz, 1, 0).tolist(),
                         "max": np.percentile(xz, 99, 0).tolist()},
        "walls": walls,
        "objects": boxes,
        "cameras": cams,
    }
    (out / "scene.json").write_text(json.dumps(scene, indent=1))
    return scene
