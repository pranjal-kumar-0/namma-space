"""Load a LiDAR scan exported from Polycam ("Raw Data": keyframe images, camera poses and LiDAR depth
maps) as pipeline Frames. The poses are metric and the depth is measured, so SfM and the depth
network are skipped and everything downstream (fusion, walls, furniture, mesh, texture) is shared.

Expected layout (a folder or a .zip of it; the folder may sit one or two levels down):
    keyframes/images/<id>.jpg   or  keyframes/corrected_images/<id>.jpg   (the latter is preferred)
    keyframes/cameras/<id>.json     fx fy cx cy [width height] and t_00..t_23 = 3x4 camera-to-world
    keyframes/depth/<id>.png        16-bit millimetres (float images / .npy in metres also accepted)
    keyframes/confidence/<id>.png   optional, low-confidence depth is dropped
"""

import json
import re
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np

from .config import FRAME_LONG_SIDE, LIDAR_MAX_FRAMES, Progress
from .frame import Frame

DEPTH_EXT = (".png", ".tiff", ".tif", ".exr", ".npy")


def is_polycam(path: Path) -> bool:
    if path.is_dir():
        return _find_root(path) is not None
    if path.suffix.lower() == ".zip" and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            return any(re.search(r"(^|/)keyframes/(corrected_)?cameras/", n) for n in z.namelist())
    return False


def _find_root(d: Path) -> Path | None:
    for cand in [d, *[p for p in d.iterdir() if p.is_dir()], *[q for p in d.iterdir() if p.is_dir()
                                                              for q in p.iterdir() if q.is_dir()]]:
        if (cand / "keyframes").is_dir():
            return cand / "keyframes"
        if (cand / "cameras").is_dir() or (cand / "corrected_cameras").is_dir():
            return cand
    return None


def _unzip(zpath: Path, dest: Path) -> Path:
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            if not (dest / n).resolve().is_relative_to(dest.resolve()):
                raise RuntimeError("unsafe path in zip")
        z.extractall(dest)
    return dest


def _pick(root: Path, *names: str) -> Path | None:
    return next((root / n for n in names if (root / n).is_dir()), None)


def _read_pose(cam: dict) -> np.ndarray:
    """4x4 camera-to-world from the camera json (t_00..t_23, or a flat 12/16 'transform')."""
    if "t_00" in cam:
        M = np.array([[cam[f"t_{r}{c}"] for c in range(4)] for r in range(3)], float)
    else:
        flat = np.array(next(cam[k] for k in ("transform", "pose", "matrix") if k in cam), float).ravel()
        M = flat.reshape(-1, 4)[:3] if flat.size in (12, 16) else None
        if M is None:
            raise ValueError("unrecognised pose format")
    return np.vstack([M, [0, 0, 0, 1]])


def _read_depth(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        d = np.load(path).astype(np.float32)
    else:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f"cannot read {path.name}")
        if raw.ndim == 3:
            raw = raw[:, :, 0]
        d = raw.astype(np.float32)
        if raw.dtype == np.uint16 or raw.dtype == np.uint8:
            d /= 1000.0                                  # millimetres -> metres
    return d


def _gl_to_cv(c2w: np.ndarray) -> np.ndarray:
    return c2w @ np.diag([1.0, -1.0, -1.0, 1.0])         # ARKit/OpenGL camera looks down -Z, Y up


def _make_frame(name, image, K, c2w_cv, depth) -> Frame:
    R = c2w_cv[:3, :3].T
    return Frame(name=name, image=image, K=K, R=R, t=-R @ c2w_cv[:3, 3], depth=depth, sfm_scale=1.0)


def _consistency(frames: list[Frame]) -> float:
    """Fraction of one frame's depth points that a neighbouring frame's depth agrees with."""
    hit = tot = 0
    gap = max(3, len(frames) // 6)               # far enough apart that a wrong axis convention shows
    for i in range(0, len(frames) - gap, max(1, len(frames) // 12)):
        a, b = frames[i], frames[i + gap]
        h, w = a.depth.shape
        vs, us = np.mgrid[4:h:12, 4:w:12]
        d = a.depth[vs, us]
        ok = (d > 0.3) & (d < 6)
        us, vs, d = us[ok], vs[ok], d[ok]
        pc = np.stack([(us - a.K[0, 2]) / a.K[0, 0] * d, (vs - a.K[1, 2]) / a.K[1, 1] * d, d], 1)
        pw = (pc - a.t) @ a.R
        pb = pw @ b.R.T + b.t
        z = pb[:, 2]
        u = b.K[0, 0] * pb[:, 0] / np.maximum(z, 1e-3) + b.K[0, 2]
        v = b.K[1, 1] * pb[:, 1] / np.maximum(z, 1e-3) + b.K[1, 2]
        bh, bw = b.depth.shape
        vis = (z > 0.2) & (u >= 0) & (u < bw - 1) & (v >= 0) & (v < bh - 1)
        db = b.depth[v[vis].astype(int), u[vis].astype(int)]
        valid = db > 0
        hit += int((np.abs(db[valid] - z[vis][valid]) < 0.04 * z[vis][valid] + 0.03).sum())
        tot += int(vis.sum())
    return hit / tot if tot > 50 else 0.0


def load_polycam(path: Path, work: Path, progress: Progress) -> list[Frame]:
    progress("frames", 0)
    path = Path(path)
    base = _unzip(path, work / "polycam") if path.is_file() else path
    root = _find_root(base)
    if root is None:
        raise RuntimeError("This doesn't look like a Polycam raw-data export (no keyframes folder).")

    cam_dir = _pick(root, "corrected_cameras", "cameras")
    img_dir = _pick(root, "corrected_images", "images")
    depth_dir = _pick(root, "depth")
    conf_dir = _pick(root, "confidence")
    if not (cam_dir and img_dir):
        raise RuntimeError("The export has no camera poses or images.")
    if not depth_dir:
        raise RuntimeError("The export has no LiDAR depth. In Polycam, scan with LiDAR and export "
                           "'Raw Data'.")

    ids = sorted(p.stem for p in cam_dir.glob("*.json"))
    if len(ids) > LIDAR_MAX_FRAMES:                     # spread evenly along the scan
        ids = [ids[i] for i in np.linspace(0, len(ids) - 1, LIDAR_MAX_FRAMES).astype(int)]

    entries = []
    for k, fid in enumerate(ids):
        imgp = next((img_dir / f"{fid}{e}" for e in (".jpg", ".jpeg", ".png") if (img_dir / f"{fid}{e}").exists()), None)
        depp = next((depth_dir / f"{fid}{e}" for e in DEPTH_EXT if (depth_dir / f"{fid}{e}").exists()), None)
        if imgp is None or depp is None:
            continue
        cam = json.loads((cam_dir / f"{fid}.json").read_text())
        img = cv2.imread(str(imgp))
        d = _read_depth(depp)
        if conf_dir:
            cp = next((conf_dir / f"{fid}{e}" for e in DEPTH_EXT if (conf_dir / f"{fid}{e}").exists()), None)
            if cp is not None and cp.suffix != ".npy":
                conf = cv2.imread(str(cp), cv2.IMREAD_UNCHANGED)
                if conf is not None and conf.ndim == 2 and conf.max() > conf.min():
                    conf = cv2.resize(conf, d.shape[::-1], interpolation=cv2.INTER_NEAREST)
                    d = np.where(conf >= conf.max() / 2, d, 0)
        ih, iw = img.shape[:2]
        if abs(d.shape[1] / d.shape[0] - iw / ih) > 0.05:
            raise RuntimeError(f"depth map {d.shape[::-1]} and image {(iw, ih)} have different shapes")
        W0, H0 = cam.get("width", iw), cam.get("height", ih)
        K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]], float)
        K[0] *= iw / W0
        K[1] *= ih / H0
        s = FRAME_LONG_SIDE / max(iw, ih)
        if s < 1:
            img = cv2.resize(img, (round(iw * s), round(ih * s)), interpolation=cv2.INTER_AREA)
            K[:2] *= s
        d = cv2.resize(d, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        d[~np.isfinite(d)] = 0
        entries.append((fid, img, K, _read_pose(cam), d))
        progress("frames", 0.9 * (k + 1) / len(ids))
    if len(entries) < 5:
        raise RuntimeError(f"only {len(entries)} usable keyframes (need image, camera and depth)")

    # The camera axis convention isn't documented; keep whichever makes the depth maps agree.
    cands = {"opengl": [_make_frame(n, i, K, _gl_to_cv(p), d) for n, i, K, p, d in entries],
             "opencv": [_make_frame(n, i, K, p, d) for n, i, K, p, d in entries]}
    score = {k: _consistency(v) for k, v in cands.items()}
    best = max(score, key=score.get)
    print(f"polycam: {len(entries)} keyframes, camera convention {best} "
          f"(depth agreement {score})")
    if score[best] < 0.3:
        raise RuntimeError("The depth maps and poses don't agree with each other, so the scan can't "
                           "be fused. Re-export the raw data, or send me a sample to check the format.")
    progress("frames", 1)
    return cands[best]
