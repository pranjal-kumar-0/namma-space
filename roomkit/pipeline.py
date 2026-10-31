"""Phone video -> floor, walls and furniture boxes. CPU-only.

Stages: frames -> SfM poses -> metric depth -> fused point cloud -> gravity align
-> walls + ceiling -> furniture boxes.
"""

import json
import shutil
from pathlib import Path

from .align import apply_T, gravity_align
from .config import Progress
from .depth import estimate_depth
from .export import write_outputs
from .frames import extract_frames
from .fusion import fuse, reset_caches
from .furniture import detect_objects, fit_boxes, lift_detections, merge_observations
from .mesh import build_mesh
from .path import build_walk_path
from .sfm import run_sfm
from .walls import ceiling_height, detect_walls


def process(video, out, progress: Progress | None = None) -> dict:
    progress = progress or (lambda stage, frac: None)
    out = Path(out)
    work = out / "work"
    frames_dir = work / "frames"
    shutil.rmtree(frames_dir, ignore_errors=True)

    progress("frames", 0)
    imgs = extract_frames(Path(video), frames_dir)
    if len(imgs) < 5:
        raise RuntimeError(f"only {len(imgs)} frames extracted; video too short?")
    progress("frames", 1)

    progress("poses", 0)
    frames = run_sfm(frames_dir, work)
    if len(frames) < 5:
        raise RuntimeError(f"only {len(frames)}/{len(imgs)} frames got camera poses")
    progress("poses", 1)

    estimate_depth(frames, progress)

    progress("fusion", 0)
    P, C = fuse(frames)
    T = gravity_align(P, frames)
    P = apply_T(T, P)
    progress("fusion", 1)

    progress("walls", 0)
    ceil = ceiling_height(P)
    walls, yaw = detect_walls(P, ceil)
    progress("walls", 1)

    detect_objects(frames, progress)
    boxes = fit_boxes(merge_observations(lift_detections(frames, T)), yaw, len(frames))

    mesh_info = None
    try:
        mesh_info = build_mesh(frames, T, P, ceil, walls, out / "mesh.ply", progress)
    except Exception as e:          # the point cloud is still a usable result
        print(f"mesh step failed: {e}")

    cams = [{"pos": apply_T(T, f.center()[None])[0].tolist(),
             "dir": (T[:3, :3] @ f.R[2]).tolist()} for f in frames]

    scene = write_outputs(out, P, C, walls, ceil, boxes, cams, yaw)
    scene["walk_path"] = build_walk_path(cams)
    scene["mesh"] = "mesh.ply" if mesh_info else None
    scene["stats"] = {"frames": len(imgs), "posed": len(frames), "points": int(len(P))}
    (out / "scene.json").write_text(json.dumps(scene, indent=1))
    reset_caches()
    progress("done", 1)
    return scene
