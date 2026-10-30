import shutil
from pathlib import Path

import cv2
import numpy as np

from .frame import Frame


def run_sfm(image_dir: Path, work: Path) -> list[Frame]:
    import pycolmap

    db = work / "database.db"
    sparse = work / "sparse"
    db.unlink(missing_ok=True)
    shutil.rmtree(sparse, ignore_errors=True)
    sparse.mkdir(parents=True)

    reader = pycolmap.ImageReaderOptions(camera_model="SIMPLE_RADIAL")
    extract = pycolmap.FeatureExtractionOptions()
    extract.sift.max_num_features = 8192
    extract.sift.peak_threshold = 0.004          # catch corners on plain walls
    pycolmap.extract_features(db, image_dir, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=extract,
                              device=pycolmap.Device.cpu)
    pycolmap.match_sequential(db, pairing_options=pycolmap.SequentialPairingOptions(overlap=15),
                              device=pycolmap.Device.cpu)

    # relaxed init: handheld baselines are too small for the defaults
    opts = pycolmap.IncrementalPipelineOptions()
    opts.mapper.init_min_tri_angle = 4.0
    opts.mapper.init_min_num_inliers = 50
    opts.mapper.abs_pose_min_num_inliers = 15
    opts.min_model_size = 3
    recs = pycolmap.incremental_mapping(db, image_dir, sparse, options=opts)
    if not recs:
        raise RuntimeError("COLMAP could not register the video frames. Try a slower, "
                           "steadier video with more texture in view.")

    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    frames = [_to_frame(img, rec, image_dir) for img in rec.images.values() if img.has_pose]
    frames.sort(key=lambda f: f.name)
    return frames


def _to_frame(img, rec, image_dir: Path) -> Frame:
    pose = img.cam_from_world
    pose = pose() if callable(pose) else pose
    M = pose.matrix()
    cam = img.camera
    if cam.model_name not in ("SIMPLE_RADIAL", "SIMPLE_PINHOLE"):
        raise RuntimeError(f"unexpected camera model {cam.model_name}")

    K = cam.calibration_matrix()
    bgr = cv2.imread(str(image_dir / img.name))
    k1 = cam.params[3] if len(cam.params) > 3 else 0.0
    if abs(k1) > 1e-4:
        bgr = cv2.undistort(bgr, K, np.array([k1, 0, 0, 0], np.float64))

    f = Frame(img.name, bgr, K, M[:, :3].copy(), M[:, 3].copy())
    uv, z = [], []
    for p2d in img.points2D:
        if p2d.has_point3D():
            zc = (f.R @ rec.points3D[p2d.point3D_id].xyz + f.t)[2]
            if zc > 0:
                uv.append(p2d.xy)
                z.append(zc)
    f.sparse_uv = np.array(uv, np.float32).reshape(-1, 2)
    f.sparse_z = np.array(z, np.float32)
    return f
