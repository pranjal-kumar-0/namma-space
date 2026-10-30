import cv2
import numpy as np

from .config import DEPTH_MODEL, MODELS_DIR, Progress
from .frame import Frame


def estimate_depth(frames: list[Frame], progress: Progress) -> float:
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    torch.set_num_threads(max(1, torch.get_num_threads() or 4))
    proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL, cache_dir=MODELS_DIR / "hf")
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL, cache_dir=MODELS_DIR / "hf").eval()

    scales = []
    for i, f in enumerate(frames):
        rgb = cv2.cvtColor(f.image, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            out = model(**proc(images=rgb, return_tensors="pt"))
        d = torch.nn.functional.interpolate(out.predicted_depth[None], size=rgb.shape[:2],
                                            mode="bilinear", align_corners=False)[0, 0].numpy()
        f.depth = d.astype(np.float32)

        if len(f.sparse_z) >= 10:
            u = np.clip(f.sparse_uv[:, 0].round().astype(int), 0, d.shape[1] - 1)
            v = np.clip(f.sparse_uv[:, 1].round().astype(int), 0, d.shape[0] - 1)
            ratio = d[v, u] / f.sparse_z
            ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
            if len(ratio) >= 10:
                f.sfm_scale = float(np.median(ratio))
                scales.append(f.sfm_scale)
        progress("depth", (i + 1) / len(frames))

    if not scales:
        raise RuntimeError("could not align depth to SfM (too few sparse points)")

    # Poses share one global scale g; each depth map is rescaled to match it, since the
    # mono-depth scale drifts frame to frame.
    g = float(np.median(scales))
    for f in frames:
        if f.sfm_scale != 1.0:
            f.depth *= g / f.sfm_scale
        f.sfm_scale = g
    return g
