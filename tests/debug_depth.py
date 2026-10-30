import pickle, sys, numpy as np
from pathlib import Path
import pipeline as pl
out = Path("outputs/synthetic_room"); work = out / "work"
cache = Path("/tmp/frames.pkl")
if cache.exists():
    frames = pickle.load(open(cache, "rb"))
else:
    frames = pl.run_sfm(work / "frames", work)
    # keep raw metric depth (undo nothing): monkeypatch scale step off
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    import cv2
    proc = AutoImageProcessor.from_pretrained(pl.DEPTH_MODEL, cache_dir=pl.MODELS_DIR / "hf")
    model = AutoModelForDepthEstimation.from_pretrained(pl.DEPTH_MODEL, cache_dir=pl.MODELS_DIR / "hf").eval()
    for f in frames:
        rgb = cv2.cvtColor(f.image, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            d = model(**proc(images=rgb, return_tensors="pt")).predicted_depth[None]
        f.depth = torch.nn.functional.interpolate(d, size=rgb.shape[:2], mode="bilinear")[0, 0].numpy()
    pickle.dump(frames, open(cache, "wb"))
for f in frames[::6]:
    u = f.sparse_uv[:, 0].round().astype(int).clip(0, f.depth.shape[1]-1); v = f.sparse_uv[:, 1].round().astype(int).clip(0, f.depth.shape[0]-1)
    dm, zs = f.depth[v, u], f.sparse_z
    s = np.median(dm / zs)
    rs = np.median(np.abs(dm / s - zs) / zs)
    A = np.stack([1/dm, np.ones_like(dm)], 1); a, b = np.linalg.lstsq(A, 1/zs, rcond=None)[0]
    ra = np.median(np.abs(1/(a/dm + b) - zs) / zs)
    print(f"{f.name} n={len(zs)} scale={s:.3f} relerr_scale={rs:.3f} relerr_affine={ra:.3f} corr={np.corrcoef(dm, zs)[0,1]:.3f}")
