import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .config import FRAME_LONG_SIDE, MAX_FRAMES, SHARPNESS_POOL


def _probe(video: Path) -> tuple[float, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "format=duration:stream=avg_frame_rate", "-of", "default=nw=1", str(video)],
        capture_output=True, text=True, check=True).stdout
    info = dict(line.split("=", 1) for line in out.split() if "=" in line)
    dur = float(info.get("duration", 0) or 0)
    num, _, den = info.get("avg_frame_rate", "0/1").partition("/")
    src_fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return dur, src_fps


def extract_frames(video: Path, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    dur, src_fps = _probe(video)

    fps = min(6.0, MAX_FRAMES / dur) if dur > 0 else 2.0
    pool = SHARPNESS_POOL
    if src_fps > 0:
        fps = min(fps, src_fps)
        pool = max(1, min(pool, int(src_fps // fps)))

    cand = out / "candidates"
    shutil.rmtree(cand, ignore_errors=True)
    cand.mkdir()
    s = FRAME_LONG_SIDE
    vf = f"fps={fps * pool:.4f},scale='if(gt(iw,ih),{s},-2)':'if(gt(iw,ih),-2,{s})'"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-vf", vf,
                    "-q:v", "2", str(cand / "%05d.jpg")], check=True)

    # keep the sharpest of each candidate group
    files = sorted(cand.glob("*.jpg"))
    for k in range(0, len(files), pool):
        group = files[k:k + pool]
        sharp = [cv2.Laplacian(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), cv2.CV_32F).var()
                 for f in group]
        group[int(np.argmax(sharp))].rename(out / f"{k // pool + 1:04d}.jpg")
    shutil.rmtree(cand)
    return sorted(out.glob("*.jpg"))
