"""CLI + stable import surface. Logic lives in the roomkit package."""

from pathlib import Path

from roomkit import process
from roomkit.config import ROOT

__all__ = ["process"]


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("-o", "--out", help="output dir (default outputs/<video name>)")
    a = ap.parse_args()

    v = Path(a.video)
    o = Path(a.out) if a.out else ROOT / "outputs" / v.stem
    t0 = time.time()
    last = [""]

    def pr(stage, frac):
        if stage != last[0] or frac >= 1:
            print(f"[{time.time() - t0:6.1f}s] {stage:8s} {frac * 100:5.1f}%", flush=True)
        last[0] = stage

    s = process(v, o, pr)
    print(f"walls={len(s['walls'])} objects={len(s['objects'])} -> {o}/scene.json")
    for b in s["objects"]:
        print(f"  {b['label']:12s} {b['size'][0]:.2f}x{b['size'][2]:.2f}x{b['size'][1]:.2f} m  views={b['views']}")
