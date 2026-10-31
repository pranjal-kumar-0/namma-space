"""Fake a Polycam raw-data export from an existing scan, to test the LiDAR import end to end.

    python tests/make_polycam_synthetic.py outputs/room_bedroom_tex /tmp/fake_polycam
Renders depth and colour from the textured mesh at the scan's camera positions and writes them in the
layout roomkit/polycam.py expects (OpenGL camera axes, 16-bit millimetre depth). The layout is my
reading of Polycam's export, so this checks the plumbing, not Polycam itself."""

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def main(scan: Path, dest: Path, n: int = 60, W: int = 960, H: int = 640, fov: float = 70):
    raw = (scan / "mesh.bin").read_bytes()
    nv, ni = np.frombuffer(raw[:8], np.uint32)
    V = np.frombuffer(raw, np.float32, nv * 3, 8).reshape(-1, 3)
    UV = np.frombuffer(raw, np.float32, nv * 2, 8 + nv * 12).reshape(-1, 2)
    I = np.frombuffer(raw, np.uint32, ni, 8 + nv * 20).reshape(-1, 3)
    tex = cv2.imread(str(scan / "mesh.jpg"))
    th = tex.shape[0]
    cams = json.loads((scan / "scene.json").read_text())["cameras"]
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.core.Tensor(V), o3d.core.Tensor(I.astype(np.uint32)))

    kf = dest / "keyframes"
    shutil.rmtree(dest, ignore_errors=True)
    for d in ("corrected_images", "corrected_cameras", "depth"):
        (kf / d).mkdir(parents=True)

    f = W / 2 / np.tan(np.radians(fov / 2))
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    for k, i in enumerate(np.linspace(0, len(cams) - 1, n).astype(int)):
        pos, d = np.array(cams[i]["pos"]), np.array(cams[i]["dir"])
        d /= np.linalg.norm(d)
        r = np.cross(d, [0, 1.0, 0]); r /= np.linalg.norm(r)
        u = np.cross(r, d)
        rays = d + ((xs - W / 2) / f)[..., None] * r - ((ys - H / 2) / f)[..., None] * u
        res = rc.cast_rays(o3d.core.Tensor(np.concatenate(
            [np.broadcast_to(pos, rays.shape), rays], -1).astype(np.float32)))
        pid, uv, t = res["primitive_ids"].numpy(), res["primitive_uvs"].numpy(), res["t_hit"].numpy()
        hit = pid < 4294967295
        tri = I[np.where(hit, pid, 0)]
        b = np.stack([1 - uv[..., 0] - uv[..., 1], uv[..., 0], uv[..., 1]], -1)
        tuv = (UV[tri] * b[..., None]).sum(2)
        img = np.zeros((H, W, 3), np.uint8)
        img[hit] = tex[((1 - tuv[..., 1]) * (th - 1)).astype(int)[hit], (tuv[..., 0] * (th - 1)).astype(int)[hit]]
        depth = np.where(hit, t, 0)                       # |ray z| = 1, so t is the z-depth in metres
        depth = np.where(np.random.default_rng(k).random(depth.shape) < 0.0, 0, depth)
        dsmall = cv2.resize(depth.astype(np.float32), (W // 4, H // 4), interpolation=cv2.INTER_NEAREST)

        name = f"{k:04d}"
        cv2.imwrite(str(kf / "corrected_images" / f"{name}.jpg"), img)
        cv2.imwrite(str(kf / "depth" / f"{name}.png"), np.clip(dsmall * 1000, 0, 65535).astype(np.uint16))
        c2w = np.eye(4)
        c2w[:3, :3] = np.stack([r, u, -d], 1)             # OpenGL: x right, y up, camera looks down -z
        c2w[:3, 3] = pos
        cam = {"fx": f, "fy": f, "cx": W / 2, "cy": H / 2, "width": W, "height": H, "blur_score": 100.0}
        cam.update({f"t_{a}{c}": float(c2w[a, c]) for a in range(3) for c in range(4)})
        (kf / "corrected_cameras" / f"{name}.json").write_text(json.dumps(cam))
    shutil.make_archive(str(dest), "zip", dest)
    print(f"wrote {dest} and {dest}.zip")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))