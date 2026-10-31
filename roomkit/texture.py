"""Bake a photo texture onto the room mesh: UV-unwrap it, then colour every texel from the
best-facing frame that sees it. Much sharper than per-vertex colours."""

from pathlib import Path

import cv2
import numpy as np

from .config import ATLAS_SIZE, MESH_MAX_DEPTH, TEX_TRIS, Progress
from .frame import Frame


def bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fx, fy = (u - x0)[:, None], (v - y0)[:, None]
    g = lambda y, x: img[y, x].astype(np.float32)
    return ((g(y0, x0) * (1 - fx) + g(y0, x0 + 1) * fx) * (1 - fy)
            + (g(y0 + 1, x0) * (1 - fx) + g(y0 + 1, x0 + 1) * fx) * fy)


def bake_texture(mesh, frames: list[Frame], T: np.ndarray, out: Path, progress: Progress) -> dict:
    """mesh: open3d legacy mesh in the aligned frame. Writes out/mesh.bin and out/mesh.jpg."""
    import open3d as o3d

    m = o3d.geometry.TriangleMesh(mesh)
    if len(m.triangles) > TEX_TRIS:
        m = m.simplify_quadric_decimation(TEX_TRIS)
    m.remove_degenerate_triangles()
    m.remove_non_manifold_edges()               # compute_uvatlas needs a manifold mesh
    m.remove_unreferenced_vertices()
    m.compute_vertex_normals()

    tm = o3d.t.geometry.TriangleMesh.from_legacy(m)
    tm.compute_uvatlas(size=ATLAS_SIZE, gutter=2.0, max_stretch=0.3, parallel_partitions=32)
    progress("texture", 0.1)
    maps = tm.bake_vertex_attr_textures(ATLAS_SIZE, {"positions", "normals", "colors"},
                                        margin=2.0, fill=0.0, update_material=False)
    pos_img, nrm_img, col_img = (maps[k].numpy() for k in ("positions", "normals", "colors"))
    covered = np.linalg.norm(nrm_img, axis=2) > 0.5          # texels inside a chart (+ margin)
    ys, xs = np.nonzero(covered)
    pos = pos_img[ys, xs].astype(np.float32)
    nrm = nrm_img[ys, xs].astype(np.float32)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)

    # start from the vertex colours (covers texels no frame sees), then overwrite from photos
    colors = np.clip(col_img[ys, xs][:, ::-1] * 255, 0, 255).astype(np.float32)   # RGB -> BGR
    best = np.zeros(len(pos), np.float32)
    T_inv = np.linalg.inv(T)
    for i, f in enumerate(frames):
        h, w = f.depth.shape
        ext = np.eye(4)
        ext[:3, :3], ext[:3, 3] = f.R, f.t * f.sfm_scale
        ext = ext @ T_inv                                    # aligned -> camera
        R, t = ext[:3, :3].astype(np.float32), ext[:3, 3].astype(np.float32)
        centre = (-R.T @ t).astype(np.float32)
        pc = pos @ R.T + t
        z = pc[:, 2]
        K = f.K
        u = K[0, 0] * pc[:, 0] / np.maximum(z, 1e-3) + K[0, 2]
        v = K[1, 1] * pc[:, 1] / np.maximum(z, 1e-3) + K[1, 2]
        ok = (z > 0.2) & (z < MESH_MAX_DEPTH) & (u > 1) & (u < w - 2) & (v > 1) & (v < h - 2)
        idx = np.nonzero(ok)[0]
        if not len(idx):
            continue
        ui, vi, zi = u[idx], v[idx], z[idx]
        d = f.depth[vi.astype(int), ui.astype(int)]
        seen = (d > 0) & (np.abs(zi - d) < 0.15 + 0.03 * zi)           # not hidden behind something nearer
        view = centre - pos[idx]
        view /= np.linalg.norm(view, axis=1, keepdims=True) + 1e-9
        cos = (nrm[idx] * view).sum(1)
        edge = np.minimum(np.minimum(ui, w - ui), np.minimum(vi, h - vi)) / (0.1 * min(w, h))
        score = np.where(seen & (cos > 0.2), cos * np.clip(edge, 0.05, 1) / zi, 0)
        win = score > best[idx]
        if win.any():
            sel = idx[win]
            colors[sel] = bilinear(f.image, ui[win], vi[win])
            best[sel] = score[win]
        progress("texture", 0.1 + 0.8 * (i + 1) / len(frames))

    atlas = np.zeros((ATLAS_SIZE, ATLAS_SIZE, 3), np.uint8)
    atlas[ys, xs] = colors.astype(np.uint8)
    cv2.imwrite(str(out / "mesh.jpg"), atlas, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # geometry: weld corners that share a vertex and a UV
    tri = np.asarray(m.triangles)                                      # (T, 3)
    uv = tm.triangle["texture_uvs"].numpy().reshape(-1, 2)             # (3T, 2)
    key = np.column_stack([tri.reshape(-1), np.round(uv * 2**20).astype(np.int64)])
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    V = np.asarray(m.vertices, np.float32)[tri.reshape(-1)[first]]
    UV = uv[first].astype(np.float32)
    idx = inv.reshape(-1).astype(np.uint32)
    with open(out / "mesh.bin", "wb") as fh:
        fh.write(np.array([len(V), len(idx)], np.uint32).tobytes())
        fh.write(V.tobytes() + UV.tobytes() + idx.tobytes())
    progress("texture", 1)
    return {"triangles": int(len(tri)), "vertices": int(len(V)),
            "photo_texels": float((best > 0).mean())}
