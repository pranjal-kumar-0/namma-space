"""Fuse the per-frame depth maps into a coloured triangle mesh (TSDF), so the room reads as
continuous surfaces instead of loose points."""

from pathlib import Path

import numpy as np

from .config import MESH_MAX_DEPTH, MESH_MAX_TRIS, MESH_RES, SNAP_TOL, Progress
from .frame import Frame
from .fusion import flying_pixel_mask
from .texture import bake_texture


def snap_to_planes(V: np.ndarray, N: np.ndarray, walls: list[dict], ceiling: float,
                   tol: float = SNAP_TOL) -> int:
    """Flatten floor, ceiling and detected walls in place: a vertex close to a plane and facing it
    (so furniture surfaces are left alone) is projected onto it. Returns how many vertices moved."""
    moved = np.zeros(len(V), bool)

    for h, ny in ((0.0, 1), (ceiling, -1)):         # floor faces up, ceiling faces down
        m = (np.abs(V[:, 1] - h) < tol) & (N[:, 1] * ny > 0.5)
        V[m, 1] = h
        moved |= m

    for w in walls:
        a = np.array([w["x1"], w["z1"]])
        v = np.array([w["x2"], w["z2"]]) - a
        length = np.linalg.norm(v)
        v /= length
        n = np.array([-v[1], v[0]])
        rel = V[:, [0, 2]] - a
        d, t = rel @ n, rel @ v
        m = ((np.abs(d) < tol) & (t > -tol) & (t < length + tol)
             & (V[:, 1] > -tol) & (V[:, 1] < ceiling + tol)
             & (np.abs(N[:, [0, 2]] @ n) > 0.5))
        V[m, 0] -= d[m] * n[0]                      # sequential projection puts corners on both walls
        V[m, 2] -= d[m] * n[1]
        moved |= m
    return int(moved.sum())


def build_mesh(frames: list[Frame], T: np.ndarray, P: np.ndarray, ceiling: float, walls: list[dict],
               path: Path, progress: Progress) -> dict:
    """T: world metres -> aligned (Y-up, floor at 0). P: the aligned point cloud, used for the bounds."""
    import open3d as o3d

    # Dense volume in the aligned frame (open3d's sparse volumes returned empty meshes here).
    lo = np.percentile(P[:, [0, 2]], 1, 0) - 0.2
    hi = np.percentile(P[:, [0, 2]], 99, 0) + 0.2
    side = float(max((hi - lo).max(), ceiling + 0.4))
    voxel = side / MESH_RES
    origin = np.array([lo[0], -0.2, lo[1]])
    vol = o3d.pipelines.integration.UniformTSDFVolume(
        length=side, resolution=MESH_RES, sdf_trunc=4 * voxel,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8, origin=origin)

    T_inv = np.linalg.inv(T)
    for i, f in enumerate(frames):
        h, w = f.depth.shape
        depth = np.where(flying_pixel_mask(f.depth), f.depth, 0).astype(np.float32)
        color = np.ascontiguousarray(f.image[:, :, ::-1])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color), o3d.geometry.Image(depth), depth_scale=1.0,
            depth_trunc=MESH_MAX_DEPTH, convert_rgb_to_intensity=False)
        K = f.K
        intr = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        ext = np.eye(4)                              # world(metres) -> camera, as in fusion.backproject
        ext[:3, :3], ext[:3, 3] = f.R, f.t * f.sfm_scale
        vol.integrate(rgbd, intr, ext @ T_inv)       # aligned -> camera
        progress("mesh", 0.8 * (i + 1) / len(frames))

    mesh = vol.extract_triangle_mesh()
    y = np.asarray(mesh.vertices)[:, 1]
    mesh.remove_vertices_by_mask((y < -0.3) | (y > ceiling + 0.3))

    # drop floating specks: keep clusters with at least 2% of the triangles
    cl, n_per, _ = mesh.cluster_connected_triangles()
    cl, n_per = np.asarray(cl), np.asarray(n_per)
    if len(n_per):
        mesh.remove_triangles_by_mask(n_per[cl] < max(50, 0.02 * len(cl)))
        mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()
    V, N = np.asarray(mesh.vertices).copy(), np.asarray(mesh.vertex_normals)
    snap_to_planes(V, N, walls, ceiling)
    mesh.vertices = o3d.utility.Vector3dVector(V)

    if len(mesh.triangles) > MESH_MAX_TRIS:
        mesh = mesh.simplify_quadric_decimation(MESH_MAX_TRIS)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if not len(mesh.triangles):
        raise RuntimeError("depth fusion produced an empty mesh")

    o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False, write_vertex_normals=False)
    info = {"triangles": len(mesh.triangles), "vertices": len(mesh.vertices), "textured": False}
    progress("mesh", 1)
    try:
        bake_texture(mesh, frames, T, path.parent, progress)
        info["textured"] = True
    except Exception as e:          # the vertex-coloured mesh is still a usable result
        print(f"texture bake failed: {e}")
    return info
