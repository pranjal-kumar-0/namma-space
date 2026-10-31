"""Fuse the per-frame depth maps into a coloured triangle mesh (TSDF), so the room reads as
continuous surfaces instead of loose points."""

from pathlib import Path

import numpy as np

from .config import MESH_MAX_DEPTH, MESH_MAX_TRIS, MESH_RES, Progress
from .frame import Frame
from .fusion import flying_pixel_mask


def build_mesh(frames: list[Frame], T: np.ndarray, P: np.ndarray, ceiling: float, path: Path,
               progress: Progress) -> dict:
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

    if len(mesh.triangles) > MESH_MAX_TRIS:
        mesh = mesh.simplify_quadric_decimation(MESH_MAX_TRIS)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if not len(mesh.triangles):
        raise RuntimeError("depth fusion produced an empty mesh")

    o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False, write_vertex_normals=False)
    progress("mesh", 1)
    return {"triangles": len(mesh.triangles), "vertices": len(mesh.vertices)}
