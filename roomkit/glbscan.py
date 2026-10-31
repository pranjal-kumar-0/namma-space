"""Import a finished textured room mesh (a Polycam .glb export) as a scan.

The mesh itself is shown in the viewer as-is. Points are sampled from its surface so the walls and
floor can be found, and the mesh is shifted so the floor sits at y=0."""

import json
import shutil
import struct
import time
from pathlib import Path

import cv2
import numpy as np

from .config import Progress
from .export import write_outputs
from .walls import ceiling_height, detect_walls

N_POINTS = 400_000
DTYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _read_glb(path: Path) -> tuple[dict, bytes]:
    b = path.read_bytes()
    if b[:4] != b"glTF":
        raise RuntimeError("not a binary glTF (.glb) file")
    j, bin_ = None, b""
    pos = 12
    while pos < len(b):
        n, kind = struct.unpack("<II", b[pos:pos + 8])
        chunk = b[pos + 8:pos + 8 + n]
        if kind == 0x4E4F534A:
            j = json.loads(chunk)
        elif kind == 0x004E4942:
            bin_ = chunk
        pos += 8 + n
    if j is None:
        raise RuntimeError("glb has no JSON chunk")
    if "KHR_draco_mesh_compression" in j.get("extensionsUsed", []):
        raise RuntimeError("Draco-compressed glb is not supported; export without compression")
    return j, bin_


def _accessor(j: dict, bin_: bytes, idx: int) -> np.ndarray:
    a = j["accessors"][idx]
    bv = j["bufferViews"][a["bufferView"]]
    dt = np.dtype(DTYPES[a["componentType"]])
    nc = NCOMP[a["type"]]
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = bv.get("byteStride") or dt.itemsize * nc
    if stride == dt.itemsize * nc:
        out = np.frombuffer(bin_, dt, a["count"] * nc, off)
    else:
        raw = np.frombuffer(bin_, np.uint8, stride * (a["count"] - 1) + dt.itemsize * nc, off)
        out = np.lib.stride_tricks.as_strided(raw, (a["count"], dt.itemsize * nc), (stride, 1)).copy().view(dt)
    return out.reshape(a["count"], nc) if nc > 1 else out


def _node_matrix(n: dict) -> np.ndarray:
    if "matrix" in n:
        return np.array(n["matrix"], float).reshape(4, 4).T
    M = np.eye(4)
    if "scale" in n:
        M = np.diag([*n["scale"], 1.0]) @ M
    if "rotation" in n:
        x, y, z, w = n["rotation"]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        M = np.block([[R, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]) @ M
    if "translation" in n:
        T = np.eye(4)
        T[:3, 3] = n["translation"]
        M = T @ M
    return M


def _primitives(j: dict, bin_: bytes):
    """Yield (positions in world, triangles, uvs or None, material index) for every triangle primitive."""
    def walk(i, parent):
        n = j["nodes"][i]
        M = parent @ _node_matrix(n)
        if "mesh" in n:
            for p in j["meshes"][n["mesh"]]["primitives"]:
                if p.get("mode", 4) != 4:
                    continue
                V = _accessor(j, bin_, p["attributes"]["POSITION"]).astype(np.float64)
                V = V @ M[:3, :3].T + M[:3, 3]
                I = (_accessor(j, bin_, p["indices"]).astype(np.int64) if "indices" in p
                     else np.arange(len(V))).reshape(-1, 3)
                uv = (_accessor(j, bin_, p["attributes"]["TEXCOORD_0"]).astype(np.float64)
                      if "TEXCOORD_0" in p["attributes"] else None)
                yield V, I, uv, p.get("material")
        for c in n.get("children", []):
            yield from walk(c, M)

    scene = j["scenes"][j.get("scene", 0)]
    for r in scene["nodes"]:
        yield from walk(r, np.eye(4))


def _texture(j: dict, bin_: bytes, mat: int | None):
    if mat is None:
        return None, np.array([200, 200, 200])
    m = j["materials"][mat].get("pbrMetallicRoughness", {})
    factor = np.array(m.get("baseColorFactor", [0.8, 0.8, 0.8, 1])[:3]) * 255
    t = m.get("baseColorTexture")
    if t is None:
        return None, factor[::-1]
    src = j["textures"][t["index"]]["source"]
    bv = j["bufferViews"][j["images"][src]["bufferView"]]
    img = cv2.imdecode(np.frombuffer(bin_, np.uint8, bv["byteLength"], bv.get("byteOffset", 0)), cv2.IMREAD_COLOR)
    return img, factor[::-1]


def _floor(y: np.ndarray) -> float:
    """Lowest dense horizontal slab."""
    cand = y[y >= np.percentile(y, 0.5)]
    hist, edges = np.histogram(cand, bins=np.arange(cand.min(), cand.max() + 0.04, 0.02))
    i = next(i for i in range(len(hist)) if hist[i] >= hist.max() * 0.3)
    return float((edges[i] + edges[i + 1]) / 2)


def import_glb(glb: Path, out: Path, progress: Progress | None = None, name: str | None = None) -> dict:
    progress = progress or (lambda stage, frac: None)
    glb, out = Path(glb), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    progress("fusion", 0)
    j, bin_ = _read_glb(glb)
    prims = list(_primitives(j, bin_))
    if not prims:
        raise RuntimeError("the glb has no triangle meshes")

    areas = []
    for V, I, _, _ in prims:
        areas.append(0.5 * np.linalg.norm(np.cross(V[I[:, 1]] - V[I[:, 0]], V[I[:, 2]] - V[I[:, 0]]), axis=1))
    total = sum(a.sum() for a in areas)
    rng = np.random.default_rng(0)
    pts, cols = [], []
    for (V, I, uv, mat), a in zip(prims, areas):
        n = int(N_POINTS * a.sum() / total)
        if n < 1:
            continue
        tri = rng.choice(len(I), n, p=a / a.sum())
        b = rng.random((n, 2))
        flip = b.sum(1) > 1
        b[flip] = 1 - b[flip]
        w = np.stack([1 - b.sum(1), b[:, 0], b[:, 1]], 1)
        pts.append((V[I[tri]] * w[..., None]).sum(1))
        img, fallback = _texture(j, bin_, mat)
        if img is not None and uv is not None:
            q = (uv[I[tri]] * w[..., None]).sum(1) % 1.0
            h, wd = img.shape[:2]
            cols.append(img[np.clip((q[:, 1] * h).astype(int), 0, h - 1), np.clip((q[:, 0] * wd).astype(int), 0, wd - 1)])
        else:
            cols.append(np.tile(fallback, (n, 1)).astype(np.uint8))
    P, C = np.concatenate(pts), np.concatenate(cols)
    progress("fusion", 1)

    floor_y = _floor(P[:, 1])                      # glTF is Y-up; put the floor at y=0
    P[:, 1] -= floor_y
    progress("walls", 0)
    keep = (P[:, 1] > -0.3) & (P[:, 1] < 6)
    ceil = ceiling_height(P[keep])
    walls, yaw = detect_walls(P[keep], ceil)
    progress("walls", 1)

    shutil.copyfile(glb, out / "model.glb")
    scene = write_outputs(out, P.astype(np.float32), C, walls, ceil, [], [], yaw)
    scene["model"] = {"file": "model.glb", "offset_y": -floor_y}
    scene["mesh"] = None
    scene["stats"] = {"source": "glb", "points": int(len(P))}
    (out / "scene.json").write_text(json.dumps(scene, indent=1))
    (out / "status.json").write_text(json.dumps({
        "id": out.name, "name": name or glb.stem, "status": "done", "stage": "done", "frac": 1,
        "created": time.time(), "walls": len(walls), "objects": 0}))
    progress("done", 1)
    return scene
