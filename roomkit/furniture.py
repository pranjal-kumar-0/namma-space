import cv2
import numpy as np

from .align import apply_T
from .config import FLOOR_STANDING, LABELS, MODELS_DIR, SEG_MODEL, Progress
from .fusion import backproject, voxel_down
from .frame import Frame


def detect_objects(frames: list[Frame], progress: Progress):
    from ultralytics import YOLOE

    model = YOLOE(str(MODELS_DIR / SEG_MODEL))
    names = model.names
    for i, f in enumerate(frames):
        r = model.predict(f.image, conf=0.2, max_det=300, verbose=False, retina_masks=True)[0]
        if r.masks is not None:
            for cls, conf, m in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(),
                                    r.masks.data.cpu().numpy()):
                label = LABELS.get(names[int(cls)].lower())
                if label is None:
                    continue
                m = m > 0.5
                if m.shape != f.depth.shape:
                    m = cv2.resize(m.astype(np.uint8), f.depth.shape[::-1],
                                   interpolation=cv2.INTER_NEAREST) > 0
                f.dets.append((label, float(conf), m))
        progress("objects", (i + 1) / len(frames))


def lift_detections(frames: list[Frame], T: np.ndarray) -> list[dict]:
    obs = []
    for fi, f in enumerate(frames):
        for label, conf, m in f.dets:
            if m.sum() < 400:
                continue
            er = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            if er.sum() < 200:
                er = m
            med = np.median(f.depth[er])
            keep = er & (np.abs(f.depth - med) < max(0.6, 0.35 * med))   # drop background leaks
            p, _ = backproject(f, keep, step=2)
            if len(p) < 50:
                continue
            p = apply_T(T, p)
            if len(p) > 3000:
                p = p[np.random.default_rng(fi).choice(len(p), 3000, replace=False)]
            obs.append({"label": label, "conf": conf, "frame": fi, "pts": p})
    return obs


def merge_observations(obs: list[dict]) -> list[dict]:
    objs = []
    for o in sorted(obs, key=lambda o: -o["conf"]):
        c = np.median(o["pts"], 0)
        ext = np.percentile(o["pts"], 90, 0) - np.percentile(o["pts"], 10, 0)
        best, bd = None, 1e9
        for ob in objs:
            if ob["label"] != o["label"]:
                continue
            dist = np.linalg.norm((ob["center"] - c)[[0, 2]])
            tol = max(0.45, 0.6 * max(ob["extent"][[0, 2]].max(), ext[[0, 2]].max()))
            if dist < tol and dist < bd:
                best, bd = ob, dist
        if best is None:
            objs.append({"label": o["label"], "pts": [o["pts"]], "confs": [o["conf"]],
                         "frames": {o["frame"]}, "center": c, "extent": ext})
        else:
            best["pts"].append(o["pts"])
            best["confs"].append(o["conf"])
            best["frames"].add(o["frame"])
            allp = np.concatenate(best["pts"])
            best["center"] = np.median(allp, 0)
            best["extent"] = np.percentile(allp, 90, 0) - np.percentile(allp, 10, 0)
    return objs


def fit_boxes(objs: list[dict], room_yaw: float, n_frames: int) -> list[dict]:
    boxes = []
    min_views = 2 if n_frames >= 10 else 1
    for ob in objs:
        if len(ob["frames"]) < min_views:
            continue
        P, _ = voxel_down(np.concatenate(ob["pts"]), None, 0.02)
        if len(P) < 60:
            continue
        med = np.median(P, 0)
        r = np.linalg.norm((P - med)[:, [0, 2]], axis=1)
        P = P[r < np.percentile(r, 92) * 1.15]
        xz = P[:, [0, 2]].astype(np.float32)

        yaw = np.deg2rad(cv2.minAreaRect(xz)[2])
        diff = (yaw - room_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
        if abs(diff) < np.deg2rad(15):               # snap to room axes if close
            yaw -= diff
        c, s = np.cos(yaw), np.sin(yaw)
        A = np.array([[c, -s], [s, c]])
        o = xz.mean(0)
        loc = (xz - o) @ A
        lo, hi = np.percentile(loc, 3, 0), np.percentile(loc, 97, 0)
        cx, cz = o + A @ ((lo + hi) / 2)
        w, d = hi - lo

        label = ob["label"]
        y_top = float(np.percentile(P[:, 1], 97))
        y_bot = 0.0 if label in FLOOR_STANDING else float(np.percentile(P[:, 1], 3))
        h = max(y_top - y_bot, 0.05)
        if label in ("door", "window"):
            if w >= d:
                d = 0.06
            else:
                w = 0.06
        boxes.append({
            "label": label,
            "center": [float(cx), float(y_bot + h / 2), float(cz)],
            "size": [float(max(w, 0.05)), float(h), float(max(d, 0.05))],
            "yaw": float(yaw),
            "confidence": float(np.mean(ob["confs"])),
            "views": len(ob["frames"]),
        })
    return boxes
