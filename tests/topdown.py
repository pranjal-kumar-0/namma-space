"""Top-down debug plot of a result: points by height, walls, object boxes."""
import json, struct, sys
import numpy as np, cv2
d = sys.argv[1]
sc = json.load(open(f"{d}/scene.json"))
buf = open(f"{d}/points.bin", "rb").read()
n = struct.unpack("<I", buf[:4])[0]
P = np.frombuffer(buf, np.float32, n * 3, 4).reshape(-1, 3)
print("ceiling", round(sc["ceiling_height"], 2), "floor_extent", np.round(sc["floor_extent"]["min"], 2), np.round(sc["floor_extent"]["max"], 2))
print("y percentiles 1/50/99:", np.round(np.percentile(P[:, 1], [1, 50, 99]), 2))
lo, hi = np.percentile(P[:, [0, 2]], 0.5, 0) - 0.5, np.percentile(P[:, [0, 2]], 99.5, 0) + 0.5
S = 100; img = np.full((int((hi[1]-lo[1])*S), int((hi[0]-lo[0])*S), 3), 255, np.uint8)
ij = ((P[:, [0, 2]] - lo) * S).astype(int); ok = (ij >= 0).all(1) & (ij[:, 0] < img.shape[1]) & (ij[:, 1] < img.shape[0])
c = np.clip(P[ok, 1] / 2.6, 0, 1)
img[ij[ok, 1], ij[ok, 0]] = np.stack([255 * (1 - c), 160 * np.ones_like(c), 255 * c], 1).astype(np.uint8)
for w in sc["walls"]:
    p1 = (int((w["x1"]-lo[0])*S), int((w["z1"]-lo[1])*S)); p2 = (int((w["x2"]-lo[0])*S), int((w["z2"]-lo[1])*S))
    cv2.line(img, p1, p2, (0, 0, 0), 3)
    print("wall len %.2f" % np.hypot(w["x2"]-w["x1"], w["z2"]-w["z1"]))
for o in sc["objects"]:
    (cx, cy, cz), (sx, sy, sz), yaw = o["center"], o["size"], o["yaw"]
    box = cv2.boxPoints(((cx, cz), (sx, sz), np.rad2deg(yaw)))
    cv2.polylines(img, [((box - lo) * S).astype(np.int32)], True, (0, 0, 255), 2)
    print(o["label"], np.round(o["center"], 2), np.round(o["size"], 2), "views", o["views"])
cv2.imwrite(sys.argv[2], img)
