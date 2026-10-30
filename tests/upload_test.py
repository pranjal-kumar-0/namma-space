"""End-to-end check of the chunked upload API, job processing and result downloads.

usage: python tests/upload_test.py BASE_URL VIDEO [--resume-test]
Mirrors what the browser does: 8 MB chunks, retry of a duplicate chunk, resume after a gap.
"""
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request

base, video = sys.argv[1].rstrip("/"), sys.argv[2]
CHUNK = 8 * 1024 * 1024
UA = {"User-Agent": "room3d-upload-test"}


def call(method, path, data=None, ctype=None):
    h = dict(UA)
    if ctype:
        h["Content-Type"] = ctype
    req = urllib.request.Request(base + path, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


data = open(video, "rb").read()
size = len(data)
print(f"file {size / 1e6:.1f} MB")

# rejects: empty, oversize
assert call("POST", "/api/uploads", json.dumps({"filename": "x.mp4", "size": 0}).encode(), "application/json")[0] == 400
assert call("POST", "/api/uploads", json.dumps({"filename": "x.mp4", "size": 5 * 1024**3}).encode(), "application/json")[0] == 413

st, body = call("POST", "/api/uploads", json.dumps({"filename": video.split("/")[-1], "size": size}).encode(), "application/json")
assert st == 200, (st, body)
uid = json.loads(body)["id"]

t0 = time.time()
off = 0
n = 0
while off < size:
    chunk = data[off:off + CHUNK]
    st, body = call("PUT", f"/api/uploads/{uid}?offset={off}", chunk, "application/octet-stream")
    assert st == 200, (st, body[:200])
    new = json.loads(body)["received"]
    if n == 1:
        # duplicate retry of the same chunk must be a no-op
        st2, b2 = call("PUT", f"/api/uploads/{uid}?offset={off}", chunk, "application/octet-stream")
        assert st2 == 200 and json.loads(b2)["received"] == new, (st2, b2)
        # out-of-order chunk must be refused with the server's offset
        st3, b3 = call("PUT", f"/api/uploads/{uid}?offset={new + CHUNK}", chunk, "application/octet-stream")
        assert st3 == 409 and json.loads(b3)["detail"]["received"] == new, (st3, b3)
        # resume: server reports what it has
        st4, b4 = call("GET", f"/api/uploads/{uid}")
        assert json.loads(b4)["received"] == new
    off = new
    n += 1
    print(f"\r  {off / size * 100:5.1f}%  {off / 1e6 / (time.time() - t0):.1f} MB/s", end="", flush=True)
print(f"\nuploaded in {time.time() - t0:.1f}s, {n} chunks")

st, body = call("POST", f"/api/uploads/{uid}/finish")
assert st == 200, (st, body)
job = json.loads(body)
print("job", job["id"], job["status"])

t0 = time.time()
while True:
    time.sleep(3)
    j = json.loads(call("GET", f"/api/jobs/{job['id']}")[1])
    print(f"\r  {j['status']:8s} {j.get('stage') or '':8s} {100 * (j.get('frac') or 0):5.1f}%  {time.time() - t0:5.0f}s", end="", flush=True)
    if j["status"] in ("done", "error"):
        break
print()
assert j["status"] == "done", j

for name in ("scene.json", "points.bin", "points.ply"):
    st, body = call("GET", f"/results/{job['id']}/{name}?download=1")
    assert st == 200 and len(body) > 100, (name, st)
    print(f"  download {name}: {len(body) / 1e6:.2f} MB ok")
scene = json.loads(call("GET", f"/results/{job['id']}/scene.json")[1])
print(f"walls={len(scene['walls'])} objects={len(scene['objects'])} posed={scene['stats']['posed']}/{scene['stats']['frames']}")
assert call("GET", "/results/../server.py/scene.json")[0] == 404
print("ALL CHECKS PASSED")
