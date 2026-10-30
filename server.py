"""Web app: upload a room video, watch progress, view the structured 3D room.

Run:  .venv/bin/python server.py [--host 127.0.0.1] [--port 8000]

Uploads are chunked (Cloudflare tunnels reject request bodies over 100 MB) and resumable:
  POST /api/uploads                      {filename, size}  -> {id, received}
  PUT  /api/uploads/{id}?offset=N        raw chunk bytes   -> {received}
  POST /api/uploads/{id}/finish                            -> job
Jobs run one at a time in a worker thread. Job state lives in outputs/<id>/status.json,
so queued/running jobs are picked up again after a restart.
"""

import json
import queue
import re
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import pipeline

ROOT = Path(__file__).parent
OUTPUTS = ROOT / "outputs"
UPLOADS = ROOT / "assets"
PARTIAL = UPLOADS / ".partial"
for d in (OUTPUTS, UPLOADS, PARTIAL):
    d.mkdir(exist_ok=True)

MAX_UPLOAD = 2 * 1024**3          # 2 GB per video
MAX_CHUNK = 64 * 1024**2          # client sends 8 MB; anything above this is refused
DISK_RESERVE = 1 * 1024**3        # keep this much disk free after an upload lands
KEEP_SOURCE_VIDEO = False         # delete the uploaded video once its job succeeds

app = FastAPI()
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
work_q: "queue.Queue[str]" = queue.Queue()


# --------------------------------------------------------------------------- job state

def save_job(job: dict):
    d = OUTPUTS / job["id"]
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "status.json.tmp"
    tmp.write_text(json.dumps(job))
    tmp.replace(d / "status.json")


def public(job: dict) -> dict:
    j = {k: v for k, v in job.items() if k != "video"}
    if job["status"] == "queued":
        with jobs_lock:
            waiting = [x for x in jobs.values() if x["status"] == "queued"]
        waiting.sort(key=lambda x: x["created"])
        j["queue_position"] = next((i for i, x in enumerate(waiting) if x["id"] == job["id"]), 0) + 1
    return j


def load_jobs():
    for d in OUTPUTS.iterdir():
        st = d / "status.json"
        if st.exists():
            job = json.loads(st.read_text())
        elif (d / "scene.json").exists():   # results made by the CLI
            job = {"id": d.name, "name": d.name, "status": "done", "created": d.stat().st_mtime}
        else:
            continue
        if job["status"] in ("queued", "running"):
            if job.get("video") and Path(job["video"]).exists():
                job.update(status="queued", stage=None, frac=0)
                work_q.put(job["id"])
            else:
                job.update(status="error", error="The server restarted and the video is gone. Upload it again.")
            save_job(job)
        jobs[job["id"]] = job


def worker():
    while True:
        job_id = work_q.get()
        job = jobs[job_id]
        job.update(status="running", stage="frames", frac=0, started=time.time())
        save_job(job)
        last_save = [0.0]

        def progress(stage, frac):
            job["stage"], job["frac"] = stage, frac
            if time.time() - last_save[0] > 2:
                save_job(job)
                last_save[0] = time.time()

        try:
            scene = pipeline.process(Path(job["video"]), OUTPUTS / job_id, progress)
            job.update(status="done", stage="done", frac=1, walls=len(scene["walls"]),
                       objects=len(scene["objects"]))
            shutil.rmtree(OUTPUTS / job_id / "work", ignore_errors=True)
            if not KEEP_SOURCE_VIDEO:
                Path(job["video"]).unlink(missing_ok=True)
        except Exception as e:
            traceback.print_exc()
            job.update(status="error", error=str(e))
            shutil.rmtree(OUTPUTS / job_id / "work", ignore_errors=True)
        job["elapsed"] = time.time() - job["started"]
        save_job(job)


# --------------------------------------------------------------------------- uploads

class NewUpload(BaseModel):
    filename: str
    size: int


def upload_paths(upload_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
        raise HTTPException(404, "Unknown upload")
    return PARTIAL / f"{upload_id}.part", PARTIAL / f"{upload_id}.json"


@app.post("/api/uploads")
def start_upload(req: NewUpload):
    if req.size <= 0:
        raise HTTPException(400, "The file is empty.")
    if req.size > MAX_UPLOAD:
        raise HTTPException(413, f"Videos can be up to {MAX_UPLOAD // 1024**3} GB. "
                                 "Trim it or record at 1080p.")
    free = shutil.disk_usage(UPLOADS).free
    if free - req.size < DISK_RESERVE:
        raise HTTPException(507, "The server is low on disk space. Try a shorter or 1080p video.")
    upload_id = uuid.uuid4().hex
    part, meta = upload_paths(upload_id)
    part.touch()
    meta.write_text(json.dumps({"filename": req.filename, "size": req.size, "created": time.time()}))
    return {"id": upload_id, "received": 0}


@app.get("/api/uploads/{upload_id}")
def upload_status(upload_id: str):
    part, meta = upload_paths(upload_id)
    if not meta.exists():
        raise HTTPException(404, "Unknown upload")
    return {"id": upload_id, "received": part.stat().st_size, **json.loads(meta.read_text())}


@app.put("/api/uploads/{upload_id}")
async def put_chunk(upload_id: str, offset: int, request: Request):
    part, meta = upload_paths(upload_id)
    if not meta.exists():
        raise HTTPException(404, "Unknown upload")
    info = json.loads(meta.read_text())
    have = part.stat().st_size
    body = await request.body()
    if len(body) > MAX_CHUNK:
        raise HTTPException(413, "Chunk too large")
    if offset + len(body) <= have:          # retry of a chunk we already stored
        return {"received": have}
    if offset != have:
        raise HTTPException(409, detail={"received": have})
    if have + len(body) > info["size"]:
        raise HTTPException(400, "More data than the declared file size")
    with open(part, "ab") as fh:
        fh.write(body)
    return {"received": have + len(body)}


@app.post("/api/uploads/{upload_id}/finish")
def finish_upload(upload_id: str):
    part, meta = upload_paths(upload_id)
    if not meta.exists():
        raise HTTPException(404, "Unknown upload")
    info = json.loads(meta.read_text())
    if part.stat().st_size != info["size"]:
        raise HTTPException(409, detail={"received": part.stat().st_size})

    name = Path(info["filename"]).name
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(name).stem)[:40].strip("_") or "video"
    job_id = f"{stem}_{time.strftime('%Y%m%d-%H%M%S')}"
    while job_id in jobs:
        job_id += "x"
    ext = Path(name).suffix.lower() if re.fullmatch(r"\.[a-z0-9]{1,5}", Path(name).suffix.lower()) else ".mp4"
    video = UPLOADS / f"{job_id}{ext}"
    part.rename(video)
    meta.unlink()

    job = {"id": job_id, "name": Path(name).stem, "status": "queued", "stage": None, "frac": 0,
           "created": time.time(), "size": info["size"], "video": str(video)}
    with jobs_lock:
        jobs[job_id] = job
    save_job(job)
    work_q.put(job_id)
    return public(job)


def sweep_partials():
    """Drop abandoned partial uploads after a day."""
    while True:
        for meta in PARTIAL.glob("*.json"):
            try:
                if time.time() - json.loads(meta.read_text())["created"] > 86400:
                    meta.with_suffix(".part").unlink(missing_ok=True)
                    meta.unlink(missing_ok=True)
            except Exception:
                pass
        time.sleep(3600)


# --------------------------------------------------------------------------- jobs + results

@app.get("/api/jobs")
def list_jobs():
    with jobs_lock:
        js = sorted(jobs.values(), key=lambda j: j.get("created", 0), reverse=True)
    return [public(j) for j in js]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "No such scan")
    return public(jobs[job_id])


RESULT_FILES = {"scene.json": "application/json", "points.bin": "application/octet-stream",
                "points.ply": "application/octet-stream"}


@app.get("/results/{job_id}/{name}")
def result_file(job_id: str, name: str, download: int = 0):
    if name not in RESULT_FILES or job_id not in jobs:
        raise HTTPException(404)
    p = OUTPUTS / job_id / name
    if not p.exists():
        raise HTTPException(404)
    filename = f"{job_id}_{name}" if download else None
    return FileResponse(p, media_type=RESULT_FILES[name], filename=filename,
                        headers={"Cache-Control": "no-cache"})


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")


@app.on_event("startup")
def startup():
    load_jobs()
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=sweep_partials, daemon=True).start()


if __name__ == "__main__":
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port)
