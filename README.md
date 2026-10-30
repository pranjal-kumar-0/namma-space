# Room3D

Turn a single handheld phone video of a room into a measurable 3D model: a colour point cloud, the
floor and ceiling, the walls as a clean floor plan, and furniture as labelled boxes with real sizes.
Everything runs on **CPU only** and the result opens in a browser viewer you can **walk through** in
first person.

The method and the capture SOP are written up in [`Room3D_Technical_Abstract.pdf`](Room3D_Technical_Abstract.pdf).

---

## 1. Requirements

- Python 3.10+
- `ffmpeg` and `ffprobe` on your `PATH` (system packages)
- ~4 GB RAM free; no GPU needed

On first run the depth and segmentation models (Depth-Anything-V2 metric-indoor, YOLOE) are downloaded
automatically into `models/`.

## 2. Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 3. Reconstruct a room

Sample clips live in the test directory (`tests/data/`). Run the pipeline on one of them:

```bash
.venv/bin/python pipeline.py tests/data/room_bedroom.mp4
```

The result is written to `outputs/room_bedroom/`. Point it at your own clip the same way:

```bash
.venv/bin/python pipeline.py path/to/your_video.mp4 -o outputs/my_room
```

While it runs it prints each stage (`frames → poses → depth → fusion → walls → objects → done`).
A typical 30–60 s clip takes a few minutes on a multi-core CPU.

## 4. View and roam

```bash
.venv/bin/python server.py        # then open http://localhost:8000
```

Open a scan from the list. Toggle layers (points, walls, floor, ceiling, furniture, labels, camera
path), and press **Walk** to drop into first-person free-roam:

- `W A S D` / arrow keys to move, **drag the mouse** to look around
- `Space` / `Ctrl` to change height, `Esc` to exit
- walls block you and keep you inside the room

## 5. Outputs

Each run writes to `outputs/<name>/`:

| File | What it is |
|------|------------|
| `scene.json` | walls, furniture boxes, camera path, floor extent, ceiling height, in metres (Y-up, floor at `y = 0`) |
| `points.ply` | colour point cloud, opens in MeshLab / CloudCompare |
| `points.bin` | the same cloud packed for the web viewer |

## 6. Capture SOP (summary)

Reconstruction quality is set at capture time. Film **one slow loop** around the room's edge with the
camera pointed inward and tilted slightly down so some floor is always visible; return to the start to
close the loop. Move steadily (~0.3–0.5 m/s), keep lighting even, and avoid fast pans, blank
textureless walls, mirrors, and glass. 20–60 seconds is enough. Full details are in the technical
abstract.

## 7. Layout

```
pipeline.py        CLI entry point (reconstruct a video)
server.py          local web app (upload + viewer)
roomkit/           pipeline stages: frames, sfm, depth, fusion, align, walls, furniture, export
web/               the Three.js viewer
tests/data/        sample videos
docs/              technical abstract (PDF + HTML)
outputs/           reconstructions (created on run)
```
