from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"
SEG_MODEL = "yoloe-11s-seg-pf.pt"

MAX_FRAMES = 90
SHARPNESS_POOL = 3
FRAME_LONG_SIDE = 900
FUSE_VOXEL = 0.025
LIDAR_MAX_FRAMES = 120     # keyframes kept from a LiDAR scan (no SfM cost, so more than video)

MESH_RES = 320             # TSDF cells per side of the room-sized cube (~2-3 cm for a normal room)
MESH_MAX_DEPTH = 6.0
MESH_MAX_TRIS = 250_000
ATLAS_SIZE = 4096          # texture atlas side in pixels
TEX_TRIS = 100_000         # the textured mesh needs fewer triangles: the photo carries the detail
SNAP_TOL = 0.15            # metres: mesh within this of a floor/wall/ceiling plane is flattened onto it
GLB_SNAP_TOL = 0.15        # same, for imported .glb meshes (raise it if their walls/ceiling stay lumpy)

Progress = Callable[[str, float], None]

# YOLOE class name -> canonical label; synonyms collapse so one object = one box.
LABELS = {
    **dict.fromkeys(["bed", "bunk bed", "canopy bed", "daybed", "day bed", "bed frame"], "bed"),
    **dict.fromkeys(["couch"], "sofa"),
    **dict.fromkeys(["chair", "armchair", "office chair", "computer chair", "swivel chair",
                     "folding chair", "rocking chair", "bean bag chair", "stool"], "chair"),
    **dict.fromkeys(["table", "side table", "kitchen table", "dinning table", "cocktail table",
                     "glass table"], "table"),
    **dict.fromkeys(["computer desk", "office desk", "writing desk"], "desk"),
    **dict.fromkeys(["cabinet", "kitchen cabinet", "bathroom cabinet", "tv cabinet", "closet",
                     "dresser", "nightstand", "drawer"], "cabinet"),
    **dict.fromkeys(["bookshelf", "bookcase", "shelf"], "bookshelf"),
    **dict.fromkeys(["television", "monitor", "computer monitor", "crt screen", "screen",
                     "desktop computer"], "tv"),
    **dict.fromkeys(["fridge"], "refrigerator"),
    **dict.fromkeys(["toilet bowl", "toilet seat"], "toilet"),
    **dict.fromkeys(["sink", "bathroom sink"], "sink"),
    **dict.fromkeys(["tub", "bath"], "bathtub"),
    **dict.fromkeys(["lamp", "table lamp", "bedside lamp"], "lamp"),
    **dict.fromkeys(["houseplant", "plant"], "plant"),
    **dict.fromkeys(["door", "bathroom door", "glass door", "barn door"], "door"),
    **dict.fromkeys(["window", "bedroom window", "bathroom window", "kitchen window",
                     "glass window", "office window", "bay window"], "window"),
    **dict.fromkeys(["counter"], "counter"),
    **dict.fromkeys(["washing machine"], "washing machine"),
    **dict.fromkeys(["air conditioner"], "air conditioner"),
    **dict.fromkeys(["bench"], "bench"),
    **dict.fromkeys(["oven"], "oven"),
    **dict.fromkeys(["bottle", "water bottle", "insulated cup", "thermos", "vacuum flask",
                     "flask", "jug", "bubble tea"], "bottle"),
    **dict.fromkeys(["laptop", "notebook computer", "laptop keyboard"], "laptop"),
    **dict.fromkeys(["keyboard", "computer keyboard"], "keyboard"),
}

# box pinned to y=0
FLOOR_STANDING = {"bed", "sofa", "chair", "table", "desk", "cabinet", "bookshelf", "refrigerator",
                  "toilet", "bathtub", "door", "counter", "washing machine", "bench", "oven"}
