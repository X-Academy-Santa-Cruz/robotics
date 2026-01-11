import os
import time
import cv2
import numpy as np
import subprocess
from pathlib import Path
import pycolmap
import argparse
import sys
# =========================
# USER CONFIG
# =========================

CAM_IDS = [0, 1, 2]       # indices of your 3 webcams
PROJECT_NAME = "auto_3cam_project"

# Capture parameters
CAPTURE_INTERVAL_SEC = 1.0    # time between snapshots (you move robot independently)
MAX_SHOTS = 25               # total snapshots to take (each = 3 images)

# Paths
BASE_DIR = Path(PROJECT_NAME)
IMG_DIR = BASE_DIR / "images"
DB_PATH = BASE_DIR / "database.db"
SPARSE_DIR = BASE_DIR / "sparse"
SPARSE_TXT_DIR = SPARSE_DIR / "0_text"
DENSE_DIR = BASE_DIR / "dense"
MODEL_OBJ = BASE_DIR / "model.obj"
MODEL_OBJ_SCALED = BASE_DIR / "model_scaled.obj"

# Known real-world distance (meters) between two points on the object
KNOWN_DISTANCE_METERS = 0.10   # e.g. 10 cm

# 1-based indices of two vertices in the OBJ that correspond to that distance
VERTEX_INDEX_1 = 100
VERTEX_INDEX_2 = 200


# =========================
# UTILS
# =========================

def run_cmd(cmd_list):
    print("RUN:", " ".join(map(str, cmd_list)))
    subprocess.check_call(cmd_list)


def ensure_dirs():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    SPARSE_DIR.mkdir(parents=True, exist_ok=True)
    DENSE_DIR.mkdir(parents=True, exist_ok=True)
    # Ensure TXT sparse output path exists before model_converter
    SPARSE_TXT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# STEP 1: AUTONOMOUS CAPTURE FROM 3 CAMERAS
# =========================

def autonomous_capture():
    """
    Open all 3 cameras and automatically capture synchronized frames
    every CAPTURE_INTERVAL_SEC seconds, up to MAX_SHOTS snapshots.
    Robot movement is assumed to be handled separately.
    """
    print("=== Step 1: Autonomous capture from 3 cameras ===")
    caps = [cv2.VideoCapture(i) for i in CAM_IDS]

    # Set resolution if supported
    for cap in caps:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    shot_idx = 0
    last_time = time.time()

    while shot_idx < MAX_SHOTS:
        now = time.time()
        if now - last_time < CAPTURE_INTERVAL_SEC:
            # Small sleep to avoid busy-waiting
            time.sleep(0.01)
            continue

        last_time = now
        frames = []
        for cam_id, cap in zip(CAM_IDS, caps):
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read from camera {cam_id}")
            frames.append((cam_id, frame))

        # Save frames
        print(f"Snapshot {shot_idx}")
        for cam_id, frame in frames:
            fname = IMG_DIR / f"shot_{shot_idx:04d}_cam{cam_id}.png"
            cv2.imwrite(str(fname), frame)
            print("Saved", fname)

        shot_idx += 1

    for cap in caps:
        cap.release()

    print("Autonomous capture complete. Total snapshots:", shot_idx)
    if shot_idx < 10:
        print("Warning: few snapshots; reconstruction quality may be poor.")


# =========================
# STEP 2: COLMAP PIPELINE
# =========================

def run_colmap(skip_dense: bool = False):
    """
    Run COLMAP SfM + basic MVS to reconstruct a 3D model.
    """
    print("=== Step 2: Running COLMAP pipeline ===")

    # 1) Feature extraction
    run_cmd([
        "colmap", "feature_extractor",
        "--database_path", str(DB_PATH),
        "--image_path", str(IMG_DIR),
        "--ImageReader.camera_model=PINHOLE"
    ])

    # 2) Matching
    run_cmd([
        "colmap", "exhaustive_matcher",
        "--database_path", str(DB_PATH)
    ])

    # 3) Sparse reconstruction
    run_cmd([
        "colmap", "mapper",
        "--database_path", str(DB_PATH),
        "--image_path", str(IMG_DIR),
        "--output_path", str(SPARSE_DIR)
    ])

    # 4) Convert to TXT for inspection (optional)
    model_bin_dir = SPARSE_DIR / "0"
    run_cmd([
        "colmap", "model_converter",
        "--input_path", str(model_bin_dir),
        "--output_path", str(SPARSE_TXT_DIR),
        "--output_type", "TXT"
    ])
    if skip_dense:
        print("Skipping dense reconstruction steps (image undistort / patch-match / fusion / meshing).")
        return

    # 5) Undistort images for dense reconstruction
    run_cmd([
        "colmap", "image_undistorter",
        "--image_path", str(IMG_DIR),
        "--input_path", str(model_bin_dir),
        "--output_path", str(DENSE_DIR),
        "--max_image_size", "2000"
    ])

    # 6) PatchMatch stereo
    run_cmd([
        "colmap", "patch_match_stereo",
        "--workspace_path", str(DENSE_DIR),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true"
    ])

    # 7) Fuse to dense point cloud
    run_cmd([
        "colmap", "stereo_fusion",
        "--workspace_path", str(DENSE_DIR),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(DENSE_DIR / "fused.ply")
    ])

    # 8) Mesh & export as OBJ (if Poisson mesher is available)
    try:
        run_cmd([
            "colmap", "poisson_mesher",
            "--input_path", str(DENSE_DIR / "fused.ply"),
            "--output_path", str(MODEL_OBJ)
        ])
    except subprocess.CalledProcessError:
        print("Poisson mesher failed or is unavailable.")
        print("You may need to mesh 'fused.ply' elsewhere and set MODEL_OBJ manually.")


# =========================
# STEP 3: SCALE OBJ WITH ONE KNOWN DISTANCE
# =========================

def load_vertices_from_obj(path: Path):
    verts = []
    with path.open("r") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.strip().split()[:4]
                verts.append([float(x), float(y), float(z)])
    return np.array(verts, dtype=np.float64)


def write_scaled_obj(input_path: Path, output_path: Path, scale: float):
    with input_path.open("r") as fin, output_path.open("w") as fout:
        for line in fin:
            if line.startswith("v "):
                _, x, y, z = line.strip().split()[:4]
                x = float(x) * scale
                y = float(y) * scale
                z = float(z) * scale
                fout.write(f"v {x} {y} {z}\n")
            else:
                fout.write(line)


def compute_scale_and_apply():
    print("=== Step 3: Scaling OBJ with known distance ===")
    if not MODEL_OBJ.exists():
        raise FileNotFoundError(
            f"OBJ model not found at {MODEL_OBJ}. Mesh it first or update MODEL_OBJ."
        )

    verts = load_vertices_from_obj(MODEL_OBJ)
    if VERTEX_INDEX_1 <= 0 or VERTEX_INDEX_2 <= 0:
        raise ValueError("Vertex indices must be positive 1-based indices.")

    if VERTEX_INDEX_1 > len(verts) or VERTEX_INDEX_2 > len(verts):
        raise IndexError("Vertex indices out of range for OBJ vertices.")

    P1 = verts[VERTEX_INDEX_1 - 1]
    P2 = verts[VERTEX_INDEX_2 - 1]

    d_model = np.linalg.norm(P1 - P2)
    if d_model <= 0:
        raise ValueError("Model distance between chosen vertices is zero/invalid.")

    scale = KNOWN_DISTANCE_METERS / d_model
    print(f"Model distance: {d_model:.6f}")
    print(f"Real distance: {KNOWN_DISTANCE_METERS:.6f} m")
    print(f"Scale factor: {scale:.6f}")

    write_scaled_obj(MODEL_OBJ, MODEL_OBJ_SCALED, scale)
    print("Scaled model written to:", MODEL_OBJ_SCALED)


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser(description="Simple 3-camera photogrammetry runner using COLMAP")
    default_skip = sys.platform == "darwin"
    parser.add_argument("--skip-dense", action="store_true", default=default_skip,
                        help=f"Skip dense reconstruction steps (default: {default_skip}). Use on macOS or when no CUDA is available.")
    parser.add_argument("--no-capture", action="store_true", default=False,
                        help="Skip the camera capture step and use existing images in the project images folder.")
    args = parser.parse_args()

    ensure_dirs()
    if not args.no_capture:
        autonomous_capture()       # Step 1: fully automatic snapshots while robot moves

    run_colmap(skip_dense=args.skip_dense)               # Step 2: reconstruction

    # Scaling requires an OBJ present; only run if available
    if MODEL_OBJ.exists():
        compute_scale_and_apply()  # Step 3: scale model
    else:
        print(f"Skipping scale step because OBJ not found at {MODEL_OBJ}.")

    print("All done.")


if __name__ == "__main__":
    main()
