#!/usr/bin/env python3
"""
Create a coarse mesh from COLMAP sparse output (`sparse/0_text/points3D.txt`).

This is a fallback for machines where COLMAP dense tools are unavailable
(e.g. macOS without CUDA). The produced mesh is low-quality compared to
an MVS-derived mesh but can serve as a quick visualisation.

Usage:
  python make_mesh_from_sparse.py --project auto_3cam_project

Outputs:
  - `auto_3cam_project/model_from_sparse.obj`
  - `auto_3cam_project/model_from_sparse.ply`
"""
import argparse
from pathlib import Path
import numpy as np

def parse_points3d_txt(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"points3D.txt not found at {p}")

    pts = []
    cols = []
    with p.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # Format: POINT3D_ID X Y Z R G B ERROR TRACK_LENGTH TRACK...
            if len(parts) < 7:
                continue
            try:
                x = float(parts[1]); y = float(parts[2]); z = float(parts[3])
                r = int(parts[4]); g = int(parts[5]); b = int(parts[6])
            except Exception:
                continue
            pts.append([x, y, z])
            cols.append([r / 255.0, g / 255.0, b / 255.0])

    return np.asarray(pts, dtype=np.float64), np.asarray(cols, dtype=np.float64)


def make_mesh_from_points(points: np.ndarray, colors: np.ndarray, out_obj: Path, out_ply: Path, depth: int = 9):
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError("open3d is required for meshing. Install via `pip install open3d`.") from e

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"Point cloud has {len(points)} points")

    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)

    print("Running Poisson surface reconstruction (this may take a while)...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    densities = np.asarray(densities)
    # Remove low-density vertices (keep the top ~95%)
    thresh = np.quantile(densities, 0.05)
    verts_to_remove = densities < thresh
    mesh.remove_vertices_by_mask(verts_to_remove)

    mesh.compute_vertex_normals()

    print(f"Writing mesh to: {out_obj} and {out_ply}")
    out_obj.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_ply), mesh)
    o3d.io.write_triangle_mesh(str(out_obj), mesh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="auto_3cam_project", help="Project folder (default: auto_3cam_project)")
    parser.add_argument("--depth", type=int, default=9, help="Poisson depth (higher = more detail, more memory)")
    args = parser.parse_args()

    base = Path(args.project)
    points_txt = base / "sparse" / "0_text" / "points3D.txt"
    out_obj = base / "model_from_sparse.obj"
    out_ply = base / "model_from_sparse.ply"

    pts, cols = parse_points3d_txt(points_txt)
    if pts.size == 0:
        raise RuntimeError("No points found in points3D.txt — ensure sparse reconstruction completed.")

    make_mesh_from_points(pts, cols, out_obj, out_ply, depth=args.depth)
    print("Done.")


if __name__ == '__main__':
    main()
