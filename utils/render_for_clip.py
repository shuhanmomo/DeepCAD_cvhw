import os
import json
import numpy as np
import open3d as o3d
import argparse
import sys
import tempfile
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.Poly import Poly_Triangulation

# Fix import path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from cadlib.extrude import CADSequence
from cadlib.visualize import create_CAD


def convert_shape_to_mesh(shape: TopoDS_Shape) -> o3d.geometry.TriangleMesh:
    """Convert OpenCascade shape directly to Open3D mesh"""
    # Create the mesh
    mesh_maker = BRepMesh_IncrementalMesh(shape, 0.1, False, 0.1, True)
    mesh_maker.Perform()

    # Initialize lists to store vertices and triangles
    vertices = []
    triangles = []
    vertex_map = {}  # Map to track unique vertices
    current_vertex_index = 0

    # Explore each face
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = explorer.Current()
        location = face.Location()

        # Get the triangulation of the face
        face_tool = BRep_Tool()
        triangulation = face_tool.Triangulation(face, location)

        if triangulation is not None:
            # Get nodes (vertices)
            for i in range(1, triangulation.NbNodes() + 1):
                point = triangulation.Node(i)
                vertex = (point.X(), point.Y(), point.Z())

                # Check if vertex already exists
                vertex_key = f"{vertex[0]:.6f},{vertex[1]:.6f},{vertex[2]:.6f}"
                if vertex_key not in vertex_map:
                    vertex_map[vertex_key] = current_vertex_index
                    vertices.append(vertex)
                    current_vertex_index += 1

            # Get triangles
            for i in range(1, triangulation.NbTriangles() + 1):
                triangle = triangulation.Triangle(i)
                # Note: OpenCascade uses 1-based indexing
                idx1, idx2, idx3 = triangle.Get()
                # Convert to 0-based indexing for vertices
                v1_key = f"{triangulation.Node(idx1).X():.6f},{triangulation.Node(idx1).Y():.6f},{triangulation.Node(idx1).Z():.6f}"
                v2_key = f"{triangulation.Node(idx2).X():.6f},{triangulation.Node(idx2).Y():.6f},{triangulation.Node(idx2).Z():.6f}"
                v3_key = f"{triangulation.Node(idx3).X():.6f},{triangulation.Node(idx3).Y():.6f},{triangulation.Node(idx3).Z():.6f}"

                triangles.append(
                    [vertex_map[v1_key], vertex_map[v2_key], vertex_map[v3_key]]
                )

        explorer.Next()

    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))

    return mesh


def setup_camera_views(n_views=8):
    """Generate camera parameters for different views.
    For each elevation angle, we do a complete spin around the model."""
    views = []

    # Define different elevation angles (in radians)
    elevations = [0.2, 0.5, 0.8]  # Low, medium, and high angles

    # For each elevation, we'll do a complete spin
    n_angles_per_spin = n_views // len(elevations)  # How many views per elevation

    for elevation in elevations:
        # Do a complete spin at this elevation
        for i in range(n_angles_per_spin):
            # Calculate azimuth angle for this position in the spin
            azimuth = (i * 2 * np.pi) / n_angles_per_spin

            # Convert spherical coordinates to Cartesian
            cam_x = np.cos(azimuth) * np.cos(elevation)
            cam_y = np.sin(azimuth) * np.cos(elevation)
            cam_z = np.sin(elevation)

            # Camera position
            position = [cam_x, cam_y, cam_z]

            # Look at center
            front = [-cam_x, -cam_y, -cam_z]

            # Calculate up vector (handle the pole case)
            if abs(cam_z) > 0.999:
                up = [0, 1, 0]  # Use a fixed up vector when looking straight down/up
            else:
                # Cross product with world up to get right vector, then cross again to get up
                right = np.cross([0, 0, 1], position)
                right = right / np.linalg.norm(right)
                up = np.cross(front, right)
                up = up / np.linalg.norm(up)

            views.append({"position": position, "front": front, "up": up})

    # If we need more views to reach n_views (due to rounding),
    # add some views at intermediate elevations
    while len(views) < n_views:
        # Random spherical coordinates
        azimuth = np.random.uniform(0, 2 * np.pi)
        elevation = np.random.uniform(0.3, 0.7)  # Mid-range elevations

        cam_x = np.cos(azimuth) * np.cos(elevation)
        cam_y = np.sin(azimuth) * np.cos(elevation)
        cam_z = np.sin(elevation)

        position = [cam_x, cam_y, cam_z]
        front = [-cam_x, -cam_y, -cam_z]

        if abs(cam_z) > 0.999:
            up = [0, 1, 0]
        else:
            right = np.cross([0, 0, 1], position)
            right = right / np.linalg.norm(right)
            up = np.cross(front, right)
            up = up / np.linalg.norm(up)

        views.append({"position": position, "front": front, "up": up})

    return views[:n_views]  # Return exactly n_views camera positions


def render_mesh(mesh, view_params, image_size=224):
    """Render a mesh from a specific viewpoint"""
    # Compute vertex normals for better rendering
    mesh.compute_vertex_normals()

    # Center and scale the mesh
    center = mesh.get_center()
    mesh.translate(-center)
    scale = np.max(np.abs(np.asarray(mesh.vertices)))
    mesh.scale(1 / scale, center=[0, 0, 0])

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=image_size, height=image_size)

    # Add geometry
    vis.add_geometry(mesh)

    # Set render options
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.background_color = np.array([1.0, 1.0, 1.0])  # White background

    # Set material
    mesh.paint_uniform_color([0.6, 0.6, 0.6])  # Darker gray color for better contrast

    # Set camera parameters
    ctr = vis.get_view_control()

    # Calculate camera distance
    scale = 2.5  # Adjust this to change how much of the view is filled

    # Update camera position
    cam_pos = np.array(view_params["position"]) * scale
    front = np.array(view_params["front"])
    up = np.array(view_params["up"])

    ctr.set_lookat([0, 0, 0])  # Look at center
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_zoom(0.7)

    # Add lighting
    render_option = vis.get_render_option()
    render_option.light_on = True

    # Render
    vis.poll_events()
    vis.update_renderer()

    # Capture image
    image = vis.capture_screen_float_buffer(False)
    vis.destroy_window()

    # Convert to PIL Image
    image_array = np.asarray(image) * 255
    image_array = image_array.astype(np.uint8)
    return Image.fromarray(image_array)


def process_cad(json_path, output_dir, n_views=8, image_size=224, pbar=None):
    """Process a single CAD file"""
    try:
        # Extract subfolder and base name from json_path
        rel_path = os.path.relpath(json_path, os.path.join("data", "cad_json"))
        subfolder = os.path.dirname(rel_path)
        base_name = Path(json_path).stem

        # Create the subfolder in output directory
        model_output_dir = os.path.join(output_dir, subfolder, base_name)

        # Check if directory exists and has all views rendered
        if os.path.exists(model_output_dir):
            existing_views = [
                f for f in os.listdir(model_output_dir) if f.endswith(".jpg")
            ]
            if len(existing_views) >= n_views:
                if pbar:
                    pbar.update(1)
                    pbar.write(
                        f"Skipping {json_path}: already has {len(existing_views)} views"
                    )
                return True

        # Create directory if it doesn't exist
        os.makedirs(model_output_dir, exist_ok=True)

        # Load and create CAD model
        with open(json_path, "r") as fp:
            data = json.load(fp)

        cad_seq = CADSequence.from_dict(data)
        cad_seq.normalize()
        shape = create_CAD(cad_seq)

        # Convert directly to Open3D mesh
        mesh = convert_shape_to_mesh(shape)

        # Generate views
        views = setup_camera_views(n_views)

        # Render each view
        for i, view in enumerate(views):
            # Check if this specific view already exists
            output_path = os.path.join(model_output_dir, f"view_{i:02d}.jpg")
            if os.path.exists(output_path):
                continue

            image = render_mesh(mesh, view, image_size)
            image.save(output_path, "JPEG", quality=95)

        if pbar:
            pbar.update(1)

        return True

    except Exception as e:
        if pbar:
            pbar.write(f"Error processing {json_path}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="source JSON file (optional)")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="which split to process when using train_val_test_split.json",
    )
    parser.add_argument(
        "--views", type=int, default=8, help="number of views to render"
    )
    parser.add_argument(
        "--size", type=int, default=224, help="size of output images (CLIP expects 224)"
    )
    parser.add_argument(
        "--subset",
        action="store_true",
        help="use subset_split.json instead of full dataset",
    )
    args = parser.parse_args()

    if args.src:
        # Process single file
        output_dir = os.path.join("data", "img_single")
        print(f"Processing single file: {args.src}")
        process_cad(args.src, output_dir, args.views, args.size)
        print(f"Output saved to: {output_dir}")
    else:
        # Process split
        split_file = os.path.join(
            "data", "subset_split.json" if args.subset else "train_val_test_split.json"
        )
        with open(split_file, "r") as f:
            splits = json.load(f)

        # Create output directory
        output_dir = os.path.join(
            "data", f"img_{args.split}_{'subset' if args.subset else ''}"
        )
        os.makedirs(output_dir, exist_ok=True)

        # Process all files in split
        json_files = splits[args.split]
        print(f"\nProcessing {len(json_files)} files from {args.split} split")
        print(f"Output directory: {output_dir}")
        print(f"Each model will have {args.views} views")
        print(f"Image size: {args.size}x{args.size} pixels")

        with tqdm(total=len(json_files), desc=f"Rendering {args.split} set") as pbar:
            for json_id in json_files:
                json_path = os.path.join("data", "cad_json", f"{json_id}.json")
                process_cad(json_path, output_dir, args.views, args.size, pbar)

        print(f"\nRendering complete! Output saved to: {output_dir}")
        print("Directory structure:")
        print(f"  {output_dir}/")
        print(f"  ├── 0000/")
        print(f"  │   ├── 00000070/")
        print(f"  │   │   ├── view_00.jpg  # First view")
        print(f"  │   │   ├── view_01.jpg  # Second view")
        print(f"  │   │   └── ... ({args.views} views)")
        print(f"  │   └── ...")
        print(f"  └── ...")


if __name__ == "__main__":
    main()
