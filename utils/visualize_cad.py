import os
import glob
import json
import numpy as np
import open3d as o3d
import argparse
import sys
import traceback  # Add this for better error tracking

# Fix import path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from cadlib.extrude import CADSequence
from cadlib.visualize import CADsolid2pc, create_CAD

def visualize_cad(json_path, num_points=8096, save_image=False):
    """Visualize a CAD model from JSON file using Open3D"""
    try:
        print(f"1. Opening JSON file: {json_path}")
        # Check if file exists
        if not os.path.exists(json_path):
            print(f"Error: File does not exist: {json_path}")
            return
            
        # Load and create CAD model
        with open(json_path, 'r') as fp:
            data = json.load(fp)
        print("2. JSON file loaded successfully")
        
        print("3. Creating CAD sequence")
        cad_seq = CADSequence.from_dict(data)
        cad_seq.normalize()
        print("4. Creating CAD shape")
        shape = create_CAD(cad_seq)
        
        # Convert to point cloud
        print("5. Converting to point cloud")
        points = CADsolid2pc(shape, num_points, os.path.basename(json_path))
        print(f"6. Generated point cloud with {len(points)} points")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Visualize
        print("7. Creating visualization window")
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)  # Make sure window is visible
        vis.add_geometry(pcd)
        
        # Set some nice visualization parameters
        print("8. Setting visualization parameters")
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.asarray([0.8, 0.8, 0.8])
        
        # Set camera position
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([1, -1, -1])
        ctr.set_up([0, 0, 1])
        
        if save_image:
            # Save image before showing
            image_path = os.path.splitext(json_path)[0] + '_render.png'
            vis.capture_screen_image(image_path, True)
            print(f"9. Saved render to: {image_path}")
        
        print("10. Running visualizer")
        vis.run()
        vis.destroy_window()
        print("11. Visualization complete")
        
    except Exception as e:
        print(f"Error processing {json_path}:")
        print(traceback.format_exc())  # This will print the full error traceback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help="source folder or file")
    parser.add_argument('--num_points', type=int, default=8096, help="number of points to sample")
    parser.add_argument('--save_images', action='store_true', help="save rendered images")
    args = parser.parse_args()
    
    print(f"\nStarting visualization with arguments:")
    print(f"Source: {args.src}")
    print(f"Number of points: {args.num_points}")
    print(f"Save images: {args.save_images}\n")
    
    if os.path.isfile(args.src):
        # Single file
        print('Visualizing CAD model...{}'.format(args.src))
        visualize_cad(args.src, args.num_points, args.save_images)
    else:
        # Directory
        json_files = sorted(glob.glob(os.path.join(args.src, "**/*.json"), recursive=True))
        print(f"Found {len(json_files)} JSON files")
        for json_file in json_files:
            print(f"\nProcessing: {json_file}")
            visualize_cad(json_file, args.num_points, args.save_images)

if __name__ == "__main__":
    main() 