import os
import json
import numpy as np
import argparse
import sys
from pathlib import Path
import vtk
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.TopoDS import TopoDS_Shape

# Fix import path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from cadlib.extrude import CADSequence
from cadlib.visualize import create_CAD

def convert_to_stl(shape: TopoDS_Shape, stl_path: str, linear_deflection=0.1, angular_deflection=0.1):
    """Convert OpenCascade shape to STL file"""
    # Create the mesh
    mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    mesh.Perform()
    
    # Write STL
    stl_writer = StlAPI_Writer()
    stl_writer.Write(shape, stl_path)
    return stl_path

def show_stl(stl_path):
    """Visualize STL file using VTK"""
    # Create the reader
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_path)
    
    # Create mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())
    
    # Create actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.8, 0.8, 0.8)  # Light gray color
    actor.GetProperty().SetAmbient(0.1)
    actor.GetProperty().SetDiffuse(0.7)
    actor.GetProperty().SetSpecular(0.3)
    
    # Create renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.2, 0.2, 0.2)  # Dark gray background
    
    # Create render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)
    
    # Create interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    # Set interaction style
    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)
    
    # Initialize and start
    interactor.Initialize()
    render_window.Render()
    interactor.Start()

def process_cad(json_path, output_dir=None, show=True):
    """Process a CAD JSON file to STL and optionally show it"""
    try:
        print(f"Processing: {json_path}")
        
        # Create output directory if needed
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(json_path), 'stl_models')
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and create CAD model
        with open(json_path, 'r') as fp:
            data = json.load(fp)
        
        cad_seq = CADSequence.from_dict(data)
        cad_seq.normalize()
        shape = create_CAD(cad_seq)
        
        # Convert to STL
        stl_path = os.path.join(output_dir, Path(json_path).stem + '.stl')
        convert_to_stl(shape, stl_path)
        print(f"Saved STL to: {stl_path}")
        
        if show:
            show_stl(stl_path)
        
        return stl_path
        
    except Exception as e:
        print(f"Error processing {json_path}:")
        import traceback
        print(traceback.format_exc())
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help="source JSON file or directory")
    parser.add_argument('--output', type=str, help="output directory for STL files")
    parser.add_argument('--no-show', action='store_true', help="don't show the visualization")
    args = parser.parse_args()
    
    if os.path.isfile(args.src):
        # Single file
        process_cad(args.src, args.output, not args.no_show)
    else:
        # Directory
        json_files = sorted([f for f in Path(args.src).rglob('*.json')])
        print(f"Found {len(json_files)} JSON files")
        for json_file in json_files:
            process_cad(str(json_file), args.output, not args.no_show)

if __name__ == "__main__":
    main() 