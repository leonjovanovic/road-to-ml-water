import os
from pathlib import Path

import pysplishsplash as sph
import pyvista as pv
import numpy as np

def run_simulator(scene_path: str, output_dir: str):
    base = sph.Exec.SimulatorBase()
    print(f"Loading scene from: {scene_path}")
    base.init(str(scene_path), outputDir=output_dir)
    gui = sph.GUI.Simulator_GUI_imgui(base)
    base.setGui(gui)
    base.run()

def read_vtp_attributes(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None, None

    try:
        mesh = pv.read(filepath)
    except Exception as e:
        print(f"Error reading VTP file {filepath} with pyvista: {e}")
        return None, None

    points = mesh.points # This is your 'position' data as a NumPy array
    
    # PyVista stores point data in mesh.point_data
    # SPlisHSPlasH typically names attributes like 'velocity', 'density', 'id'
    attributes_data = {}
    for name in mesh.point_data: # Changed from array_names to iterate through dict keys
        attributes_data[name] = mesh.point_data[name]
    
    print(f"Read {filepath}: {points.shape[0]} particles")
    print(f"  Available attributes: {list(attributes_data.keys())}")
    return points, attributes_data, mesh.bounds


OUTPUT_DIR = "D:/Users/Leon Jovanovic/Documents/Computer Science/Machine Learning/road-to-ml-water/Stage 1 - Foundational 2D Fluid Simulation/data/run_1"
SCENES_DIR_PATH = "D:/Users/Leon Jovanovic/Documents/Computer Science/Computer Graphics/SPlisHSPlasH/data/Scenes"
SCENE_NAME = "SurfaceTension_BreakDamZR2020.json"
scene_path = f"{SCENES_DIR_PATH}/{SCENE_NAME}"
run_simulator(scene_path, OUTPUT_DIR)
read_vtp_attributes(f"{OUTPUT_DIR}/vtk/ParticleData_Fluid_1.vtk")

