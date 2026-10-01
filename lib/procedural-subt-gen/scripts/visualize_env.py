import pyvista as pv
from pyvista.plotting.plotter import Plotter
import argparse
import os
import numpy as np
from xml.etree.ElementTree import ElementTree
import vtk
vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_of_env",required=True)
    return parser.parse_args()

def main():
    args=parse_args()
    folder = args.folder_of_env
    fta_dist_path = os.path.join(folder, "fta_dist.txt")
    mesh_path = os.path.join(folder, "mesh.obj")
    model_path = os.path.join(folder, "model.sdf")
    axis_path = os.path.join(folder, "axis.txt")
    fta_dist_data = np.loadtxt(fta_dist_path)
    mesh_data = pv.read(mesh_path)
    model_data = ElementTree()
    model_data.parse(model_path)
    axis_data = np.loadtxt(axis_path)
    plotter = Plotter()
    plotter.add_mesh(mesh_data,style="wireframe")
    plotter.add_mesh(pv.PolyData(axis_data[:,:3]),scalars=axis_data[:,7])
    print(axis_data.shape)
    plotter.show()
    
    
    

if __name__ == "__main__":
    main()