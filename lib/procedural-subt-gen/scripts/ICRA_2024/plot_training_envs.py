import pyvista as pv
import os

pv.rcParams["transparent_background"] = True
BASE_FOLDER = "/media/lorenzo/SAM500/datasets/slibon/50_tunnels_4000_per_tunnel"
img_save_folder = "/home/lorenzo/Documents/conferences/ICRA_25/video/training_envs"
for n, folder in enumerate(os.listdir(BASE_FOLDER)):
    plotter = pv.Plotter(off_screen=True)
    path_to_mesh = os.path.join(BASE_FOLDER, folder, "mesh.obj")
    mesh = pv.read_meshio(path_to_mesh)
    plotter.add_mesh(mesh, color="white")
    img_path = os.path.join(img_save_folder, f"{n:03d}.png")
    plotter.show(screenshot=img_path)
