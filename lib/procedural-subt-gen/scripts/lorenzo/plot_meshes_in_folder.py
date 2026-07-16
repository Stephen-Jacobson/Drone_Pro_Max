import pyvista as pv
import os
import argparse


def search_file_in_folder_recursive(folder, file_name):
    paths = []
    for element in os.listdir(folder):
        path_to_element = os.path.join(folder, element)
        if os.path.isdir(path_to_element):
            paths += search_file_in_folder_recursive(path_to_element, file_name)
        elif element == file_name:
            paths.append(path_to_element)
    return paths


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", required=True, type=str)
    return parser.parse_args()


def main():
    args = get_args()
    paths_to_meshes = search_file_in_folder_recursive(args.folder_path, "mesh.obj")
    paths_to_meshes.sort()
    for path_to_mesh in paths_to_meshes:
        print(path_to_mesh)
        plotter = pv.Plotter(off_screen=True)
        mesh = pv.read(path_to_mesh)
        plotter.add_mesh(
            mesh,
            edge_color="black",
            color="w",
            show_edges=True,
            line_width=0.1,
            render_lines_as_tubes=True,
            # diffuse=0,
            # lighting=False,
            # ambient=0.5,
        )
        plotter.disable_anti_aliasing()
        env_name = path_to_mesh.split("/")[-2]
        plotter.show(
            screenshot=f"/home/lorenzo/Documents/my_papers/ROBOT2023/images/e_{env_name}.png"
        )


if __name__ == "__main__":
    main()
