import os
import pyvista as pv
import cv2
import logging


logging.basicConfig(level=logging.DEBUG)
# PARAMS
base_video_folder = "/home/lorenzo/Documents/my_papers/ICRA2024_procedural/video/variations_on_same_environment"
folder_with_meshes = "/home/lorenzo/Documents/my_papers/ICRA2024_procedural/video/variations_on_same_environment/meshes"
configure_camera = True
azimuth_increment = 0
elevation_increment = 0
window_size = (1920, 1080)


# START
class MyPlotter:
    def __init__(self, azimuth_increment, elevation_increment):
        self.n_captures_per_change = 10
        self.azimuth_increment = azimuth_increment
        self.elevation_increment = elevation_increment
        self.plotter = pv.Plotter(off_screen=True, window_size=window_size)
        self.plotter.set_background(color="white")
        self.n_image = 0
        self.n_trial = len(os.listdir(base_video_folder))
        self.images_folder = os.path.join(base_video_folder, f"trial_n_{self.n_trial}")
        os.mkdir(self.images_folder)
        self.set_camera_params()
        self.images = []

    def plot_mesh(self, mesh, color, capture_at_end=True):
        actor = self.plotter.add_mesh(mesh, show_edges=True, color=color)
        if capture_at_end:
            for _ in range(self.n_captures_per_change):
                self.save_capture()
        return actor

    def save_capture(self):
        path_to_image = os.path.join(self.images_folder, f"last_image.png")
        self.plotter.show(
            auto_close=False,
            screenshot=path_to_image,
        )
        self.plotter.camera.azimuth += self.azimuth_increment
        self.plotter.camera.elevation += self.elevation_increment
        self.n_image += 1
        self.images.append(cv2.imread(path_to_image))

    def print_camera_params(self):
        print(f"self.plotter.camera.position = {self.plotter.camera.position}")
        print(f"self.plotter.camera.focal_point= {self.plotter.camera.focal_point}")
        print(f"self.plotter.camera.roll = {self.plotter.camera.roll}")

    def set_camera_params(self, camera=None):
        if not camera is None:
            self.plotter.camera = camera
            return
        mult = 2
        self.plotter.camera.position = (
            104.5786207334545 * mult,
            111.92568096131357 * mult,
            86.94361170474676 * mult,
        )
        self.plotter.camera.focal_point = (
            -8.070702349350167,
            -0.8108855992463173,
            13.159967378983225,
        )
        self.plotter.camera.roll = -113.67940228246788

    def create_video(self):
        logging.info("Creating Video")
        path_to_video_file = os.path.join(self.images_folder, "video.avi")
        height, width, layers = self.images[0].shape
        self.video_writer = cv2.VideoWriter(path_to_video_file, 0, 24, (width, height))
        for frame in self.images:
            self.video_writer.write(frame)
        cv2.destroyAllWindows()
        self.video_writer.release()

    def clear_all(self):
        self.plotter.clear()


def main():
    my_plotter = MyPlotter(
        azimuth_increment=azimuth_increment, elevation_increment=elevation_increment
    )
    mesh_file_names = os.listdir(folder_with_meshes)
    mesh_file_names.sort()
    n_files = len(mesh_file_names)
    my_plotter.plotter.camera.azimuth += 90
    for n_mesh_file, mesh_file in enumerate(mesh_file_names):
        print(f"Mesh {n_mesh_file+1:04d} out of {n_files}", end="\r", flush=True)
        path_to_mesh = os.path.join(folder_with_meshes, mesh_file)
        mesh = pv.read(path_to_mesh)
        if n_mesh_file == 0 and configure_camera:
            configuration_plotter = pv.Plotter(window_size=window_size)
            configuration_plotter.add_mesh(mesh, show_edges=True)
            configuration_plotter.show()
            my_plotter.set_camera_params(configuration_plotter.camera)
        my_plotter.clear_all()
        my_plotter.plot_mesh(mesh, color="white")
    my_plotter.create_video()


if __name__ == "__main__":
    main()
