from subt_proc_gen.geometry import Point3D, Spline3D, Vector3D
from subt_proc_gen.display_functions import plot_spline, plot_nodes
import pyvista as pv


def test_1():
    points = (
        Point3D((0, 0, 0)),
        Point3D((10, 0, 0)),
        Point3D((10, 10, 0)),
        Point3D((10, 20, 0)),
    )
    spline_1 = Spline3D(points, final_dir=Vector3D((0, 1, 0)))
    spline_2 = Spline3D(points, initial_dir=Vector3D((1, 0, 0)))
    spline_3 = Spline3D(
        points, initial_dir=Vector3D((1, 0, 0)), final_dir=Vector3D((0, 1, 0))
    )
    spline_1(-1)
    plotter = pv.Plotter()
    plot_spline(plotter, spline_1)
    plot_spline(plotter, spline_2, color="b")
    plot_spline(plotter, spline_3, color="g")
    plot_nodes(plotter, points)
    plotter.show()


def main():
    test_1()


if __name__ == "__main__":
    main()
