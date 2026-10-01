import numpy as np
import math as m
from scipy import interpolate
from subt_proc_gen.helper_functions import get_indices_close_to_point
from copy import copy

###############################################################
# CLASSES
###############################################################


class Point3D:
    def __init__(self, coords):
        self.set_coords(coords)

    def set_coords(self, coords):
        if isinstance(coords, list) or isinstance(coords, tuple):
            assert len(coords) == 3
            self._coords = np.reshape(np.array(coords, dtype=np.double), (1, 3))
        elif isinstance(coords, np.ndarray):
            assert coords.size == 3
            self._coords = np.reshape(coords.astype(np.double), (1, 3))
        elif isinstance(coords, Point3D):
            self._coords = coords.xyz
        else:
            raise Exception("Input for point not valid")

    def __sub__(self, other):
        if isinstance(other, Vector3D):
            return Point3D(self.xyz - other.xyz)
        elif isinstance(other, Point3D):
            return Vector3D(self.xyz - other.xyz)

    def __add__(self, other):
        if isinstance(other, Vector3D):
            return Point3D(self.xyz + other.xyz)
        elif isinstance(other, Point3D):
            raise TypeError("A Point cannot be added other Point")
        else:
            raise TypeError(f"Adding {type(other)} to Point not supported")

    def __eq__(self, other):
        if isinstance(other, Point3D):
            return hash(self) == hash(other)
        return False

    def __hash__(self) -> int:
        return hash((int(self.x), int(self.y), int(self.z)))

    @property
    def x(self):
        return self._coords[0, 0]

    @property
    def y(self):
        return self._coords[0, 1]

    @property
    def z(self):
        return self._coords[0, 2]

    @property
    def xyz(self):
        return self._coords


class Vector3D:
    @classmethod
    def from_inclination_yaw_length(cls, inclination, yaw, length):
        phi = inclination_to_phi(inclination)
        return Vector3D((length, yaw, phi), spherical_coords=True)

    @classmethod
    def random(cls, inclination_range, yaw_range, length=1):
        return cls.from_inclination_yaw_length(
            np.random.uniform(
                inclination_range[0],
                inclination_range[1],
            ),
            np.random.uniform(
                yaw_range[0],
                yaw_range[1],
            ),
            length=length,
        )

    def __init__(self, coords, spherical_coords=False):
        if isinstance(coords, Vector3D):
            self.__set_cartesian(coords.xyz)
        else:
            if spherical_coords:
                coords = format_coords(coords)
                self.__set_spherical(coords)
            else:
                coords = format_coords(coords)
                self.__set_cartesian(coords)

    def __abs__(self):
        return self._length

    def __add__(self, other):
        if isinstance(other, Point3D):
            return Point3D(self.xyz + other.xyz)
        elif isinstance(other, Vector3D):
            return Vector3D(self.xyz + other.xyz)
        else:
            raise NotImplementedError(f"Vector does not support adding {type(other)}")

    def __sub__(self, other):
        if isinstance(other, Point3D):
            return Point3D(self.xyz - other.xyz)
        elif isinstance(other, Vector3D):
            return Vector3D(self.xyz - other.xyz)
        else:
            raise NotImplementedError(
                f"Vector does not support subtracting {type(other)}"
            )

    def __str__(self):
        return f"[{self.x},{self.x},{self.z}]"

    def __eq__(self, other):
        if isinstance(other, Vector3D):
            return np.all(np.isclose(self.xyz, other.xyz, rtol=1e-10))
        return False

    def __set_cartesian(self, coords):
        self._cartesian = coords
        self._length = np.linalg.norm(self._cartesian, axis=1)
        if self._length == 0:
            self._unit_cartesian_vector = self._cartesian
        else:
            self._unit_cartesian_vector = self._cartesian / self.length
        self._spherical = np.reshape(
            np.array(
                (
                    self.length,
                    m.atan2(self.y, self.x),
                    m.atan2(self.xy_l, self.z),
                )
            ),
            (1, 3),
        )

    def __set_spherical(self, coords):
        self._spherical = coords
        self._length = coords[0, 0]
        self._cartesian = np.reshape(
            np.array(
                (
                    self.pho * m.sin(self.phi) * m.cos(self.theta),
                    self.pho * m.sin(self.phi) * m.sin(self.theta),
                    self.pho * m.cos(self.phi),
                )
            ),
            (1, 3),
        )
        self._unit_cartesian_vector = self._cartesian / self.length

    def set_distance(self, new_length):
        if self.length != 0:
            self.__set_cartesian(self.xyz / self.length * new_length)

    def normalize(self):
        self.set_distance(1.0)

    @property
    def length(self):
        return self._length.item()

    @property
    def cartesian_unitary(self) -> np.ndarray:
        return self._unit_cartesian_vector

    @property
    def x(self):
        return self._cartesian[0, 0].item()

    @property
    def y(self):
        return self._cartesian[0, 1].item()

    @property
    def z(self):
        return self._cartesian[0, 2].item()

    @property
    def xy(self):
        return self._cartesian[0, (0, 1)]

    @property
    def xyz(self):
        return self._cartesian

    @property
    def xy_l(self):
        return np.linalg.norm(self.xy, axis=0).item()

    @property
    def pho(self):
        return self._spherical[0, 0].item()

    @property
    def theta(self):
        return self._spherical[0, 1].item()

    @property
    def phi(self):
        return self._spherical[0, 2].item()

    @property
    def ptp(self):
        """pho_theta_phi"""
        return self._spherical

    @property
    def inclination(self):
        return phi_to_inclination(self.phi)

    @property
    def normalized(self):
        copy_of_self = copy(self)
        copy_of_self.normalize()
        return copy_of_self


class Spline3D:
    """Wrapper around the scipy spline to
    interpolate a series of 3d points along x,y and z"""

    def __init__(
        self, points, initial_dir: Vector3D = None, final_dir: Vector3D = None
    ):
        for p in points:
            assert isinstance(p, Point3D)
        self._points = list(points)
        self._n_input_points = len(self._points)
        self._input_points = np.zeros(shape=[self._n_input_points, 3], dtype=np.double)
        self._initial_dir = initial_dir
        self._final_dir = final_dir
        for i, p in enumerate(self._points):
            self._input_points[i, :] = p.xyz
        self._calculation_points = self._input_points
        if not self._initial_dir is None:
            self._initial_dir = self._initial_dir.normalized
            self._initial_dir.set_distance(0.1)
            initial_point = points[0] - self._initial_dir
            self._calculation_points = np.vstack(
                (initial_point.xyz, self._calculation_points)
            )
            self._initial_offset = 0.1
        else:
            self._initial_offset = 0
        if not self._final_dir is None:
            self._final_dir = self._final_dir.normalized
            self._final_dir.set_distance(0.1)
            final_point = points[-1] + self._final_dir
            self._calculation_points = np.vstack(
                (self._calculation_points, final_point.xyz)
            )
        self._n_calculation_points = len(self._calculation_points)
        dist_array = np.zeros(shape=[self._n_calculation_points, 1])
        for i in range(1, self._n_calculation_points):
            dist_array[i, :] = dist_array[i - 1] + np.linalg.norm(
                self._calculation_points[i, :] - self._calculation_points[i - 1, :]
            )
        self._dist_array = dist_array
        self._distance = float(dist_array[-1, 0])
        self._degree = 3 if len(dist_array) > 3 else len(dist_array) - 1
        self.xspline = interpolate.splrep(
            dist_array, self._calculation_points[:, 0], k=self._degree
        )
        self.yspline = interpolate.splrep(
            dist_array, self._calculation_points[:, 1], k=self._degree
        )
        self.zspline = interpolate.splrep(
            dist_array, self._calculation_points[:, 2], k=self._degree
        )
        self._discretized_cache = dict()

    def to_json(self):
        pass

    def __call__(self, d):
        # assert d >= 0 and d <= self._distance
        d += self._initial_offset
        x = interpolate.splev(d, self.xspline)
        y = interpolate.splev(d, self.yspline)
        z = interpolate.splev(d, self.zspline)
        p = np.array([x, y, z], ndmin=2)
        x1 = interpolate.splev(d + 0.001, self.xspline)
        y1 = interpolate.splev(d + 0.001, self.yspline)
        z1 = interpolate.splev(d + 0.001, self.zspline)
        p1 = np.array([x1, y1, z1], ndmin=2)
        v = p1 - p
        v /= np.linalg.norm(v)
        return p, v

    def discretize(self, precision):
        if precision in self._discretized_cache:
            return self._discretized_cache[precision]
        # number of sampling points
        nd = int(np.ceil(self._distance / precision))
        # sampling distances
        ds = np.linspace(0, self._distance, nd)
        # sampled points and directions
        ps, vs = np.zeros([nd, 3]), np.zeros([nd, 3])
        for n, d in enumerate(ds):
            p, v = self(d)
            ps[n, :] = np.reshape(p, -1)
            vs[n, :] = np.reshape(v, -1)
        self._discretized_cache[precision] = ds, ps, vs
        return ds, ps, vs

    def get_most_perpendicular_point_in_spline(
        self, point, threshold_distance, discretization_precision
    ):
        ds, ps, vs = self.discretize(discretization_precision)
        # get only the points of the spline close to point
        ids = get_indices_close_to_point(
            ps, point, threshold_distance, horizontal_distance=False
        )
        if len(ids) == 0:
            return None
        ps, vs = ps[ids, :], vs[ids, :]
        # get the vectors that go from the point to the spline points
        vpps = point - ps
        dists = np.linalg.norm(vpps, axis=1)
        vpps /= np.reshape(dists, [-1, 1])
        # get the cross product between the spline orientation at each
        # of the spline point and the vector that goes from the spline
        # point to the point
        cps = np.cross(vpps, vs, axis=1)
        cpns = np.linalg.norm(cps, axis=1)
        i = np.argmax(cpns)
        if max(cpns) < 0.95:
            return None
        return (
            ds[i],
            ps[i, :],
            vs[i, :],
        )

    def distance_matrix(self, other, discretization_precision):
        assert isinstance(other, Spline3D)
        self_d = self.discretize(discretization_precision)[1]
        other_d = other.discretize(discretization_precision)[1]
        distance_matrix = np.ones([])

    def collides(self, other, threshold_distance):
        assert isinstance(other, Spline3D)

    def get_closest(self, point, precision):
        point = Point3D(point)
        ds, ps, vs = self.discretize(precision)
        diff = ps - point.xyz
        dist = np.linalg.norm(diff, axis=1)
        idx = np.argmin(dist)
        return (
            np.reshape(ds[idx], -1),
            np.reshape(ps[idx, :], (1, 3)),
            np.reshape(vs[idx, :], (1, 3)),
        )

    @property
    def metric_length(self):
        return self._distance


###############################################################
# FUNCTIONS
###############################################################
def phi_to_inclination(phi):
    return m.pi / 2 - phi


def inclination_to_phi(inclination):
    return -inclination + m.pi / 2


def format_coords(coords):
    if isinstance(coords, np.ndarray):
        if coords.shape == (1, 3):
            return coords
        else:
            assert coords.size == 3
            return np.reshape(coords, (1, 3)).astype(np.double)
    if (
        isinstance(coords, list)
        or isinstance(coords, tuple)
        or isinstance(coords, set)
        or isinstance(coords, frozenset)
    ):
        assert len(coords) == 3
        return np.reshape(np.array(coords, dtype=np.double), (1, 3))


def get_two_perpendicular_vectors(i_vector):
    vector = Vector3D(i_vector)
    pho, theta, phi = np.reshape(vector.ptp, -1)
    derived_vector = Vector3D(
        (pho, theta, phi - np.deg2rad(0.01)), spherical_coords=True
    )
    u = Vector3D(np.cross(vector.cartesian_unitary, derived_vector.cartesian_unitary))
    v = Vector3D(np.cross(vector.cartesian_unitary, u.cartesian_unitary))
    u.normalize()
    v.normalize()
    return u, v


def get_close_points_indices(point: np.ndarray, points: np.ndarray, distance):
    point = np.reshape(point, (1, 3))
    points = np.reshape(points, (-1, 3))
    diff = points - point
    dist = np.linalg.norm(diff, axis=1)
    return np.where(dist < distance)


def get_close_points_to_point(point: np.ndarray, points: np.ndarray, distance):
    point = np.reshape(point, (1, 3))
    points = np.reshape(points, (-1, 3))
    diff = points - point
    dist = np.linalg.norm(diff, axis=1)
    return np.reshape(points[np.where(dist < distance), :], (-1, 3))


def warp_angle_2pi(angle):
    return (angle + np.pi * 2) % (np.pi * 2)


def warp_angle_pi(angle):
    a = warp_angle_2pi(angle)
    return a if a < np.pi else 2 * np.pi - a


def distance_matrix(array1: np.ndarray, array2: np.ndarray):
    # array1 and array2 are of dimensions: AxN and BxN respevely
    # N is the dimensionality of the points
    A = array1.shape[0]
    B = array2.shape[0]
    assert array1.shape[1] == array2.shape[1]
    N = array1.shape[1]
    return np.reshape(
        np.linalg.norm(
            np.ones(
                (
                    A,
                    B,
                    N,
                )
            )
            * np.reshape(array1, (A, 1, N))
            - np.reshape(array2, (1, B, N)),
            axis=2,
        ),
        (A, B),
    )


def check_spline_collision(
    spline1: Spline3D,
    spline2: Spline3D,
    collision_distance,
    omision_points=None,
    omision_distances=None,
):
    discretization_precision = collision_distance / 4
    spd1 = spline1.discretize(discretization_precision)[1]
    spd2 = spline2.discretize(discretization_precision)[1]
    if not omision_points is None:
        if omision_distances is None:
            omision_distances = [
                collision_distance * 2 for _ in range(len(omision_points))
            ]
        for op, od in zip(omision_points, omision_distances):
            spd1 = np.delete(spd1, get_close_points_indices(op, spd1, od), axis=0)
            # spd2 = np.delete(spd2, get_close_points_indices(op, spd2, od), axis=0)
    return np.any(distance_matrix(spd1, spd2) < collision_distance)


def spline_collides_with_itself(spline: Spline3D, collision_distance):
    length = spline.metric_length
    res = collision_distance / 10
    _, points, _ = spline.discretize(res)
    dist_mat = distance_matrix(points, points)
    close_points_idi_idj = np.array(np.where(dist_mat < collision_distance)).T
    for i, j in close_points_idi_idj:
        if abs(i - j) > 10:
            return True
    return False


def get_uniform_points_in_sphere(n):
    points = np.zeros((n, 3))
    inc = m.pi * (3 - m.sqrt(5))
    offset = 2 / float(n)
    for i in range(n):
        y = ((i * offset) - 1) + (offset / 2)
        r = m.sqrt(1 - pow(y, 2))
        phi = (i % n) * inc
        x = m.cos(phi) * r
        z = m.sin(phi) * r
        points[i, :] = np.array((x, y, z))
    return points


###############################################################
# TESTING
###############################################################
def test1():
    p1 = Point3D(np.random.random([1, 3]))
    p2 = Point3D(np.random.random([1, 3]))
    v2 = p1 - p2
    assert isinstance(v2, Vector3D)
    p3 = p2 + v2
    assert isinstance(p3, Point3D)
    assert p3 == p1


def test2():
    p1 = Point3D(np.random.random([1, 3]))
    p2 = Point3D(np.random.random([1, 3]))
    v1 = p1 - p2
    v2 = Vector3D(v1.ptp, spherical_coords=True)
    assert v2 == v1


def test3():
    import pyvista as pv

    points = get_uniform_points_in_sphere(1000)
    points *= 10
    plotter = pv.Plotter()
    points = pv.PolyData(points)
    plotter.add_mesh(points)
    plotter.show()


def main():
    test1()
    test2()


if __name__ == "__main__":
    for _ in range(200):
        main()
    test3()
