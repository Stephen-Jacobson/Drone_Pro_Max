import numpy as np
from enum import Enum


class PerlinParams:
    # Default params
    _default_res = 3
    _default_octaves = 3
    _default_persistence = 2
    _default_lacunarity = 2

    # Random params
    _random_res_range = [1, 5]
    _random_octaves_range = [2, 3]
    _random_persitence_range = [1.5, 5]
    _random_lacunarity_range = [2, 3]

    def __init__(
        self,
        res: int = None,
        octaves: int = None,
        persitence: float = None,
        lacunarity: int = None,
    ):
        self.res = res if not res is None else self._default_res
        self.octaves = octaves if not octaves is None else self._default_octaves
        self.persistence = persitence if not persitence is None else self._default_persistence
        self.lacunarity = lacunarity if not lacunarity is None else self._default_lacunarity

    @classmethod
    def random(cls):
        res = np.random.randint(cls._random_res_range[0], cls._random_res_range[1])
        octaves = np.random.randint(cls._random_octaves_range[0], cls._random_octaves_range[1])
        persitence = np.random.randint(cls._random_persitence_range[0], cls._random_persitence_range[1])
        lacunarity = np.random.randint(cls._random_lacunarity_range[0], cls._random_lacunarity_range[1])
        return cls(res, octaves, persitence, lacunarity)

    @classmethod
    def from_defaults(cls):
        return cls()


class TunnelPtClGenParams:
    """Contains parameters for the generarion of a pointcloud around a single tunnel"""

    # Default params
    _default_dist_between_circles = 0.5
    _default_n_points_per_circle = 30
    _default_radius = 5
    _default_noise_multiplier = 0.9
    # Random params
    _random_radius_interval = (5, 6)
    _random_noise_multiplier_interval = (0.1, 0.9)

    def __init__(
        self,
        dist_between_circles: float = None,
        n_points_per_circle: int = None,
        radius: float = None,
        noise_multiplier: float = None,
        perlin_params: PerlinParams = None,
    ):
        self.dist_between_circles = dist_between_circles if not dist_between_circles is None else self._default_dist_between_circles
        self.n_points_per_circle = n_points_per_circle if not n_points_per_circle is None else self._default_n_points_per_circle
        self.radius = radius if not radius is None else self._default_radius
        self.noise_multiplier = noise_multiplier if not noise_multiplier is None else self._default_noise_multiplier
        self.perlin_params = perlin_params if not perlin_params is None else PerlinParams.from_defaults()

    @classmethod
    def from_defaults(cls):
        return cls()

    @classmethod
    def random(cls):
        dist_between_circles = cls._default_dist_between_circles
        n_points_per_circle = cls._default_n_points_per_circle
        radius = np.random.uniform(
            cls._random_radius_interval[0],
            cls._random_radius_interval[1],
        )
        noise_multiplier = np.random.uniform(
            cls._random_noise_multiplier_interval[0],
            cls._random_noise_multiplier_interval[1],
        )
        perlin_params = PerlinParams.random()
        return TunnelPtClGenParams(
            dist_between_circles=dist_between_circles,
            n_points_per_circle=n_points_per_circle,
            radius=radius,
            noise_multiplier=noise_multiplier,
            perlin_params=perlin_params,
        )


class IntersectionPtClType(Enum):
    """Types of intersections to create
    - no_cavity: The points of the tunnels that are inside other tunnels are delted
    - cavity_along_one_tunnel: The pointcloud around one of the tunnels is inflated, and the
    """

    no_cavity = 1
    spherical_cavity = 2
    cavity_along_one_tunnel = 3


class IntersectionPtClGenParams:
    """Params that control how the pointcloud of an intersection is generated"""

    _default_radius = 20
    _default_type = IntersectionPtClType.no_cavity
    _default_points_per_sm = 3  # Points per sqare meter
    _default_noise_multiplier = 0.4

    _random_radius_range = (20, 25)
    _random_type_choices = (
        IntersectionPtClType.no_cavity,
        # IntersectionPtClType.spherical_cavity,
    )
    _random_noise_multiplier_range = (0.2, 0.4)

    def __init__(
        self,
        radius: float = None,
        ptcl_type: IntersectionPtClType = None,
        perlin_params: PerlinParams = None,
        points_per_sm: int = None,
        noise_multiplier: float = None,
    ):
        self.radius = radius if not radius is None else self._default_radius
        self.ptcl_type = ptcl_type if not ptcl_type is None else self._default_type
        self.perlin_params = perlin_params if not perlin_params is None else PerlinParams.from_defaults()
        self.points_per_sm = points_per_sm if not points_per_sm is None else self._default_points_per_sm
        self.noise_multiplier = noise_multiplier if not noise_multiplier is None else self._default_noise_multiplier

    @classmethod
    def from_defaults(cls):
        return cls()

    @classmethod
    def random(cls):
        radius = np.random.uniform(
            cls._random_radius_range[0],
            cls._random_radius_range[1],
        )
        ptcl_type = np.random.choice(cls._random_type_choices)
        perlin_params = PerlinParams.random()
        points_per_sm = cls._default_points_per_sm
        noise_multiplier = np.random.uniform(
            cls._random_noise_multiplier_range[0],
            cls._random_noise_multiplier_range[1],
        )
        return cls(
            radius=radius,
            ptcl_type=ptcl_type,
            perlin_params=perlin_params,
            points_per_sm=points_per_sm,
            noise_multiplier=noise_multiplier,
        )


class TunnelNetworkPtClGenStrategies(Enum):
    """Different strategies to set the parameters of the ptcl
    generation of each of the tunnels"""

    random = 1
    default = 2


class TunnelNetworkPtClGenParams:
    """Params that control the the overall generation of the pointcloud of the
    complete tunnel network"""

    def __init__(
        self,
        ptcl_gen_strategy: TunnelNetworkPtClGenStrategies = None,
        pre_set_tunnel_params: dict = dict(),
        pre_set_intersection_params: dict = dict(),
    ):
        self.strategy = ptcl_gen_strategy if not ptcl_gen_strategy is None else self._default_ptcl_gen_strategy
        self.pre_set_tunnel_params = pre_set_tunnel_params
        self.pre_set_intersection_params = pre_set_intersection_params

    @classmethod
    def from_defaults(cls, pre_set_tunnel_params=dict(), pre_set_intersection_params=dict()):
        return cls(
            ptcl_gen_strategy=TunnelNetworkPtClGenStrategies.default,
            pre_set_tunnel_params=pre_set_tunnel_params,
            pre_set_intersection_params=pre_set_intersection_params,
        )

    @classmethod
    def random(cls, pre_set_tunnel_params=dict(), pre_set_intersection_params=dict()):
        return cls(
            ptcl_gen_strategy=TunnelNetworkPtClGenStrategies.random,
            pre_set_tunnel_params=pre_set_tunnel_params,
            pre_set_intersection_params=pre_set_intersection_params,
        )


class TunnelNetworkMeshGenParams:
    """Params that control how the mesh is generated from the ptcl"""

    _default_poisson_depth = 11
    _default_simplification_voxel_size = 0.5
    _default_voxelization_voxel_size = 5
    _default_fta_distance = -1
    _default_floor_smoothing_iter = 20
    _default_floor_smoothing_r = 2

    _random_fta_distance_range = [-2, 0]

    def __init__(
        self,
        poisson_depth: int = None,
        simplification_voxel_size: float = None,
        voxelization_voxel_size: float = None,
        fta_distance: float = None,
        floor_smoothing_iter: int = None,
        floor_smoothing_r: float = None,
    ):
        self.poisson_depth = poisson_depth if not poisson_depth is None else self._default_poisson_depth
        self.simplification_voxel_size = (
            simplification_voxel_size if not simplification_voxel_size is None else self._default_simplification_voxel_size
        )
        self.voxelization_voxel_size = voxelization_voxel_size if not voxelization_voxel_size is None else self._default_voxelization_voxel_size
        self.fta_distance = fta_distance if not fta_distance is None else self._default_fta_distance
        self.floor_smoothing_iter = floor_smoothing_iter if not floor_smoothing_iter is None else self._default_floor_smoothing_iter
        self.floor_smoothing_r = floor_smoothing_r if not floor_smoothing_r is None else self._default_floor_smoothing_r

    @classmethod
    def from_defaults(cls):
        return cls()

    @classmethod
    def random(cls):
        instance = cls()
        instance.fta_distance = np.array(np.random.uniform(cls._random_fta_distance_range[0], cls._random_fta_distance_range[1])).item(0)
        return instance
