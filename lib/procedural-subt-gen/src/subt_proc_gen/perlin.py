import numpy as np
from perlin_numpy import generate_fractal_noise_2d, generate_fractal_noise_3d
from subt_proc_gen.param_classes import PerlinParams


class CylindricalPerlinNoiseGenerator:
    def __init__(
        self, length: float, radius: float, perlin_params: PerlinParams, min_size=100
    ):
        # Copy arguments
        self.length = length
        self.radius = radius
        self.perlin_params = perlin_params
        self.min_size = min_size
        # Unpack perlin params
        res = perlin_params.res
        octaves = perlin_params.octaves
        persistence = perlin_params.persistence
        lacunarity = perlin_params.lacunarity
        # Derived arguments
        circumference = radius * 2 * np.pi
        ratio_len_cirum = int(np.ceil(length / circumference))
        min_dimension = (lacunarity ** (octaves - 1)) * res
        if min_dimension < min_size:
            min_dimension *= int(np.ceil(min_size / min_dimension))
        shape = np.array(
            (
                min_dimension * ratio_len_cirum,
                min_dimension,
            ),
            dtype=int,
        )
        self.noise_image = generate_fractal_noise_2d(
            shape=shape,
            res=(
                int(res * ratio_len_cirum),
                int(res),
            ),
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity,
            tileable=(False, True),
        )
        self.circumference = circumference

    def __call__(self, coords: np.ndarray):
        """Coords must be an array of size Nx2, the first column must be the position along the axis,
        and the second the angle around the cross section"""
        shape = self.noise_image.shape
        l_coord = np.floor(coords[:, 0] / self.circumference * shape[1]).astype(int)
        a_coord = np.floor(coords[:, 1] / (2 * np.pi) * (shape[1] - 1)).astype(int)
        noise = self.noise_image[l_coord, a_coord]
        return noise


class SphericalPerlinNoiseMapper:
    def __init__(
        self,
        perlin_params: PerlinParams,
        min_size: int = 50,
    ):
        # Copy arguments
        self.perlin_params = perlin_params
        # Extract perlin params
        res = perlin_params.res
        octaves = perlin_params.lacunarity
        persistence = perlin_params.persistence
        lacunarity = perlin_params.lacunarity
        # Derived arguments
        min_dimension = (lacunarity ** (octaves - 1)) * res
        if min_dimension < min_size:
            min_dimension *= int(np.ceil(min_size / min_dimension))
        shape = np.array(
            [min_dimension for _ in range(3)],
            dtype=int,
        )
        noise_cube = generate_fractal_noise_3d(
            shape=shape,
            res=[res for _ in range(3)],
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity,
            tileable=(False, False, False),
        )
        self.noise_cube = (noise_cube - 1) / 2

    def __call__(self, coords: np.ndarray):
        """Coords must be an array of size Nx3, each row representing an xyz coordinate. Each coordinate can go from -1 to +1"""
        shape = self.noise_cube.shape
        scaled_coords = np.floor((coords + 1) / 2 * (shape[0] - 1)).astype(int)
        x = scaled_coords[:, 0]
        y = scaled_coords[:, 1]
        z = scaled_coords[:, 2]
        return self.noise_cube[x, y, z]
