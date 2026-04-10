import pyvista as pv
import numpy as np
from noise import pnoise2
import random

seed = random.randint(0, 10000)

x, y, z = 1000, 1000, 100
values = np.zeros((x, y, z), dtype=np.uint8)

#gets height per x y spot on grid to give a bumpy terrain, from gpt, for bumpier, increase octaves, persistence, amplitude, for less bumpy, decrease octaves, persistence eg 0.3, lower scale
# Scale: up - denser bumps, down - wide smooth hills
# Persistence: up - rough/jagged, down - smooth

def fractal_height(x, y, seed=0, scale=0.01, octaves=5, persistence=0.1, amplitude=100):
    height = 0
    frequency = 1
    amp = 1

    max_amp = 0  # normalization factor

    for i in range(octaves):
        n = pnoise2(
            (x + seed) * scale * frequency,
            (y + seed) * scale * frequency
        )
        # n = 1 - abs(n)
        height += amp * n
        max_amp += amp
        amp *= persistence
        frequency *= 2

    # normalize to [-1, 1]
    height /= max_amp

    # map to [0, A]
    A = amplitude
    z = (height + 1) * (A / 2)

    # optional safety clamp (recommended)
    return max(0, min(A, z))

def show_grid():
    grid = pv.ImageData()
    grid.dimensions = np.array(values.shape) + 1
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)

    grid.cell_data["values"] = values.flatten(order="F")

    thresholded = grid.threshold(value=0.5, scalars="values")
    
    plotter = pv.Plotter()
    plotter.add_mesh(thresholded, show_edges=False, color='#e07a5f')
    plotter.show()

def main():
    for i in range(x):
        for j in range(y):
            terr_z = fractal_height(i, j)
            if terr_z == 0:
                print("gay")
            for k in range(int(terr_z)):
                values[i][j][k] = 1
    
    show_grid()

if __name__ == "__main__":
    main()

