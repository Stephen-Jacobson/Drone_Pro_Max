import pyvista as pv
import numpy as np
from noise import pnoise2
import random

seed = random.randint(0, 10000)

x, y, z = 100, 100, 100
scale = 0.01
values = np.zeros((x, y, z), dtype=np.uint8)

#gets height per x y spot on grid to give a bumpy terrain, from gpt, for bumpier, increase octaves, persistence, amplitude, for less bumpy, decrease octaves, persistence eg 0.3, lower scale
# Scale: up - denser bumps, down - wide smooth hills
# Persistence: up - rough/jagged, down - smooth

def fractal_height(x, y, seed=seed, scale=scale, octaves=5, persistence=0.01, amplitude=50):
    height = 0
    frequency = 1
    amp = 1

    max_amp = 0  # normalization factor

    for i in range(octaves):
        n = pnoise2(
            (x + seed) * scale * frequency,
            (y + seed) * scale * frequency
        )
        n = 1 - abs(n)        #uncommenting will make look more duney
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

# TODO generate random start and end points on map

# def generate_start_and_end():

# TODO generate random points until end, maybe with changable varyation
# like can be a very winding path or points are more in a line shape, then connect points using cubic splines which will create smooth path

# def generate_random_path():


# uses number 1 as block indicator for terrain 
# uses number 2 as block indicator for trees 
# uses number 3 as block indicator for tree line
# added tree line so could stop pathfinder from going above trees 
def generate_terrain(tree_chance=0.01, tree_height=15, gen_trees=True, tree_line_height=15, tree_line_recede=0):
    for i in range(x):
        for j in range(y):
            terr_z = int(fractal_height(i, j))
            if gen_trees:
                tile_num = i * y + j
                spawn_tree(i, j, terr_z, tree_height, tree_chance, tile_num)
            for k in range(terr_z + 1):
                values[i][j][k] = 1
            # if scale > 0.7:
            #     tree_line_z = int(fractal_height(i, j, scale=scale-scale*0.5));
            for k in range(tree_line_height + 1, z-terr_z):
                recede = tree_line_height + 1 + tree_line_recede
                
                if values[i][j][terr_z + k] == 0 and terr_z + k > recede:
                        
                    values[i][j][terr_z + k] = 3

def spawn_tree(tx, ty, tz, tree_height, tree_chance, tile_num):
    random.seed(seed + tile_num)
    tree = random.random()
    if tree <= tree_chance:
        for i in range(1, tree_height + 1):
            values[tx][ty][tz + i] = 2

def show_grid():
    grid = pv.ImageData()
    grid.dimensions = np.array(values.shape) + 1
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)

    grid.cell_data["values"] = values.flatten(order="F")

    terrain = grid.threshold([0.5, 1.5], scalars="values")
    trees = grid.threshold([1.5, 2.5], scalars="values")
    tree_line = grid.threshold([2.5, 3.5], scalars="values")
    
    plotter = pv.Plotter()
    plotter.add_mesh(terrain, show_edges=False, color='#e07a5f')
    plotter.add_mesh(trees, show_edges=False, color='#432818')
    plotter.add_mesh(tree_line, show_edges=False, color='#00b4d8', opacity=0.1)
    plotter.show()

def main():
    generate_terrain()
    show_grid()

if __name__ == "__main__":
    main()

