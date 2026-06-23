import os                                       # have to do this so it works for me, idk why, delete if it messes with things
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

import pyvista as pv
import numpy as np
from noise import pnoise2
import random
from scipy.interpolate import CubicSpline
from skimage.draw import line

from drone import Drone

pv.global_theme.allow_empty_mesh = True

seed = random.randint(0, 10000)

def reseed(new_seed=None):
    """Re-roll the terrain/tree RNG seed. Call this before generate_terrain()/
    generate_trees() whenever you want a genuinely NEW map — without this,
    seed never changes after module import, so every "regenerated" world
    was actually bit-identical (same heights, same trees, same start/end)."""
    global seed
    seed = new_seed if new_seed is not None else random.randint(0, 10000)

x, y, z = 100, 30, 100
scale = 0.01
values = np.zeros((x, y, z), dtype=np.uint8)        #smallest dtype so that takes least amount of memory - 0-255
ground_level = np.zeros((x,y), dtype=np.uint16)     #bigger but still small, must hold ground levl values - 0-65535

tree_line_height = 15
tree_line_recede = 0
tree_density = 0.01

# gets height per x y spot on grid to give a bumpy terrain, from gpt, for bumpier, increase octaves, persistence, amplitude, for less bumpy, decrease octaves, persistence eg 0.3, lower scale
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

def generate_start_and_end():
    min_distance = max(x, y) / 2.0
 
    # Build a flat list of all valid candidate positions across the whole grid
    all_candidates = []
    for cx in range(x):
        for cy in range(y):
            z_levels = valid_z_values(cx, cy)
            if z_levels:
                all_candidates.append((cx, cy))
 
    if len(all_candidates) < 2:
        raise RuntimeError("Not enough valid grid positions to place start and end")
 
    random.shuffle(all_candidates)
 
    # Pick a random start, then find any end that satisfies min_distance
    # Shuffle candidates so both start and end positions are unpredictable
    for i, (sx, sy) in enumerate(all_candidates):
        sz = random.choice(valid_z_values(sx, sy))
        start = (sx, sy, sz)
 
        # Collect all candidates far enough away and pick one at random
        far_enough = [
            (cx, cy) for cx, cy in all_candidates
            if ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5 >= min_distance
        ]
 
        if not far_enough:
            continue
 
        ex, ey = random.choice(far_enough)
        ez = random.choice(valid_z_values(ex, ey))
        end = (ex, ey, ez)
 
        # Mark positions in grid: 5 for drone/start, 4 for end
        values[start[0]][start[1]][start[2]] = 5
        values[end[0]][end[1]][end[2]] = 4
        return start, end
 
    raise RuntimeError("Could not find a valid start/end pair meeting the minimum distance")
 
def valid_z_values(cx, cy):
    valid_levels = list(range(ground_level[cx][cy] + 1, ground_level[cx][cy] + tree_line_height + tree_line_recede))       #valid range of z values based on what will be open
 
    return valid_levels
    
def generate_points(
    start,
    end,
    n_points,
    min_distance=0.0,
    max_distance=1.0,
    overlap=0.2
):
    start = np.array(start[:2], dtype=float)
    end = np.array(end[:2], dtype=float)

    direction = end - start
    length = np.linalg.norm(direction)
    if length == 0:
        raise ValueError("Start and end cannot be the same point")

    unit_dir = direction / length

    # perpendicular vector
    perp = np.array([-unit_dir[1], unit_dir[0]])

    segment_size = 1.0 / (n_points + 1)

    width, height = values.shape[0], values.shape[1]

    points = []

    for i in range(n_points):
        t_center = (i + 1) * segment_size

        t_min = max(0.0, t_center - segment_size * (0.5 + overlap * 0.5))
        t_max = min(1.0, t_center + segment_size * (0.5 + overlap * 0.5))

        t = random.uniform(t_min, t_max)

        base_point = start + t * direction

        # --------- MIN/MAX DISTANCE FIX ----------
        offset_mag = random.uniform(min_distance, max_distance)
        if random.random() < 0.5:
            offset_mag *= -1

        point = base_point + perp * offset_mag

        x, y = point

        # --------- BOUNDS CHECK ----------
        if x < 0 or y < 0 or x >= width or y >= height:
            continue  # skip invalid points

        x = int(x)
        y = int(y)

        # --------- SAFE GRID WRITE ----------
        values[x, y, 0] = 6

        points.append({
            "point": (x, y),
            "t": t,
            "offset": offset_mag,
            "region": (t_min, t_max)
        })
    
    points.insert(0, {"point": (int(start[0]), int(start[1])), "t": 0.0, "offset": 0, "region": (0, 0)})
    points.append(   {"point": (int(end[0]),   int(end[1])),   "t": 1.0, "offset": 0, "region": (1, 1)})


    return points

def spline_to_grid(points, value=7, num_samples=None):
    pts = sorted(points, key=lambda p: p["t"])

    t = np.array([p["t"] for p in pts], dtype=float)
    x = np.array([p["point"][0] for p in pts], dtype=float)
    y = np.array([p["point"][1] for p in pts], dtype=float)

    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)

    # --- Estimate arc length to guarantee dense enough sampling ---
    t_est = np.linspace(t.min(), t.max(), 2000)
    xs_est, ys_est = cs_x(t_est), cs_y(t_est)
    total_length = np.sum(np.sqrt(np.diff(xs_est)**2 + np.diff(ys_est)**2))

    if num_samples is None:
        num_samples = max(int(total_length * 3), 500)  # ≥3 samples per pixel

    t_smooth = np.linspace(t.min(), t.max(), num_samples)
    xs = cs_x(t_smooth)
    ys = cs_y(t_smooth)

    w, h = values.shape[:2]

    for i in range(len(xs) - 1):
        x0 = int(round(np.clip(xs[i],     0, w - 1)))
        y0 = int(round(np.clip(ys[i],     0, h - 1)))
        x1 = int(round(np.clip(xs[i + 1], 0, w - 1)))
        y1 = int(round(np.clip(ys[i + 1], 0, h - 1)))

        rr, cc = line(x0, y0, x1, y1)

        for xi, yi in zip(rr, cc):
            values[xi, yi, 0] = value  # no bounds check needed after clamp

def within_clearance(cx, cy, cz, block_type, clearance=1):
    region = values[
        max(0, cx-clearance):cx+clearance+1,
        max(0, cy-clearance):cy+clearance+1,
        max(0, cz-clearance):cz+clearance+1
    ]
    return (region == block_type).any()

def replace_within_clearance(cx, cy, cz, block_type, new_type, clearance=1):
    region = values[
        max(0, cx-clearance):cx+clearance+1,
        max(0, cy-clearance):cy+clearance+1,
        max(0, cz-clearance):cz+clearance+1
    ]

    region[region == block_type] = new_type


# TODO generate random points until end, maybe with changable varyation
# like can be a very winding path or points are more in a line shape, then connect points using cubic splines which will create smooth path

# def generate_random_path():


# uses number 1 as block indicator for terrain 
# uses number 2 as block indicator for trees 
# uses number 3 as block indicator for tree line
# uses number 4 as block indicator for end block
# uses number 5 as block indicator for drone
# uses number 6 as block indicator for possible path blocks
# uses number 7 as block indicator for path blocks
# uses number 8 as block indicator for drone surroundings
# added tree line so could stop pathfinder from going above trees 
def generate_terrain():
    for i in range(x):
        for j in range(y):
            terr_z = int(fractal_height(i, j, seed=seed))
            ground_level[i][j] = terr_z
            
            for k in range(terr_z + 1):
                values[i][j][k] = 1
            # if scale > 0.7:
            #     tree_line_z = int(fractal_height(i, j, scale=scale-scale*0.5));
            for k in range(tree_line_height + 1, z-terr_z):
                recede = tree_line_height + 1 + tree_line_recede
                
                if values[i][j][terr_z + k] == 0 and terr_z + k > recede:
                        
                    values[i][j][terr_z + k] = 3

def generate_trees(tree_chance=tree_density, tree_height=15, gen_trees=True):
    if gen_trees:
        for i in range(x):
            for j in range(y):
                terr_z = ground_level[i][j]
                
                tile_num = i * y + j
                spawn_tree(i, j, terr_z, tree_height, tree_chance, tile_num)

def spawn_tree(tx, ty, tz, tree_height, tree_chance, tile_num):
    random.seed(seed + tile_num)
    tree = random.random()
    if tree <= tree_chance:
        for i in range(1, tree_height + 1):
            values[tx][ty][tz + i] = 2

def show_path(path_history, grid_snapshot=None):
    # use a provided snapshot if available, otherwise fall back to the live grid
    # (live grid is usually dirty with lidar 8-markers and drone 5-markers from training)
    display_values = grid_snapshot if grid_snapshot is not None else values

    grid = pv.ImageData()
    grid.dimensions = np.array(display_values.shape) + 1
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)
    grid.cell_data["values"] = display_values.flatten(order="F")

    plotter = pv.Plotter()
    plotter.set_background([30, 30, 40])
    plotter.add_mesh(grid.threshold([0.5, 1.5], scalars="values"), color='#e07a5f', opacity=0.5)           # terrain
    plotter.add_mesh(grid.threshold([1.5, 2.5], scalars="values"), color='#432818')           # trees
    plotter.add_mesh(grid.threshold([2.5, 3.5], scalars="values"), color='#00b4d8', opacity=0.1)  # tree-line
    plotter.add_mesh(grid.threshold([3.5, 4.5], scalars="values"), color='#ff0000')           # end block
    plotter.add_mesh(grid.threshold([4.5, 5.5], scalars="values"), color='#00d20e')           # start/drone

    if len(path_history) > 1:
        pts = np.array(path_history, dtype=np.float32)
        spline = pv.Spline(pts, len(pts) * 10)
        plotter.add_mesh(spline, color='#00d20e', line_width=3)

    plotter.show()

def show_grid(drone=None):
    grid = pv.ImageData()
    grid.dimensions = np.array(values.shape) + 1
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)
    grid.cell_data["values"] = values.flatten(order="F")

    terrain      = grid.threshold([0.5, 1.5], scalars="values")
    trees        = grid.threshold([1.5, 2.5], scalars="values")
    tree_line    = grid.threshold([2.5, 3.5], scalars="values")
    end_block    = grid.threshold([3.5, 4.5], scalars="values")
    drone_block  = grid.threshold([4.5, 5.5], scalars="values")
    path         = grid.threshold([6.5, 7.5], scalars="values")

    plotter = pv.Plotter()
    plotter.set_background([30, 30, 40])
    plotter.add_mesh(terrain, show_edges=False, color='#e07a5f', opacity=0.5)
    plotter.add_mesh(trees, show_edges=False, color='#432818')
    plotter.add_mesh(tree_line, show_edges=False, color='#00b4d8', opacity=0.1)
    plotter.add_mesh(end_block, show_edges=False, color='#ff0000')
    plotter.add_mesh(drone_block, show_edges=False, color="#00d20e")
    plotter.add_mesh(path, show_edges=False, color='#ff7d00')

    drone_point = None
    if drone is not None:
        drone_point = pv.Cube(center=drone.pos, x_length=1, y_length=1, z_length=1)
        plotter.add_mesh(drone_point, color='#f15bb5')
        plotter.show(interactive_update=True)
    else:
        plotter.show()

    return plotter, drone_point

def main():
    generate_terrain()
    generate_trees()
    start, end = generate_start_and_end()
    replace_within_clearance(start[0], start[1], start[2], 2, 0)
    replace_within_clearance(end[0], end[1], end[2], 2, 0)
    points = generate_points(start, end, 20, 30, 80)
    spline_to_grid(points)
    print(f"Start: {start}, End: {end}")

    drone = Drone(id=0, sight_range=10, values=values, pos=start, goal=end)
    plotter, drone_point = show_grid(drone)
    assert drone_point is not None

    prev_pos = drone.pos.copy()

    def step(caller):
        direction = np.random.uniform(-1, 1, size=3)
        drone.move(direction, amount=1)
        drone_point.translate(drone.pos - prev_pos, inplace=True)
        prev_pos[:] = drone.pos

    plotter.add_timer_event(max_steps=1000, duration=300, callback=step)
    plotter.show()

if __name__ == "__main__":
    main()