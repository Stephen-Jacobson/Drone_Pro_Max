import pyvista as pv
import numpy as np
from noise import pnoise2
import random
from scipy.interpolate import CubicSpline
from skimage.draw import line

pv.global_theme.allow_empty_mesh = True

seed = random.randint(0, 10000)

x, y, z = 10, 10, 100
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
    edge_band = max(1, int(min(x, y) * 0.2))
    minimum_span = int(min(x, y) * 0.6)

    def _tree_clear(cx, cy, cz, clearance=1):
        x_min = max(0, cx - clearance)
        x_max = min(x - 1, cx + clearance)
        y_min = max(0, cy - clearance)
        y_max = min(y - 1, cy + clearance)
        z_min = max(0, cz - clearance)
        z_max = min(z - 1, cz + clearance)
        for nx in range(x_min, x_max + 1):
            for ny in range(y_min, y_max + 1):
                for nz in range(z_min, z_max + 1):
                    if values[nx][ny][nz] == 2:
                        return False
        return True

    def _collect_points(x_range, y_range):
        points = []
        for cx in x_range:
            for cy in y_range:
                z_levels = valid_z_values(cx, cy)
                if z_levels:
                    points.append((cx, cy, random.choice(z_levels)))
        return points

    left_points = _collect_points(range(0, edge_band), range(0, y))
    right_points = _collect_points(range(x - edge_band, x), range(0, y))
    bottom_points = _collect_points(range(0, x), range(0, edge_band))
    top_points = _collect_points(range(0, x), range(y - edge_band, y))

    pair_options = [
        (left_points, right_points),
        (bottom_points, top_points),
    ]
    random.shuffle(pair_options)

    for side_a, side_b in pair_options:
        if not side_a or not side_b:
            continue

        start = random.choice(side_a)
        end = max(
            side_b,
            key=lambda p: (p[0] - start[0]) ** 2 + (p[1] - start[1]) ** 2,
        )

        span = ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) ** 0.5
        if span >= minimum_span:
            # Mark positions in grid: 5 for drone/start, 4 for end
            values[start[0]][start[1]][start[2]] = 5
            values[end[0]][end[1]][end[2]] = 4
            return start, end

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
            terr_z = int(fractal_height(i, j))
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

def mark_voxel(x, y, z, thickness=0):
    # numpy slice instead of triple Python loop — ~27x faster
    x0 = max(0, x - thickness);  x1 = min(values.shape[0], x + thickness + 1)
    y0 = max(0, y - thickness);  y1 = min(values.shape[1], y + thickness + 1)
    z0 = max(0, z - thickness);  z1 = min(values.shape[2], z + thickness + 1)
    slab = values[x0:x1, y0:y1, z0:z1]
    slab[~np.isin(slab, [1, 2, 4, 5])] = 8

def cast_all_rays(origin, directions, max_range, thickness=1):
    """March ALL rays together each step instead of one at a time."""
    N = len(directions)
    d = directions / np.linalg.norm(directions, axis=1, keepdims=True)  # (N,3)

    with np.errstate(divide='ignore', invalid='ignore'):  # silences the warnings
        step  = np.sign(d).astype(int)                                      # (N,3)
        delta = np.where(np.abs(d) < 1e-9, np.inf, np.abs(1.0 / d))       # (N,3)
        tmax  = np.where(
            np.abs(d) < 1e-9,
            np.inf,
            ((np.floor(origin) + (step > 0)) - origin) / d                 # (N,3)
        )

    xyz    = np.tile(np.floor(origin).astype(int), (N, 1))  # (N,3) — all start positions
    active = np.ones(N, dtype=bool)

    max_steps = int(max_range * np.sqrt(3)) + 10  # guaranteed to cover full range

    for _ in range(max_steps):
        if not active.any():
            break

        ai = np.where(active)[0]
        x, y, z = xyz[ai, 0], xyz[ai, 1], xyz[ai, 2]

        # --- bounds check (vectorised) ---
        in_bounds = (
            (x >= 0) & (x < values.shape[0]) &
            (y >= 0) & (y < values.shape[1]) &
            (z >= 0) & (z < values.shape[2])
        )
        active[ai[~in_bounds]] = False

        ib = ai[in_bounds]
        if not len(ib):
            break

        xb, yb, zb = xyz[ib, 0], xyz[ib, 1], xyz[ib, 2]
        vals = values[xb, yb, zb]

        # --- stop rays that hit solid ---
        hit = np.isin(vals, [1, 2])
        active[ib[hit]] = False

        # --- mark free voxels ---
        to_mark = ib[~hit & ~np.isin(vals, [4, 5])]
        for i in to_mark:                        # only unmarked voxels, not every ray
            mark_voxel(xyz[i, 0], xyz[i, 1], xyz[i, 2], thickness)

        # --- advance all still-active rays ---
        still = np.where(active)[0]
        if not len(still):
            break

        ax     = np.argmin(tmax[still], axis=1)  # which axis to step each ray
        t_next = tmax[still, ax]

        # deactivate rays that have exceeded max_range
        done = t_next >= max_range
        active[still[done]] = False

        go    = still[~done]
        ax_go = ax[~done]
        tmax[go, ax_go] += delta[go, ax_go]
        xyz [go, ax_go] += step [go, ax_go]

def fibonacci_sphere_directions(n_rays, jitter=0.3):
    golden = np.pi * (3 - np.sqrt(5))
    i      = np.arange(n_rays, dtype=float)
    y      = np.clip(1 - ((i + np.random.uniform(-jitter, jitter, n_rays)) / (n_rays - 1)) * 2, -1, 1)
    r      = np.sqrt(1 - y * y)
    theta  = golden * i
    return np.column_stack([np.cos(theta) * r, y, np.sin(theta) * r])

def get_lidar_surroundings(org, max_range=10, n_rays=300):
    print(f"Casting {n_rays} rays (vectorised)")
    directions = fibonacci_sphere_directions(n_rays)
    cast_all_rays(org, directions, max_range, thickness=1)

def show_grid():
    grid = pv.ImageData()
    grid.dimensions = np.array(values.shape) + 1
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)

    grid.cell_data["values"] = values.flatten(order="F")

    terrain = grid.threshold([0.5, 1.5], scalars="values")
    trees = grid.threshold([1.5, 2.5], scalars="values")
    tree_line = grid.threshold([2.5, 3.5], scalars="values")
    end_block = grid.threshold([3.5, 4.5], scalars="values")
    drone_block = grid.threshold([4.5, 5.5], scalars="values")
    path_points = grid.threshold([5.5, 6.5], scalars="values")
    path = grid.threshold([6.5, 7.5], scalars="values")
    surroundings = grid.threshold([7.5, 8.5], scalars="values")
    
    plotter = pv.Plotter()
    plotter.set_background([30, 30, 40])

    plotter.add_mesh(terrain, show_edges=False, color='#e07a5f')
    plotter.add_mesh(trees, show_edges=False, color='#432818')
    plotter.add_mesh(tree_line, show_edges=False, color='#00b4d8', opacity=0.1)
    plotter.add_mesh(end_block, show_edges=False, color='#ff0000')
    plotter.add_mesh(drone_block, show_edges=False, color="#00d20e")
    plotter.add_mesh(surroundings, show_edges=False, color="#ffb703", opacity=0.4)
    # plotter.add_mesh(path_points, show_edges=False, color="#f2542d")
    # plotter.add_mesh(path, show_edges=False, color="#ff7d00")
    plotter.show()

def main():
    generate_terrain()
    generate_trees()
    start, end = generate_start_and_end()
    replace_within_clearance(start[0], start[1], start[2], 2, 0)
    replace_within_clearance(end[0], end[1], end[2], 2, 0)
    get_lidar_surroundings(start)
    points = generate_points(start, end, 20, 30, 80)
    spline_to_grid(points)
    print(f"Start: {start}, End: {end}")
    show_grid()

if __name__ == "__main__":
    main()

