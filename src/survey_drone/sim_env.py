import os

os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

import time
import open3d as o3d
import numpy as np
from noise import pnoise2
import random
from scipy.interpolate import CubicSpline
from skimage.draw import line

import pyvista as pv

from drone import Drone
import cave_gen

seed = random.randint(0, 10000)


def reseed(new_seed=None):
    """Re-roll the terrain/tree RNG seed. Call this before generate_terrain()/
    generate_trees() whenever you want a genuinely NEW map — without this,
    seed never changes after module import, so every "regenerated" world
    was actually bit-identical (same heights, same trees, same start/end)."""
    global seed
    seed = new_seed if new_seed is not None else random.randint(0, 10000)


x, y, z = 300, 300, 200


def set_dim(nx, ny, nz=100):
    x = nx
    y = ny
    z = nz


scale = 0.01
values = np.zeros(
    (x, y, z), dtype=np.uint8
)  # smallest dtype so that takes least amount of memory - 0-255
ground_level = np.zeros(
    (x, y), dtype=np.uint16
)  # bigger but still small, must hold ground levl values - 0-65535

tree_line_height = 15
tree_line_recede = 0
tree_density = 0.01
# Ion evven know what the rigt value is supposed to be here tbh
cave_density = 0.05
cave_room_size = 1.5

# gets height per x y spot on grid to give a bumpy terrain, from gpt, for bumpier, increase octaves, persistence, amplitude, for less bumpy, decrease octaves, persistence eg 0.3, lower scale
# Scale: up - denser bumps, down - wide smooth hills
# Persistence: up - rough/jagged, down - smooth


def fractal_height(
    x, y, seed=seed, scale=scale, octaves=5, persistence=0.01, amplitude=50
):
    height = 0
    frequency = 1
    amp = 1

    max_amp = 0  # normalization factor

    for i in range(octaves):
        n = pnoise2((x + seed) * scale * frequency, (y + seed) * scale * frequency)
        n = 1 - abs(n)  # uncommenting will make look more duney
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
            (cx, cy)
            for cx, cy in all_candidates
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

    raise RuntimeError(
        "Could not find a valid start/end pair meeting the minimum distance"
    )


def valid_z_values(cx, cy):
    valid_levels = list(
        range(
            ground_level[cx][cy] + 1,
            ground_level[cx][cy] + tree_line_height + tree_line_recede,
        )
    )  # valid range of z values based on what will be open

    return valid_levels


def generate_points(
    start, end, n_points, min_distance=0.0, max_distance=1.0, overlap=0.2
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

        points.append(
            {"point": (x, y), "t": t, "offset": offset_mag, "region": (t_min, t_max)}
        )

    points.insert(
        0,
        {
            "point": (int(start[0]), int(start[1])),
            "t": 0.0,
            "offset": 0,
            "region": (0, 0),
        },
    )
    points.append(
        {"point": (int(end[0]), int(end[1])), "t": 1.0, "offset": 0, "region": (1, 1)}
    )

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
    total_length = np.sum(np.sqrt(np.diff(xs_est) ** 2 + np.diff(ys_est) ** 2))

    if num_samples is None:
        num_samples = max(int(total_length * 3), 500)  # ≥3 samples per pixel

    t_smooth = np.linspace(t.min(), t.max(), num_samples)
    xs = cs_x(t_smooth)
    ys = cs_y(t_smooth)

    w, h = values.shape[:2]

    for i in range(len(xs) - 1):
        x0 = int(round(np.clip(xs[i], 0, w - 1)))
        y0 = int(round(np.clip(ys[i], 0, h - 1)))
        x1 = int(round(np.clip(xs[i + 1], 0, w - 1)))
        y1 = int(round(np.clip(ys[i + 1], 0, h - 1)))

        rr, cc = line(x0, y0, x1, y1)

        for xi, yi in zip(rr, cc):
            values[xi, yi, 0] = value  # no bounds check needed after clamp


def within_clearance(cx, cy, cz, block_type, clearance=1):
    region = values[
        max(0, cx - clearance) : cx + clearance + 1,
        max(0, cy - clearance) : cy + clearance + 1,
        max(0, cz - clearance) : cz + clearance + 1,
    ]
    return (region == block_type).any()


def replace_within_clearance(cx, cy, cz, block_type, new_type, clearance=1):
    region = values[
        max(0, cx - clearance) : cx + clearance + 1,
        max(0, cy - clearance) : cy + clearance + 1,
        max(0, cz - clearance) : cz + clearance + 1,
    ]

    region[region == block_type] = new_type


# TODO generate random points until end, maybe with changable varyation
# like can be a very winding path or points are more in a line shape, then connect points using cubic splines which will create smooth path
#
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
            # for k in range(tree_line_height + 1, z - terr_z):
            #     recede = tree_line_height + 1 + tree_line_recede

            #     if values[i][j][terr_z + k] == 0 and terr_z + k > recede:
            #         values[i][j][terr_z + k] = 3

    # NOTE: cave_gen entry point
#    cave_gen.gen_caves(
#        values, ground_level, seed, density=cave_density, room_sz=cave_room_size)


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


BLOCK_COLORS = {
    1: [0.878, 0.478, 0.373],  # terrain   — #e07a5f
    2: [0.263, 0.157, 0.094],  # trees     — #432818
    3: [0.0, 0.706, 0.847],  # tree-line — #00b4d8
    4: [1.0, 0.0, 0.0],  # end block — #ff0000
    5: [0.0, 0.824, 0.055],  # start     — #00d20e
    7: [1.0, 0.490, 0.0],  # path      — #ff7d00
}


def _voxel_grid(values, block_type, color):
    dense = values == block_type
    if not dense.any():
        return None
    coords = np.argwhere(dense).astype(np.float64) + 0.5
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(coords), 1)))
    return o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.0)


def _camera(vis, lookat=None):
    ctrl = vis.get_view_control()
    if lookat is None:
        lookat = [x / 2, y / 2, z / 2]
    ctrl.set_front([-0.3, -0.5, -0.8])
    ctrl.set_up([0, 1, 0])
    ctrl.set_lookat(lookat)
    ctrl.set_zoom(0.6)
    vis.get_render_option().background_color = np.array([30, 30, 40]) / 255.0


def _find_cave_positions():
    z_idx = np.arange(z, dtype=np.int16)
    below_ground = z_idx[None, None, :] < ground_level[:, :, None]
    cave_mask = (values == 0) & below_ground
    coords = np.argwhere(cave_mask)
    if len(coords) == 0:
        return []
    return [tuple(c) for c in coords]


def show_path(path_history, grid_snapshot=None):
    display_values = grid_snapshot if grid_snapshot is not None else values

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for t, c in BLOCK_COLORS.items():
        if t == 7:
            continue
        vg = _voxel_grid(display_values, t, c)
        if vg is not None:
            vis.add_geometry(vg)

    if len(path_history) > 1:
        pts = np.array(path_history, dtype=np.float64)
        lines = [[i, i + 1] for i in range(len(pts) - 1)]
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector([[0.0, 0.824, 0.055]] * len(lines))
        vis.add_geometry(ls)

    _camera(vis)
    vis.run()
    vis.destroy_window()


def _make_terrain_mesh():
    solid = np.isin(values, [1, 2, 3]).astype(np.uint8)
    grid = pv.ImageData()
    grid.dimensions = np.array(solid.shape) + 1
    grid.origin = (0, 0, 0)
    grid.spacing = (1, 1, 1)
    grid.cell_data["solid"] = solid.ravel(order="F")
    grid_pd = grid.cell_data_to_point_data()
    return grid_pd.contour([0.5], scalars="solid", compute_normals=True)


def show_grid(drone=None):
    pl = pv.Plotter()

    terrain = _make_terrain_mesh()
    pl.add_mesh(terrain, color="#e07a5f", smooth_shading=True)

    drone_mesh = None
    if drone is not None:
        drone_mesh = pv.Cube(center=drone.pos, x_length=1, y_length=1, z_length=1)
        pl.add_mesh(drone_mesh, color="#f011b6")

        start_pos = np.argwhere(values == 5)
        if len(start_pos):
            s = pv.Sphere(center=start_pos[0].astype(float), radius=1.2)
            pl.add_mesh(s, color="#00d20e")

        end_pos = np.argwhere(values == 4)
        if len(end_pos):
            e = pv.Sphere(center=end_pos[0].astype(float), radius=1.2)
            pl.add_mesh(e, color="#ff0000")

        lookat = np.array(drone.pos, dtype=float)
        pl.camera_position = [
            lookat + [50, -30, 40],
            lookat,
            [0, 0, 1],
        ]

    if drone is None:
        pl.show()
        return None, None

    return pl, drone_mesh


def main():
    generate_terrain()
    #generate_trees()
    cave_gen.gen_caves(                                                           
        values, 
        ground_level, 
        seed, 
        density=cave_density, 
        room_sz=cave_room_size
    )

    cave_positions = _find_cave_positions()

    if len(cave_positions) >= 2:
        random.shuffle(cave_positions)
        start = cave_positions[0]
        dists = [np.linalg.norm(np.array(p) - np.array(start)) for p in cave_positions]
        end = cave_positions[int(np.argmax(dists))]
    else:
        print("WARNING: no tunnel voxels found, falling back to surface start/end")
        start, end = generate_start_and_end()

    replace_within_clearance(start[0], start[1], start[2], 2, 0)
    replace_within_clearance(end[0], end[1], end[2], 2, 0)
    print(f"Start: {start}, End: {end}")

    drone = Drone(id=0, sight_range=10, values=values, pos=start, goal=end)
    pl, drone_mesh = show_grid(drone)
    assert drone_mesh is not None

    prev_pos = drone.pos.copy()
    step_count = [0]

    pl.show(interactive_update=True)
    while step_count[0] < 1000:
        direction = np.random.uniform(-1, 1, size=3)
        drone.move(direction, amount=1)
        drone_mesh.translate(drone.pos - prev_pos, inplace=True)
        prev_pos[:] = drone.pos
        step_count[0] += 1
        pl.update()
        time.sleep(0.03)

    pl.close()


if __name__ == "__main__":
    main()
