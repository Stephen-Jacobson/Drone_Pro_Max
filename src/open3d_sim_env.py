import os                                       # have to do this so it works for me, idk why, delete if it messes with things
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

import numpy as np
from noise import pnoise2
import open3d as o3d
import random
import time
from skimage.measure import marching_cubes
from drone import Drone

max_x, max_y = 30, 30

# NOTE: the old code reused the name `max_z` for two different things:
# (1) the height-scale for the Perlin heightmap, and (2) the vertical size
# of the voxel grid. Since Python globals are looked up at call-time (not
# def-time), the second assignment silently clobbered the first, so the
# "flatten the hills" setting never actually applied -- the heightmap ended
# up scaled across the *entire* 0-64 voxel column. That's why terrain looked
# broken and why there was barely any solid rock left for the worms to dig
# into. Fixed by splitting these into separate, clearly-named constants:

GRID_DEPTH = 64        # vertical resolution of the voxel grid (world height in voxels)
BASE_HEIGHT = 28        # guaranteed-solid floor thickness -> raise this for more underground room to carve caves in
HEIGHT_VARIATION = 4   # how tall the hills get ABOVE the base (raise this for more dramatic peaks)


def generate_height_perlin(scale, octaves, persistence, lacunarity, seed):
    z = np.zeros((max_x, max_y))
    for i in range(max_x):
        for j in range(max_y):
            z[i, j] = pnoise2(i/scale, j/scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, repeatx=1024, repeaty=1024, base=seed)
    # Normalise to 0-1, then apply a power curve to flatten peaks before scaling.
    # Power > 1 pulls mid-range values down (flatter), power < 1 would exaggerate them.
    z = (z - z.min()) / (z.max() - z.min())
    z = z ** 2.5  # flatten: gentle undulations rather than sharp peaks
    # Terrain now sits on a raised floor (BASE_HEIGHT) with gentle hills on top
    # (HEIGHT_VARIATION). This is the "start higher, more bottom space" fix --
    # every column is guaranteed at least BASE_HEIGHT voxels of solid rock,
    # which is what gives the worms room to carve real tunnels instead of
    # poking straight through to bedrock or open air.
    z = BASE_HEIGHT + z * HEIGHT_VARIATION
    return z

def spawn_drone(voxels, height_above=1, x=None, y=None, color=None):
    if x is None:
        start_x = random.randint(0, max_x)
    else:
        start_x = x
    if y is None:
        start_y = random.randint(0, max_y)
    else:
        start_y = y

    solid_idx = np.flatnonzero(voxels[start_x, start_y, :] != 0)
    if solid_idx.size == 0:
        raise ValueError(f"no solid ground under column ({start_x}, {start_y})")

    surface_idx = solid_idx.max()
    max_z = voxels.shape[2] - 1  # top of the voxel grid

    # desired height, then clamp into the valid range:
    #   floor  -> surface_idx + 1, so it can never sit inside/below the ground
    #   ceiling -> max_z, so it can never spawn outside the grid
    start_z = surface_idx + 1 + height_above
    start_z = min(max(start_z, surface_idx + 1), max_z)

    if color is None:
        color = np.array([1.0, 0.0, 0.0])  # default red
    else:
        color = np.array(color, dtype=np.float32)

    drone = Drone(id=0, sight_range=10, values=voxels, pos=(start_x, start_y, start_z), goal=(0, 0, 0), color=color)

    return drone






def generate_voxels(scale=30.0, octaves=6, persistence=0.6, lacunarity=1.0, seed=random.randint(0, 10000),
                     num_worms=6, worm_length=250, worm_radius=1.6):
    """Steps 1-3 only: build the raw integer-typed voxel grid (terrain +
    carved caves) where each voxel is a material ID (0 = air).

    This keeps compatibility with the rest of the script by returning a
    (max_x, max_y, GRID_DEPTH) array, but uses small integers to represent
    different materials (so you can add e.g. dirt/rock/water later).
    """
    # ── 1. Heightmap ───────────────────────────────────────────────────────
    z_grid = generate_height_perlin(scale, octaves, persistence, lacunarity, seed)
    # z_grid shape: (max_x, max_y), values in [BASE_HEIGHT, BASE_HEIGHT + HEIGHT_VARIATION]

    # ── 2. Fill solid voxels below surface with material IDs ───────────────
    z_idx = np.arange(GRID_DEPTH)
    solid_mask = z_idx[np.newaxis, np.newaxis, :] <= z_grid[:, :, np.newaxis]

    # Use dtype uint8 for compact material IDs: 0 = air, 1 = topsoil, 2 = rock
    voxels = np.zeros((max_x, max_y, GRID_DEPTH), dtype=np.uint8)
    # Simple material assignment: nearer the surface is "topsoil" (1),
    # deeper layers are "rock" (2). Adjust thresholds as desired.
    topsoil_thickness = 3
    # For each column, set voxels True where solid_mask, then assign material
    for i in range(max_x):
        for j in range(max_y):
            col = solid_mask[i, j]
            if not col.any():
                continue
            # highest solid index for this column (surface)
            solid_idxs = np.flatnonzero(col)
            if len(solid_idxs) == 0:
                continue
            surface_idx = solid_idxs.max()
            # mark all solid as rock first
            voxels[i, j, col] = 2
            # then overwrite the topmost layers as topsoil
            top0 = max(surface_idx - topsoil_thickness + 1, 0)
            top1 = surface_idx + 1  # slice end is exclusive
            voxels[i, j, top0:top1] = 1  # topsoil (near surface)

    # ── 3. Worms carve caves directly into the voxel grid ──────────────────
    # apply_perlin_worms will set carved voxels to 0 (air)
    # voxels = apply_perlin_worms(voxels, z_grid, num_worms=num_worms, worm_length=worm_length, radius=worm_radius, seed=seed)
    generate_random_crop_plots(voxels, 3, 3)
    return voxels

def generate_random_crop_plots(voxels, num_x, num_y, path_width=1, path_material=3):
    """Stamp a set of random straight path lines into the terrain.

    Each line is drawn as a thick 2D brush in the x/y plane and replaces the
    surface voxels under it with a new material ID (default 3 = dark brown).
    The `path_width` parameter controls how wide each path is in voxel cells.
    Values <= 1 now produce a single-voxel path; larger values make a square brush.
    """
    max_angle = 30  # degrees
    max_dim = max(max_x, max_y)
    margin = max(2, min(max_x, max_y) // 8)

    def stamp_path_at(x_idx, y_idx):
        # Use a square brush centered on the sampled point. Width 1/0 => one voxel.
        brush_size = max(1, int(path_width))
        x0 = x_idx - brush_size // 2
        y0 = y_idx - brush_size // 2
        for dx in range(brush_size):
            for dy in range(brush_size):
                xi = int(np.clip(x0 + dx, 0, max_x - 1))
                yi = int(np.clip(y0 + dy, 0, max_y - 1))
                if np.any(voxels[xi, yi, :] != 0):
                    solid_idxs = np.flatnonzero(voxels[xi, yi, :] != 0)
                    surface_idx = solid_idxs.max()
                    # Replace a shallow slab at the surface with the path material.
                    depth = min(3, surface_idx + 1)
                    voxels[xi, yi, max(0, surface_idx - depth + 1):surface_idx + 1] = path_material

    def draw_line(angle_deg, anchor_x, anchor_y):
        theta = np.radians(angle_deg)
        step_count = int(max_dim * 4)
        for t in np.linspace(-max_dim, max_dim, step_count):
            x_idx = int(round(anchor_x + t * np.cos(theta)))
            y_idx = int(round(anchor_y + t * np.sin(theta)))
            if 0 <= x_idx < max_x and 0 <= y_idx < max_y:
                stamp_path_at(x_idx, y_idx)

    # Create angled paths starting from random edges, not from the interior.
    for _ in range(num_x + num_y):
        side = random.choice(["left", "right", "top", "bottom"])
        if side == "left":
            anchor_x = margin
            anchor_y = random.randint(margin, max_y - 1 - margin)
            angle_deg = random.uniform(-max_angle, max_angle)
        elif side == "right":
            anchor_x = max_x - 1 - margin
            anchor_y = random.randint(margin, max_y - 1 - margin)
            angle_deg = random.uniform(180 - max_angle, 180 + max_angle)
        elif side == "top":
            anchor_x = random.randint(margin, max_x - 1 - margin)
            anchor_y = max_y - 1 - margin
            angle_deg = random.uniform(90 - max_angle, 90 + max_angle)
        else:  # bottom
            anchor_x = random.randint(margin, max_x - 1 - margin)
            anchor_y = margin
            angle_deg = random.uniform(-90 - max_angle, -90 + max_angle)
        draw_line(angle_deg, anchor_x, anchor_y)

    return voxels


def label_path_regions(voxels, path_material=3):
    """Label connected surface sections between path lines.

    Returns a 2D label map in x/y space where each non-path region gets its
    own integer label. Path cells are labeled 0.
    """
    path_mask = np.any(voxels == path_material, axis=2)
    labels = np.zeros((max_x, max_y), dtype=np.int32)
    next_label = 1
    for x in range(max_x):
        for y in range(max_y):
            if path_mask[x, y] or labels[x, y] != 0:
                continue
            # flood fill the current region
            stack = [(x, y)]
            labels[x, y] = next_label
            while stack:
                cx, cy = stack.pop()
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if 0 <= nx < max_x and 0 <= ny < max_y:
                        if not path_mask[nx, ny] and labels[nx, ny] == 0:
                            labels[nx, ny] = next_label
                            stack.append((nx, ny))
            next_label += 1
    return labels, next_label - 1


def fill_regions_top_materials(voxels, labels, region_materials=None, depth=1, prob4=0.8):
    """Fill the top `depth` solid voxels of each labeled region with a material.

    If `region_materials` (dict of region_id -> material_id) is given, those
    assignments are used directly. Otherwise, each region is randomly assigned
    material 4 (probability `prob4`) or material 5.
    """
    # No explicit mapping given -> invent one: roll a weighted coin per region
    # so each region consistently gets either material 4 or material 5.
    if region_materials is None:
        region_materials = {
            region_id: (4 if random.random() < prob4 else 5)
            for region_id in np.unique(labels)
            if region_id > 0  # skip background / invalid labels (0 or negative)
        }

    # Walk each region and paint its top surface with its assigned material.
    for region_id, material_id in region_materials.items():
        if region_id <= 0:
            continue  # guard against a bad id sneaking in via a manual dict

        # All (x, y) footprint cells that belong to this region.
        region_mask = labels == region_id

        for x, y in zip(*np.nonzero(region_mask)):
            # Look down this column of the voxel grid for solid (non-zero) cells.
            solid_idxs = np.flatnonzero(voxels[x, y, :] != 0)
            if solid_idxs.size == 0:
                continue  # empty column, nothing to paint

            # Highest solid voxel in the column = the "ground level" surface.
            surface_idx = solid_idxs.max()

            # Start of the slab we're recoloring, clamped so it can't go below z=0
            # (e.g. a column only 2 voxels deep with depth=5 just paints what's there).
            z0 = max(0, surface_idx - depth + 1)

            # Recolor the top `depth` voxels of this column with the new material,
            # leaving everything below untouched.
            voxels[x, y, z0:surface_idx + 1] = material_id

    return voxels


def mesh_from_voxels(voxels):
    """Steps 4-6: marching cubes -> smooth, continuous TriangleMesh."""
    # ── 4. Marching cubes on occupancy mask ─────────────────────────────────
    occ = (voxels != 0)
    verts, faces, normals, _ = marching_cubes(
        occ.astype(np.float32),
        level=0.5,          # isosurface between 0 (air) and 1 (solid)
        spacing=(1, 1, 1),  # voxel size
    )

    # ── 5. Color by material (sample material ID at each vertex) ──────────
    # Map material IDs to RGB colors. Extend this dict if you add more IDs.
    material_colors = {
        0: np.array([0.00, 0.00, 0.00]),  # air (unused)
        1: np.array([0.545, 0.415, 0.243]),  # topsoil / dirt
        2: np.array([0.45, 0.45, 0.45]),  # rock
        3: np.array([0.35, 0.20, 0.10]),  # dark brown path
        4: np.array([0.60, 0.80, 0.40]),  # light green region
        5: np.array([0.60, 0.25, 0.35]),  # light maroonish region
    }

    # Sample voxel material at the integer voxel under each vertex
    ix = np.clip(np.floor(verts[:, 0]).astype(int), 0, voxels.shape[0] - 1)
    iy = np.clip(np.floor(verts[:, 1]).astype(int), 0, voxels.shape[1] - 1)
    iz = np.clip(np.floor(verts[:, 2]).astype(int), 0, voxels.shape[2] - 1)
    mat = voxels[ix, iy, iz].astype(int)

    max_id = max(material_colors.keys())
    # build a lookup array for fast indexing; unknown IDs clamp to max known
    lut = np.zeros((max_id + 1, 3), dtype=np.float64)
    for k, v in material_colors.items():
        if k <= max_id:
            lut[k] = v
    mat_clamped = np.clip(mat, 0, max_id)
    colors = lut[mat_clamped]

    # ── 6. Build mesh ──────────────────────────────────────────────────────
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices      = o3d.utility.Vector3dVector(verts)
    mesh.triangles     = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    return mesh


def voxelgrid_from_voxels(voxels, voxel_size=1.0):
    """Turn the raw boolean voxel grid into an o3d.geometry.VoxelGrid --
    a blocky, Minecraft-style view with NO marching cubes / mesh involved.
    Each True voxel becomes one cube. Built via a colored point cloud
    (one point per solid voxel center) because VoxelGrid.create_from_point_cloud
    is vectorized in C++ -- looping add_voxel() in Python one voxel at a
    time would also work but is far slower for grids this size."""
    solid_idx = np.argwhere(voxels != 0)  # (N, 3) int array of solid (x, y, z) cells
    if len(solid_idx) == 0:
        raise ValueError("No solid voxels left to display (carved away everything?)")

    points = solid_idx.astype(np.float64) + 0.5  # voxel centers, so they snap into the right cell

    # Color by material using the same mapping as mesh_from_voxels
    material_colors = {
        1: np.array([0.545, 0.415, 0.243]),  # topsoil / dirt
        2: np.array([0.45, 0.45, 0.45]),  # rock
        3: np.array([0.35, 0.20, 0.10]),  # dark brown path
        4: np.array([0.60, 0.80, 0.40]),  # light green region : crops which should be sprayed
        5: np.array([0.60, 0.25, 0.35]),  # light maroonish region : crops which shouldnt
    }
    max_id = max(material_colors.keys())
    lut = np.zeros((max_id + 1, 3), dtype=np.float64)
    for k, v in material_colors.items():
        if k <= max_id:
            lut[k] = v

    mats = voxels[solid_idx[:, 0], solid_idx[:, 1], solid_idx[:, 2]].astype(int)
    mats_clamped = np.clip(mats, 0, max_id)
    colors = lut[mats_clamped]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)


def voxel_edges_lineset(voxels, line_color=(0, 0, 0)):
    """Build a LineSet outlining the edges of each voxel cube, so adjacent
    blocks get a visible boundary between them -- the classic Minecraft
    grid-line look. Drawn alongside the VoxelGrid (not instead of it).

    Only voxels with at least one exposed face are included: a voxel
    that's fully buried (all 6 neighbors solid) would never show its
    edges anyway since solid neighbors are drawn in front of them, so
    skipping those keeps the LineSet much smaller/faster."""
    occ = (voxels != 0)
    pad = np.pad(occ, 1, mode='constant', constant_values=False)
    fully_buried = (
        pad[2:, 1:-1, 1:-1] & pad[:-2, 1:-1, 1:-1] &
        pad[1:-1, 2:, 1:-1] & pad[1:-1, :-2, 1:-1] &
        pad[1:-1, 1:-1, 2:] & pad[1:-1, 1:-1, :-2]
    )
    surface = occ & ~fully_buried
    base = np.argwhere(surface)  # (N, 3) -- lower corner of each cube to outline
    if len(base) == 0:
        raise ValueError("No surface voxels to outline")

    # 8 corners of a unit cube, and the 12 edges connecting them by corner index
    corner_offsets = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                                [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    edge_pairs = np.array([[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
                            [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]])

    n = len(base)
    corners = (base[:, None, :] + corner_offsets[None, :, :]).reshape(-1, 3)        # (n*8, 3)
    edges   = (np.arange(n)[:, None, None] * 8 + edge_pairs[None, :, :]).reshape(-1, 2)  # (n*12, 2)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners.astype(np.float64))
    line_set.lines  = o3d.utility.Vector2iVector(edges)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(line_color, (len(edges), 1)))
    return line_set


def generate_terrain(scale=10.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=random.randint(0, 10000),
                      num_worms=6, worm_length=250, worm_radius=1.6):
    """Kept for backwards compatibility: same as before, returns a mesh."""
    voxels = generate_voxels(scale, octaves, persistence, lacunarity, seed,
                              num_worms, worm_length, worm_radius)
    return mesh_from_voxels(voxels)


def apply_perlin_worms(voxels, z_grid, num_worms=6, worm_length=250, radius=1.6, seed=0,
                        turn_h=0.15, turn_v=0.05):
    """Simple "drunk worm" cave carver: a point wanders through the solid
    terrain and a sphere of voxels is deleted around it at every step --
    exactly the "point moves, blocks disappear around it" idea. Two things
    were broken in the original version:

      1. Worms started at a fixed global z-range that often landed in open
         air for low-terrain columns, so nothing got carved.
      2. Worms `break`-ed the moment they touched the edge of the (small,
         30x30) map -- most worms died after ~15-25 of their 200 steps,
         carving almost nothing.

    Fixed by starting each worm relative to the *local* terrain height (so
    it always starts inside solid rock) and by having worms bounce off the
    map edges instead of dying, so every worm uses its full length.
    """
    rng = np.random.RandomState(seed)
    voxels = voxels.copy()
    margin = radius + 1.5  # keep the worm (and its carve-sphere) inside the grid

    for _ in range(num_worms):
        # Start somewhere inside solid rock, relative to the LOCAL terrain
        # height at that (x, y) -- not a fixed global range -- so the worm
        # never starts stranded in open air.
        x = rng.uniform(margin, max_x - margin)
        y = rng.uniform(margin, max_y - margin)
        local_h = max(z_grid[int(x), int(y)], margin * 2 + 1)
        z = rng.uniform(margin, max(local_h - margin, margin + 1))

        angle_h = rng.uniform(0, 2 * np.pi)   # horizontal heading
        angle_v = rng.uniform(-0.2, 0.2)       # vertical tilt
        r = radius

        for _ in range(worm_length):
            # Carve a sphere of `radius` voxels at current position
            ix, iy, iz = int(x), int(y), int(z)
            x0, x1 = max(0, int(ix - r)), min(max_x, int(ix + r + 1))
            y0, y1 = max(0, int(iy - r)), min(max_y, int(iy + r + 1))
            z0, z1 = max(0, int(iz - r)), min(GRID_DEPTH, int(iz + r + 1))

            cx, cy, cz = np.ogrid[x0:x1, y0:y1, z0:z1]
            sphere = (cx - x)**2 + (cy - y)**2 + (cz - z)**2 <= r**2
            # carve by setting material ID to 0 (air)
            sub = voxels[x0:x1, y0:y1, z0:z1]
            sub[sphere] = 0
            voxels[x0:x1, y0:y1, z0:z1] = sub

            # Nudge direction (small turns -> long winding tunnels rather
            # than tight round blobs; Perlin would go here too, random walk is fine)
            angle_h += rng.uniform(-turn_h, turn_h)
            angle_v += rng.uniform(-turn_v, turn_v)
            angle_v  = np.clip(angle_v, -0.4, 0.4)   # keep mostly horizontal

            nx = x + np.cos(angle_h) * np.cos(angle_v)
            ny = y + np.sin(angle_h) * np.cos(angle_v)
            nz = z + np.sin(angle_v)

            # Bounce off the boundary instead of dying, so the worm actually
            # uses its full length instead of stopping after a few steps.
            if nx <= margin or nx >= max_x - margin:
                angle_h = np.pi - angle_h
            if ny <= margin or ny >= max_y - margin:
                angle_h = -angle_h
            if nz <= margin or nz >= GRID_DEPTH - margin:
                angle_v = -angle_v

            x = np.clip(x + np.cos(angle_h) * np.cos(angle_v), margin, max_x - margin)
            y = np.clip(y + np.sin(angle_h) * np.cos(angle_v), margin, max_y - margin)
            z = np.clip(z + np.sin(angle_v), margin, GRID_DEPTH - margin)

    return voxels


def make_drone_mesh(drone, radius=0.4):
    """Build a small sphere mesh for the drone, colored from drone.get_color()
    and centered on the drone's true float position (+0.5 to match the same
    voxel-center offset the terrain/point-cloud rendering uses)."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh.translate(drone.get_position() + 0.5)
    mesh.paint_uniform_color(drone.get_color())
    mesh.compute_vertex_normals()
    return mesh


def run_simulation():
    SHOW_AS = "v"   # "m" = smooth marching-cubes surface, "v" = blocky cubes

    voxels = generate_voxels()
    labels, count = label_path_regions(voxels, path_material=3)
    fill_regions_top_materials(voxels, labels, prob4=0.8)

    # Spawn drone with custom color — this now actually shows up, since the
    # drone is its own mesh instead of a voxel material ID.
    drone = spawn_drone(voxels, color=[1.0, 0.2, 1.0])

    # Terrain is built ONCE and never touched again. The drone used to be
    # baked into this same geometry, which forced a full clear_geometries()
    # + rebuild every single frame — and clear_geometries() resets the
    # camera view, which is what made rotating feel impossible.
    if SHOW_AS == "m":
        terrain_geoms = [mesh_from_voxels(voxels)]
    else:
        terrain_geoms = [voxelgrid_from_voxels(voxels), voxel_edges_lineset(voxels)]

    drone_mesh = make_drone_mesh(drone)

    print(f"Visualizing terrain as '{SHOW_AS}' with drone movement...")
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Open3D Terrain Visualization", width=1024, height=768)

    for geom in terrain_geoms:
        vis.add_geometry(geom)
    vis.add_geometry(drone_mesh)

    state = {
        "paused": False,
        "last_move_time": time.time(),
        "move_count": 0,
        "last_drone_pos": drone.get_position().copy(),
    }

    def toggle_pause(vis_):
        state["paused"] = not state["paused"]
        print("-- paused --" if state["paused"] else "-- resumed --")
        return False  # False = no redraw needed just for the toggle

    # Spacebar pauses/resumes drone movement. You can freely rotate/zoom/pan
    # at any time (paused or not) — that's handled by Open3D's own mouse
    # input loop inside vis.run() and was never actually blocked by anything
    # except the clear_geometries() view-reset above.
    vis.register_key_callback(ord(" "), toggle_pause)

    move_interval = 0.3  # seconds between drone moves
    max_moves = 20

    def animation_callback(vis_):
        if state["paused"] or state["move_count"] >= max_moves:
            return False

        now = time.time()
        if now - state["last_move_time"] < move_interval:
            return False
        state["last_move_time"] = now

        direction = np.random.randn(3)
        drone.move(direction, amount=3)
        state["move_count"] += 1

        new_pos = drone.get_position()
        delta = new_pos - state["last_drone_pos"]   # true float-space delta
        drone_mesh.translate(delta)
        state["last_drone_pos"] = new_pos.copy()
        vis_.update_geometry(drone_mesh)

        print(f"Move {state['move_count']}/{max_moves}: drone at {new_pos}")
        return True  # tells Open3D a redraw is needed

    vis.register_animation_callback(animation_callback)

    print("Press SPACE to pause/resume drone movement. Rotate/zoom/pan anytime — close window to exit.")
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    run_simulation()