import os                                       # have to do this so it works for me, idk why, delete if it messes with things
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

import numpy as np
from noise import pnoise2
import open3d as o3d
import random
from skimage.measure import marching_cubes

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
HEIGHT_VARIATION = 14   # how tall the hills get ABOVE the base (raise this for more dramatic peaks)


def generate_height_perlin(scale, octaves, persistence, lacunarity, seed):
    z = np.zeros((max_x, max_y))
    for i in range(max_x):
        for j in range(max_y):
            z[i, j] = pnoise2(i/scale, j/scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, repeatx=1024, repeaty=1024, base=seed)
    # Normalise to 0-1, then apply a power curve to flatten peaks before scaling.
    # Power > 1 pulls mid-range values down (flatter), power < 1 would exaggerate them.
    z = (z - z.min()) / (z.max() - z.min())
    z = z ** 1.8  # flatten: gentle undulations rather than sharp peaks
    # Terrain now sits on a raised floor (BASE_HEIGHT) with gentle hills on top
    # (HEIGHT_VARIATION). This is the "start higher, more bottom space" fix --
    # every column is guaranteed at least BASE_HEIGHT voxels of solid rock,
    # which is what gives the worms room to carve real tunnels instead of
    # poking straight through to bedrock or open air.
    z = BASE_HEIGHT + z * HEIGHT_VARIATION
    return z


def generate_voxels(scale=10.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=random.randint(0, 10000),
                     num_worms=6, worm_length=250, worm_radius=1.6):
    """Steps 1-3 only: build the raw boolean voxel grid (terrain + carved
    caves) without touching marching cubes at all. Split out from
    generate_terrain() so you can grab the voxels directly -- e.g. to view
    them as blocks instead of a mesh."""
    # ── 1. Heightmap ───────────────────────────────────────────────────────
    z_grid = generate_height_perlin(scale, octaves, persistence, lacunarity, seed)
    # z_grid shape: (max_x, max_y), values in [BASE_HEIGHT, BASE_HEIGHT + HEIGHT_VARIATION]

    # ── 2. Fill solid voxels below surface ─────────────────────────────────
    z_idx = np.arange(GRID_DEPTH)
    voxels = z_idx[np.newaxis, np.newaxis, :] <= z_grid[:, :, np.newaxis]
    # shape: (max_x, max_y, GRID_DEPTH), True = solid, False = air

    # ── 3. Worms carve caves directly into the voxel grid ──────────────────
    voxels = apply_perlin_worms(voxels, z_grid, num_worms=num_worms, worm_length=worm_length,
                                 radius=worm_radius, seed=seed)
    return voxels


def mesh_from_voxels(voxels):
    """Steps 4-6: marching cubes -> smooth, continuous TriangleMesh."""
    # ── 4. Marching cubes ──────────────────────────────────────────────────
    verts, faces, normals, _ = marching_cubes(
        voxels.astype(np.float32),
        level=0.5,          # isosurface between 0 (air) and 1 (solid)
        spacing=(1, 1, 1),  # voxel size
    )

    # ── 5. Color by height ─────────────────────────────────────────────────
    z_min, z_max = verts[:, 2].min(), verts[:, 2].max()
    norm_z = (verts[:, 2] - z_min) / (z_max - z_min + 1e-8)

    colors = np.zeros((len(verts), 3))
    colors[:, 0] = 0.345 + norm_z * 0.305
    colors[:, 1] = 0.506 + norm_z * 0.144
    colors[:, 2] = 0.341 + norm_z * 0.309

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
    solid_idx = np.argwhere(voxels)  # (N, 3) int array of solid (x, y, z) cells
    if len(solid_idx) == 0:
        raise ValueError("No solid voxels left to display (carved away everything?)")

    points = solid_idx.astype(np.float64) + 0.5  # voxel centers, so they snap into the right cell

    # Color by height, same gradient used for the mesh
    z = solid_idx[:, 2].astype(np.float64)
    norm_z = (z - z.min()) / (z.max() - z.min() + 1e-8)
    colors = np.zeros((len(points), 3))
    colors[:, 0] = 0.345 + norm_z * 0.305
    colors[:, 1] = 0.506 + norm_z * 0.144
    colors[:, 2] = 0.341 + norm_z * 0.309

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
    pad = np.pad(voxels, 1, mode='constant', constant_values=False)
    fully_buried = (
        pad[2:, 1:-1, 1:-1] & pad[:-2, 1:-1, 1:-1] &
        pad[1:-1, 2:, 1:-1] & pad[1:-1, :-2, 1:-1] &
        pad[1:-1, 1:-1, 2:] & pad[1:-1, 1:-1, :-2]
    )
    surface = voxels & ~fully_buried
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
            voxels[x0:x1, y0:y1, z0:z1][sphere] = False

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


if __name__ == "__main__":
    SHOW_AS = "m"   # "m" = smooth marching-cubes surface, "v" = blocky cubes

    # Generate the raw voxel grid once (terrain + carved caves)
    voxels = generate_voxels()

    if SHOW_AS == "m":
        geometries = [mesh_from_voxels(voxels)]
    else:
        geometries = [voxelgrid_from_voxels(voxels), voxel_edges_lineset(voxels)]

    # Render and visualize the terrain window
    print(f"Visualizing terrain as '{SHOW_AS}'... Close window to exit script.")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Open3D Terrain Visualization",
        width=1024,
        height=768,
        mesh_show_back_face=True,
    )