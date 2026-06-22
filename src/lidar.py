import numpy as np

def mark_voxel(values, x, y, z, thickness=0):
    # numpy slice instead of triple Python loop — ~27x faster
    x0 = max(0, x - thickness);  x1 = min(values.shape[0], x + thickness + 1)
    y0 = max(0, y - thickness);  y1 = min(values.shape[1], y + thickness + 1)
    z0 = max(0, z - thickness);  z1 = min(values.shape[2], z + thickness + 1)
    slab = values[x0:x1, y0:y1, z0:z1]
    slab[~np.isin(slab, [1, 2, 4, 5])] = 8

def get_goal_vector(drone_pos, goal_pos, max_range):
    diff     = goal_pos - drone_pos                          # (3,)
    distance = np.linalg.norm(diff)                         # scalar
    direction = diff / (distance + 1e-8)                    # (3,) unit vector, avoid div/0
    return direction, distance / max_range                  # both normalised

def cast_all_rays(values, origin, directions, max_range, thickness=1):
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
    distances = np.full(N, max_range, dtype=np.float32)  # default = max_range (miss)

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
        distances[ib[hit]] = np.linalg.norm(xyz[ib[hit]] - origin, axis=1)  # record hit distance
        active[ib[hit]] = False

        # --- mark free voxels ---
        to_mark = ib[~hit & ~np.isin(vals, [4, 5])]
        for i in to_mark:                        # only unmarked voxels, not every ray
            mark_voxel(values, xyz[i, 0], xyz[i, 1], xyz[i, 2], thickness)

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

    return distances

def fibonacci_sphere_directions(n_rays, jitter=0.3):
    golden = np.pi * (3 - np.sqrt(5))
    i      = np.arange(n_rays, dtype=float)
    y      = np.clip(1 - ((i + np.random.uniform(-jitter, jitter, n_rays)) / (n_rays - 1)) * 2, -1, 1)
    r      = np.sqrt(1 - y * y)
    theta  = golden * i
    return np.column_stack([np.cos(theta) * r, y, np.sin(theta) * r])

def get_lidar_surroundings(values, org, max_range=10, n_rays=100):
    # print(f"Casting {n_rays} rays (vectorised)")
    directions = fibonacci_sphere_directions(n_rays)
    distances = cast_all_rays(values, org, directions, max_range, thickness=1)
    obs = distances / max_range

    return obs