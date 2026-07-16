import math
import random

import numpy as np
from subt_proc_gen.param_classes import (
    IntersectionPtClGenParams,
    TunnelPtClGenParams,
)
from subt_proc_gen.tunnel import (
    ConnectorTunnelGenerationParams,
    GrownTunnelGenerationParams,
    TunnelNetwork,
    TunnelNetworkParams,
)


def carve_sphere(values, cx, cy, cz, radius, cave_val):
    gx, gy, gz = values.shape
    x0, x1 = max(0, int(cx - radius)), min(gx, int(cx + radius + 1))
    y0, y1 = max(0, int(cy - radius)), min(gy, int(cy + radius + 1))
    z0, z1 = max(0, int(cz - radius)), min(gz, int(cz + radius + 1))
    for ix in range(x0, x1):
        for iy in range(y0, y1):
            for iz in range(z0, z1):
                if (ix - cx) ** 2 + (iy - cy) ** 2 + (iz - cz) ** 2 <= radius**2:
                    if values[ix, iy, iz] != 0:
                        values[ix, iy, iz] = cave_val


def gen_caves(values, ground_lvl, seed, cave_val=0, **kwargs):
    gx, gy, gz = values.shape
    y_offset = gy // 2

    rng = random.Random(seed)

    tunnel_network_params = TunnelNetworkParams(
        collision_distance=6,
        max_inclination_rad=np.deg2rad(15),
        min_intersection_angle_rad=np.deg2rad(30),
        min_distance_between_intersections=20,
        flat=False,
    )
    tunnel_network = TunnelNetwork(params=tunnel_network_params, initial_node=False)

    n_tunnels = rng.randint(1, 3)

    for _ in range(n_tunnels):
        length = rng.uniform(80, 180)
        h_noise = rng.uniform(2, 6)
        v_noise = rng.uniform(1, 4)

        params = GrownTunnelGenerationParams(
            distance=length,
            horizontal_tendency_rad=np.deg2rad(rng.uniform(-30, 30)),
            vertical_tendency_rad=np.deg2rad(rng.uniform(-5, 5)),
            horizontal_noise_rad=np.deg2rad(h_noise),
            vertical_noise_rad=np.deg2rad(v_noise),
            min_segment_length=rng.uniform(5, 10),
            max_segment_length=rng.uniform(15, 25),
        )
        success, tunnel = tunnel_network.add_random_grown_tunnel(
            params=params,
            n_trials=50,
            yaw_range=(np.deg2rad(-60), np.deg2rad(60)),
        )
        if not success:
            continue

    if len(tunnel_network.tunnels) == 0:
        return values, None

    all_mouths = []

    for tunnel in tunnel_network.tunnels:
        spline = tunnel.spline
        _, aps, _ = spline.discretize(0.5)
        positions = np.array(aps)
        offset = np.array([gx * 0.3 + seed % 100, y_offset, 0])
        positions += offset

        base_radius = rng.uniform(3.0, 6.0)
        n_pos = len(positions)
        mouth_found = False

        for i, pos in enumerate(positions):
            ix, iy = int(round(pos[0])), int(round(pos[1]))
            if not (0 <= ix < gx and 0 <= iy < gy):
                continue

            local_ground = int(ground_lvl[ix, iy])
            if local_ground < 1:
                continue

            t = i / max(n_pos - 1, 1)
            local_radius = (
                base_radius + math.sin(t * math.pi * 2) * 1.0 + rng.uniform(-0.3, 0.3)
            )
            local_radius = max(2.5, min(8.0, local_radius))

            cz = local_ground - int(local_radius) - 1
            if cz < 0:
                continue

            carve_sphere(values, pos[0], pos[1], cz, local_radius, cave_val)

            if not mouth_found:
                mouth_found = True
                all_mouths.append((ix, iy, cz))

    cave_mask = (values == 0) & (
        np.arange(gz, dtype=np.int16)[None, None, :] < ground_lvl[:, :, None]
    )
    coords = np.argwhere(cave_mask)
    if len(coords) == 0:
        return values, None

    if all_mouths:
        return values, all_mouths[0]

    return values, None
