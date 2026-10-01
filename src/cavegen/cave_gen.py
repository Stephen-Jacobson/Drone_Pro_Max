import numpy as np
from subt_proc_gen.tunnel import (
    TunnelNetwork,
    TunnelNetworkParams,
    GrownTunnelGenerationParams,
    ConnectorTunnelGenerationParams,
)
from subt_proc_gen.param_classes import (
    TunnelPtClGenParams,
    IntersectionPtClGenParams,
)
import random


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


def carve_along_spline(values, spline, radius, cave_val, step=0.5):
    _, aps, _ = spline.discretize(step)
    for ap in aps:
        carve_sphere(values, ap[0], ap[1], ap[2], radius, cave_val)


def gen_caves(values, ground_lvl, seed, cave_val=0, **kwargs):
    gx, gy, gz = values.shape
    y_offset = gy // 2
    z_offset = 2

    rng = random.Random(seed)

    tunnel_network_params = TunnelNetworkParams(
        collision_distance = 6,
        max_inclination_rad = np.deg2rad(30),
        min_intersection_angle_rad = np.deg2rad(30),
        min_distance_between_intersections = 20,
        flat = False,
    )
    tunnel_network = TunnelNetwork(params=tunnel_network_params, initial_node=False)

    n_tunnels = rng.randint(2, 4)
    n_connectors = rng.randint(0, 2)

    for _ in range(n_tunnels):
        length = rng.uniform(400, 400)
        h_tend = rng.uniform(-30, 30)
        v_tend = rng.uniform(-10, 10)
        h_noise = rng.uniform(5, 2)
        v_noise = rng.uniform(2, 1)
        min_seg = rng.uniform(3, 2)
        max_seg = rng.uniform(10, 20)

        params = GrownTunnelGenerationParams(
            distance = length,
            horizontal_tendency_rad = np.deg2rad(h_tend),
            vertical_tendency_rad = np.deg2rad(v_tend),
            horizontal_noise_rad = np.deg2rad(h_noise),
            vertical_noise_rad = np.deg2rad(v_noise),
            min_segment_length = min_seg,
            max_segment_length = max_seg,
        )
        success, tunnel = tunnel_network.add_random_grown_tunnel(
            params=params,
            n_trials=50,
            yaw_range=(0, 2 * np.pi),
        )

        if not success:
            continue

    for _ in range(n_connectors):
        if len(tunnel_network.tunnels) < 2:
            break
        conn_params = ConnectorTunnelGenerationParams(
            segment_length=rng.uniform(8, 15),
            node_position_horizontal_noise=rng.uniform(0, 3),
            node_position_vertical_noise=rng.uniform(0, 2),
        )
        success, _ = tunnel_network.add_random_connector_tunnel(
            params=conn_params,
            n_trials=50,
        )

    if len(tunnel_network.tunnels) == 0:
        return values

    for tunnel in tunnel_network.tunnels:
        radius = rng.uniform(3.0, 6.0)
        terrain_z = (
            int(np.mean(ground_lvl[ground_lvl > 0]))
            if np.any(ground_lvl > 0)
            else gz // 2
        )

        spline = tunnel.spline
        _, aps, _ = spline.discretize(0.5)

        positions = np.array(aps)
        offset = np.array([gx * 0.3 + seed % 100, y_offset, terrain_z - z_offset])
        positions += offset

        for pos in positions:
            ix, iy = int(round(pos[0])),int(round(pos[1]))
            if 0 <= ix < gx and 0 <= iy < gy:
                local_ground = ground_lvl[ix, iy]
                cz = local_ground - radius - 1 # put the entire sphere below surface
                carve_sphere(values, pos[0], pos[1], cz, radius, cave_val)

    return values
