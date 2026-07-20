import numpy as np


def get_target_vector(pos, target, norm):
    """Same shape/contract as lidar.get_target_vector(pos, target, norm) in
    drone_env.py: returns (direction, distance) where `direction` is a unit
    vector pointing from `pos` to `target`, and `distance` is the raw
    distance divided by `norm` and clipped to [0, 1].
    """
    pos = np.asarray(pos, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    delta = target - pos
    raw_distance = float(np.linalg.norm(delta))

    if raw_distance == 0.0:
        direction = np.zeros_like(delta)
    else:
        direction = delta / raw_distance

    distance = float(np.clip(raw_distance / norm, 0.0, 1.0))
    return direction, distance


def pick_block(density_grid, pos, t):
    """Pick a block out of a density grid and return the direction +
    distance from `pos` to it, in the same (direction, distance) format
    drone_env.py's lidar.get_target_vector/get_goal_vector produce.

    NOTE: grid size is NOT fixed -- it's read from `density_grid.shape`
    on every call, so this works the same way whether it's handed the
    small downscaled coverage_map (e.g. 8x8) or the real full-resolution
    density map (e.g. 30x30). The normalising diagonal used for `distance`
    is recomputed from that shape each time, so `distance` is always
    relative to the grid actually passed in.

    Args:
        density_grid: HxW array-like of floats in [0, 1]. A cell with
            density 0 is not a valid pick; anything > 0 is.
        pos: (row, col) position of the object in the grid.
        t: float in [0, 1]. Valid blocks are ordered in matrix (row-major)
            order -- row 0 left to right, then row 1 left to right, etc.
            t=0.0 selects the first valid block in that order, t close to
            1.0 selects the last, and values in between interpolate across
            that ordering. This ordering is independent of `pos`, so it
            stays consistent as the drone moves.

    Returns:
        dict with:
            "block":     (row, col) of the picked block
            "direction": unit vector (2,) from pos to the block
            "distance":  normalised distance in [0, 1] (raw distance /
                         the picked grid's own diagonal)
            "raw_distance": un-normalised straight-line distance
        or None if there are no valid (nonzero) blocks in the grid.
    """
    density_grid = np.asarray(density_grid, dtype=np.float32)
    if density_grid.ndim != 2:
        raise ValueError(f"expected a 2D grid, got shape {density_grid.shape}")
    if not (0.0 <= t <= 1.0):
        raise ValueError("t must be between 0 and 1")

    grid_h, grid_w = density_grid.shape
    grid_diagonal = float(np.linalg.norm([grid_h - 1, grid_w - 1]))

    pos = np.asarray(pos, dtype=np.float32)

    valid_cells = [(r, c) for r in range(grid_h) for c in range(grid_w)
                   if density_grid[r, c] > 0.0]
    if not valid_cells:
        return None

    # valid_cells is already in matrix (row-major) order, since it was built
    # by iterating r then c in increasing order. This keeps the ordering
    # fixed regardless of pos, so t means the same thing no matter where
    # the drone currently is.
    index = min(int(t * len(valid_cells)), len(valid_cells) - 1)
    block = valid_cells[index]

    direction, distance = get_target_vector(pos, block, grid_diagonal)
    raw_distance = float(np.linalg.norm(np.array(block, dtype=np.float32) - pos))

    return {
        "block": block,
        "direction": direction,
        "distance": distance,
        "raw_distance": raw_distance,
    }


if __name__ == "__main__":
    GRID_SIZE = 30  # e.g. drone_env.py's real full-resolution coverage grid
    rng = np.random.default_rng(0)
    grid = rng.random((GRID_SIZE, GRID_SIZE))
    grid[grid < 0.6] = 0.0  # sparsify so plenty of cells are 0 (invalid)

    pos = (0, 0)
    for t in (0.0, 0.25, 0.5, 0.75, 0.999):
        result = pick_block(grid, pos, t)
        print(f"t={t:.3f} -> {result}")