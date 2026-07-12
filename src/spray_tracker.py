import numpy as np


class SprayTracker(object):
    """Tracks where the drone's spray rays have landed.

    Keeps a 2D (x, y) grid over the top of the terrain counting how many
    times each column has been hit by a spray ray, plus per-region
    "how much of this region has been watered" completeness, using the
    region labels/materials produced by label_path_regions() and
    fill_regions_top_materials() in open3d_sim_env.py.

    This class only tracks state -- it does not compute any RL reward.
    That's left for later; region_completeness()/needs_water_completeness()
    are the numbers a reward function would presumably read from.
    """

    def __init__(self, voxels, labels, region_materials, n_rays, water_material=4, no_water_material=5,
                 required_hits_per_cell=None):
        self.shape = voxels.shape[:2]

        # How many spray-ray hits each (x, y) column has ever received.
        self.hit_counts = np.zeros(self.shape, dtype=np.int32)

        self.labels = labels.copy()
        self.region_materials = dict(region_materials)

        self.n_rays = n_rays

        # Number of spray-ray hits required for a single (x, y) cell/column to
        # be considered watered. Defaults to one-fifth of the spray cone size,
        # which keeps the old behavior but makes the threshold explicit.
        if required_hits_per_cell is None:
            required_hits_per_cell = max(1, int(np.ceil(float(n_rays) / 5.0)))
        self.required_hits_per_cell = int(required_hits_per_cell)

        # Which regions actually need watering vs. shouldn't be sprayed,
        # based on the material fill_regions_top_materials() assigned them.
        self.needs_water_regions = {rid for rid, m in self.region_materials.items()
                                     if m == water_material}
        self.no_water_regions = {rid for rid, m in self.region_materials.items()
                                  if m == no_water_material}

        # Cache each region's total cell count once (doesn't change over time).
        self._region_total_cells = {}
        for region_id in np.unique(self.labels):
            if region_id <= 0:
                continue
            self._region_total_cells[region_id] = int((self.labels == region_id).sum())

    def register_hits(self, hit_xyz):
        """Record spray hits. `hit_xyz` is the (K, 3) array returned by
        Drone.spray() / lidar.cast_spray_rays() -- only the (x, y) columns
        are used here, the z (depth of the hit) doesn't matter for coverage."""
        hit_xyz = np.asarray(hit_xyz)
        if hit_xyz.size == 0:
            return
        xs, ys = hit_xyz[:, 0], hit_xyz[:, 1]
        np.add.at(self.hit_counts, (xs, ys), 1)

    def region_cell_counts(self):
        """dict region_id -> (total_cells, watered_cells), where a cell
        counts as watered once it has reached the configured hit threshold."""
        watered_mask = self.hit_counts >= self.required_hits_per_cell
        out = {}
        for region_id, total in self._region_total_cells.items():
            watered = int((watered_mask & (self.labels == region_id)).sum())
            out[region_id] = (total, watered)
        return out

    def region_completeness(self):
        """dict region_id -> fraction in [0, 1] of that region's cells that
        have reached the configured spray-hit threshold. Computed for every
        region, whether or not it actually needs watering."""
        return {rid: (watered / total if total > 0 else 0.0)
                for rid, (total, watered) in self.region_cell_counts().items()}

    def needs_water_completeness(self):
        """Completeness restricted to regions that need watering (material
        4). This is the coverage number most relevant to a future reward."""
        comp = self.region_completeness()
        return {rid: comp[rid] for rid in self.needs_water_regions if rid in comp}

    def needs_water_total_completeness(self):
        """Single area-weighted fraction in [0, 1]: total watered cells
        across ALL needs-water regions combined, divided by total cells
        across those same regions.

        This differs from averaging needs_water_completeness()'s per-region
        fractions -- that unweighted mean treats a tiny region and a huge
        region as equally important. This version instead reflects "what
        fraction of the actual needs-water ground has been sprayed",
        matching what the path/coverage picture visually suggests."""
        counts = self.region_cell_counts()
        total = sum(t for rid, (t, w) in counts.items() if rid in self.needs_water_regions)
        watered = sum(w for rid, (t, w) in counts.items() if rid in self.needs_water_regions)
        return watered / total if total > 0 else 1.0

    def overspray_counts(self):
        """dict region_id -> total hit count landed on regions that should
        NOT be watered (material 5). Tracked for later use, not penalized yet."""
        return {rid: int(self.hit_counts[self.labels == rid].sum())
                for rid in self.no_water_regions}

    def all_needs_water_done(self, threshold=1.0):
        """True once every region that needs watering has reached at least
        `threshold` fraction complete (default: fully watered)."""
        comp = self.needs_water_completeness()
        if not comp:
            return True
        return all(c >= threshold for c in comp.values())

    def reset(self):
        """Zero out all hit counts (e.g. for a new episode) without
        recomputing region labels/materials."""
        self.hit_counts[:] = 0