import lidar
import numpy as np

class Drone(object):
    id = 0
    sight_range = 0
    lidar = None

    def __init__(self, id, sight_range, values, pos, goal):
        self.id = id
        self.pos = np.array(pos, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)
        self.sight_range = sight_range
        self.values = values

    def move(self, direction, amount=1):
        direction = np.array(direction, dtype=np.float32)
        
        # guard against NaN actions
        if np.any(np.isnan(direction)) or np.any(np.isinf(direction)):
            return False
        
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        new_pos = self.pos + direction * amount
        shape = np.array(self.values.shape)

        # treat flying out of bounds as a crash — don't clip, stay put and signal hit
        if np.any(new_pos < 0) or np.any(new_pos >= shape):
            return True

        nx, ny, nz = int(new_pos[0]), int(new_pos[1]), int(new_pos[2])
        hit = self.values[nx, ny, nz] in [1, 2, 3]   # terrain or tree or tree line

        if hit:
            # don't fly through solid terrain/trees — stay put, signal the collision.
            # (previously this still moved into the voxel and overwrote it with the
            # drone marker, which hid every real collision from the env's crash check)
            return True

        ox, oy, oz = int(self.pos[0]), int(self.pos[1]), int(self.pos[2])
        if self.values[ox, oy, oz] == 5:   # only clear the cell if it's the drone marker
            self.values[ox, oy, oz] = 0
        self.pos = new_pos
        self.values[nx, ny, nz] = 5

        return False  # no collision this step
    
    def get_id(self):
        return self.id
    
    def get_sight_range(self):
        return self.sight_range
    
    def set_id(self, id):
        self.id = id

    def set_sight_range(self, sight_range):
        self.sight_range = sight_range

    def get_obs(self):
        scan = lidar.get_lidar_surroundings(self.values, self.pos, self.sight_range)
        direction, distance = lidar.get_goal_vector(self.pos, self.goal, self.sight_range)

        return np.concatenate([scan, direction, [distance]])  # (n_rays + 4,)