import lidar
import numpy as np

class Drone(object):
    id = 0
    sight_range = 0
    lidar = None
    color = None
    marker_value = 5  # voxel value used to mark drone position

    def __init__(self, id, sight_range, values, pos, goal, color=None):
        self.id = id
        self.pos = np.array(pos, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)
        self.sight_range = sight_range
        self.values = values
        self.color = color if color is not None else np.array([1.0, 0.0, 0.0])  # default red

    def move(self, direction, amount=1.0):
        direction = np.array(direction, dtype=np.float32)
        amount = float(amount)  # ensure float movement amounts
        
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
            return True

        # NOTE: we no longer write the drone into self.values. The voxel grid
        # is terrain-only now; the drone's true position lives in self.pos
        # (a float) and is rendered separately as its own mesh. Writing a
        # marker_value into the integer grid was what snapped the drone's
        # visible position onto whole voxel cells no matter what float value
        # self.pos actually held.
        self.pos = new_pos

        return False  # no collision this step
    
    def update_position(self, new_pos):
        """Update drone position directly (for teleporting or initial placement).
        
        Args:
            new_pos: tuple or array-like of (x, y, z) with float values allowed
        """
        new_pos = np.array(new_pos, dtype=np.float32)
        shape = np.array(self.values.shape)
        
        # check bounds
        if np.any(new_pos < 0) or np.any(new_pos >= shape):
            return False  # out of bounds
        
        # check collision with terrain
        nx, ny, nz = int(new_pos[0]), int(new_pos[1]), int(new_pos[2])
        if self.values[nx, ny, nz] in [1, 2, 3]:
            return False  # collision
        
        # place at new position — values grid stays terrain-only, drone
        # position is tracked purely in self.pos (see note in move())
        self.pos = new_pos
        return True  # success
    
    def get_position(self):
        """Return current position as float array."""
        return self.pos.copy()
    
    def set_color(self, color):
        """Set drone color (RGB array or tuple, values 0-1)."""
        self.color = np.array(color, dtype=np.float32)
    
    def get_color(self):
        """Get drone color as RGB array."""
        return self.color.copy()
    
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