import socket
import struct
import threading

# Must exactly match messages.h
# 4s = uint32 tick, 3f = float[3], repeated for each field, 16f = lidar, B = uint8, 3x = padding
DRONE_STATE_FMT  = "=I 3f 3f 3f 3f 3f 16f 3f B 3x"
DRONE_STATE_SIZE = struct.calcsize(DRONE_STATE_FMT)  # must be 124

VEL_CMD_FMT  = "=I 4f"
VEL_CMD_SIZE = struct.calcsize(VEL_CMD_FMT)          # must be 24

class DroneTransport:
    def __init__(
        self,
        listen_host="0.0.0.0",
        listen_port=9001,    # Python listens here for DroneState from engine
        engine_host="127.0.0.1",
        engine_port=9000,    # engine listens here for VelCommand from Python
    ):
        assert DRONE_STATE_SIZE == 124, f"DroneState size mismatch: {DRONE_STATE_SIZE}"
        assert VEL_CMD_SIZE     == 24,  f"VelCommand size mismatch: {VEL_CMD_SIZE}"

        self.engine_addr = (engine_host, engine_port)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((listen_host, listen_port))
        self.sock.settimeout(2.0)

        self._lock        = threading.Lock()
        self._latest      = None   # most recent DroneState dict
        self._running     = False
        self._recv_thread = None

    # ------------------------------------------------------------------ #
    #  Background receive thread — engine pushes state at ~400 Hz,        #
    #  we just keep the latest and let step() read it at 50 Hz            #
    # ------------------------------------------------------------------ #

    def start(self):
        self._running = True
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

    def stop(self):
        self._running = False
        self.sock.close()

    def _recv_loop(self):
        while self._running:
            try:
                data, _ = self.sock.recvfrom(256)
                if len(data) != DRONE_STATE_SIZE:
                    continue
                state = self._unpack_state(data)
                with self._lock:
                    self._latest = state
            except socket.timeout:
                continue
            except OSError:
                break

    def _unpack_state(self, data):
        fields = struct.unpack(DRONE_STATE_FMT, data)
        return {
            "tick":      fields[0],
            "pos":       list(fields[1:4]),
            "vel":       list(fields[4:7]),
            "accel":     list(fields[7:10]),
            "gyro":      list(fields[10:13]),
            "attitude":  list(fields[13:16]),
            "lidar":     list(fields[16:32]),
            "waypoint":  list(fields[32:35]),
            "collided":  bool(fields[35]),
        }

    # ------------------------------------------------------------------ #
    #  Called by Gymnasium step() — blocks until a fresh state arrives    #
    # ------------------------------------------------------------------ #

    def get_state(self, timeout=1.0):
        """Returns the latest DroneState dict. Blocks if none yet received."""
        import time
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest is not None:
                    return self._latest
            time.sleep(0.001)
        raise TimeoutError("No DroneState received from engine within timeout")

    def get_fresh_state(self, after_tick, timeout=1.0):
        """Blocks until a state newer than after_tick arrives. Use in step()."""
        import time
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest and self._latest["tick"] > after_tick:
                    return self._latest
            time.sleep(0.001)
        raise TimeoutError(f"No state newer than tick {after_tick} received")

    # ------------------------------------------------------------------ #
    #  Called by Gymnasium step() to send NN action to engine             #
    # ------------------------------------------------------------------ #

    def send_command(self, tick, vx, vy, vz, yaw_rate):
        data = struct.pack(VEL_CMD_FMT, tick, vx, vy, vz, yaw_rate)
        self.sock.sendto(data, self.engine_addr)
