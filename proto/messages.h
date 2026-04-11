#pragma once
#include <stdint.h>

// Engine → Python: full drone state, sent every physics tick
typedef struct {
    uint32_t tick;           // physics tick counter
    float    pos[3];         // x, y, z  (metres, Z up)
    float    vel[3];         // vx, vy, vz (m/s)
    float    accel[3];       // body-frame acceleration (m/s²)
    float    gyro[3];        // body-frame angular velocity (rad/s)
    float    attitude[3];    // roll, pitch, yaw (radians)
    float    lidar[16];      // 16 ray distances (metres, -1 = no hit)
    float    waypoint[3];    // current waypoint x, y, z
    uint8_t  collided;       // 1 if collision this tick
    uint8_t  _pad[3];        // alignment padding
} DroneState;                // 124 bytes

// Python → Engine: velocity command from NN, sent at ~50 Hz
typedef struct {
    uint32_t tick;           // echo back the tick this responds to
    float    vx;             // m/s
    float    vy;             // m/s
    float    vz;             // m/s (Z up)
    float    yaw_rate;       // rad/s
} VelCommand;                // 24 bytes
