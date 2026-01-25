import pandas as pd
import numpy as np
import math
import argparse
import random
from datetime import datetime, timedelta

# Constants
G = 9.81

def generate_segment(start_lat, start_lon, start_heading, speed_mps, duration_sec, turn_rate_deg_per_sec, dt=0.01):
    """
    Generates a segment of driving data.
    acc_y will be calculated from centripetal force: a = v^2 / r = v * omega
    """
    t = np.arange(0, duration_sec, dt)
    n = len(t)
    
    lats = []
    lons = []
    headings = []
    speeds = []
    acc_xs, acc_ys, acc_zs = [], [], []
    gyro_xs, gyro_ys, gyro_zs = [], [], []
    timestamps = []
    
    curr_lat = start_lat
    curr_lon = start_lon
    curr_heading = start_heading
    
    # Omega (rad/s)
    omit_rad = math.radians(turn_rate_deg_per_sec)
    
    for i in range(n):
        # Update Position
        # One simple approximation for small distances
        # d_lat = v * cos(heading) * dt / R_earth
        # d_lon = v * sin(heading) * dt / (R_earth * cos(lat))
        
        R_earth = 6371000
        dx = speed_mps * math.cos(math.radians(curr_heading)) * dt # North
        dy = speed_mps * math.sin(math.radians(curr_heading)) * dt # East
        
        d_lat = math.degrees(dx / R_earth)
        d_lon = math.degrees(dy / (R_earth * math.cos(math.radians(curr_lat))))
        
        curr_lat += d_lat
        curr_lon += d_lon
        
        curr_heading += turn_rate_deg_per_sec * dt
        curr_heading = (curr_heading + 360) % 360
        
        lats.append(curr_lat)
        lons.append(curr_lon)
        headings.append(curr_heading)
        speeds.append(speed_mps)
        
        # Physics / IMU
        # gyro_z is turn rate
        gyro_z = omit_rad
        # acc_y is centripetal acc = v * omega
        acc_y = speed_mps * omit_rad
        
        # Add noise
        acc_xs.append(np.random.normal(0, 0.1)) # Forward accel noise
        acc_ys.append(acc_y + np.random.normal(0, 0.2))
        acc_zs.append(G + np.random.normal(0, 0.1)) # Gravity
        
        gyro_xs.append(np.random.normal(0, 0.01))
        gyro_ys.append(np.random.normal(0, 0.01))
        gyro_zs.append(gyro_z + np.random.normal(0, 0.02))
        
    return {
        'lat': lats,
        'lon': lons,
        'heading': headings,
        'speed': speeds,
        'acc_x': acc_xs,
        'acc_y': acc_ys,
        'acc_z': acc_zs,
        'gyro_x': gyro_xs,
        'gyro_y': gyro_ys,
        'gyro_z': gyro_zs
    }

def generate_trip(out_file):
    print(f"Generating trip to {out_file}...")
    
    data = {k: [] for k in ['utc_ts', 'lat', 'lon', 'speed', 'heading', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']}
    
    start_time = datetime.now()
    curr_lat, curr_lon = 37.7749, -122.4194 # SF
    curr_heading = 0.0
    
    segments = [
        ('straight', 10, 0),
        ('curve_right_mild', 5, 10),
        ('straight', 5, 0),
        ('curve_left_sharp', 4, -25),
        ('straight', 10, 0),
        ('curve_right_urgent', 3, 40),
        ('straight', 5, 0)
    ]
    
    dt = 0.05 # 20 Hz
    current_ts = start_time
    
    for label, duration, turn_rate in segments:
        speed = 15.0 # m/s ~ 54 km/h
        if 'sharp' in label: speed = 10.0
        if 'urgent' in label: speed = 20.0 # Fast turn
        
        seg = generate_segment(curr_lat, curr_lon, curr_heading, speed, duration, turn_rate, dt)
        
        N = len(seg['lat'])
        for i in range(N):
            data['utc_ts'].append((current_ts + timedelta(seconds=i*dt)).timestamp())
            for k in seg:
                data[k].append(seg[k][i])
                
        # Update state for next segment
        curr_lat = seg['lat'][-1]
        curr_lon = seg['lon'][-1]
        curr_heading = seg['heading'][-1]
        current_ts += timedelta(seconds=N*dt)
        
    df = pd.DataFrame(data)
    df.to_csv(out_file, index=False)
    print(f"Saved {len(df)} samples to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, required=True, help="Output CSV file")
    args = parser.parse_args()
    
    generate_trip(args.out)
