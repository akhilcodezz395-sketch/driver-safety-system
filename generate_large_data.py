import pandas as pd
import numpy as np
import math
import random
from datetime import datetime, timedelta
from python.simulate import generate_segment

def generate_large_dataset(out_file, num_trips=50):
    """
    Generate a large diverse dataset with multiple trips
    """
    print(f"Generating {num_trips} trips for large dataset...")
    
    all_data = {k: [] for k in ['utc_ts', 'lat', 'lon', 'speed', 'heading', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']}
    
    dt = 0.05  # 20 Hz
    
    # Define diverse scenario templates
    scenarios = [
        # Safe scenarios (straight roads, low speed)
        {'type': 'safe', 'segments': [('straight', 15, 0, 10.0)]},
        {'type': 'safe', 'segments': [('straight', 20, 0, 12.0)]},
        {'type': 'safe', 'segments': [('gentle_curve', 10, 5, 10.0)]},
        
        # Mild scenarios (gentle curves, moderate speed)
        {'type': 'mild', 'segments': [('curve', 8, 12, 15.0)]},
        {'type': 'mild', 'segments': [('curve', 6, -10, 14.0)]},
        {'type': 'mild', 'segments': [('straight', 5, 0, 18.0), ('curve', 5, 15, 18.0)]},
        
        # Urgent scenarios (sharp curves, higher speed)
        {'type': 'urgent', 'segments': [('sharp_curve', 5, 25, 18.0)]},
        {'type': 'urgent', 'segments': [('sharp_curve', 4, -30, 20.0)]},
        {'type': 'urgent', 'segments': [('straight', 3, 0, 22.0), ('sharp_curve', 4, 28, 22.0)]},
        
        # Hectic scenarios (very sharp curves, high speed)
        {'type': 'hectic', 'segments': [('very_sharp', 3, 45, 25.0)]},
        {'type': 'hectic', 'segments': [('very_sharp', 3, -50, 28.0)]},
        {'type': 'hectic', 'segments': [('straight', 2, 0, 30.0), ('very_sharp', 3, 55, 30.0)]},
    ]
    
    for trip_idx in range(num_trips):
        # Random starting position
        curr_lat = 37.7749 + random.uniform(-0.1, 0.1)
        curr_lon = -122.4194 + random.uniform(-0.1, 0.1)
        curr_heading = random.uniform(0, 360)
        
        start_time = datetime.now() + timedelta(hours=trip_idx)
        current_ts = start_time
        
        # Pick random scenarios for this trip
        num_scenarios = random.randint(3, 6)
        trip_scenarios = random.choices(scenarios, k=num_scenarios)
        
        for scenario in trip_scenarios:
            for seg_name, duration, turn_rate, speed in scenario['segments']:
                # Add some randomness
                speed += random.uniform(-2, 2)
                turn_rate += random.uniform(-3, 3)
                duration += random.uniform(-1, 1)
                
                seg = generate_segment(curr_lat, curr_lon, curr_heading, speed, duration, turn_rate, dt)
                
                N = len(seg['lat'])
                for i in range(N):
                    all_data['utc_ts'].append((current_ts + timedelta(seconds=i*dt)).timestamp())
                    for k in seg:
                        all_data[k].append(seg[k][i])
                
                # Update state
                curr_lat = seg['lat'][-1]
                curr_lon = seg['lon'][-1]
                curr_heading = seg['heading'][-1]
                current_ts += timedelta(seconds=N*dt)
        
        if (trip_idx + 1) % 10 == 0:
            print(f"Generated {trip_idx + 1}/{num_trips} trips...")
    
    df = pd.DataFrame(all_data)
    df.to_csv(out_file, index=False)
    print(f"✅ Saved {len(df)} samples to {out_file}")
    print(f"   That's {len(df) / 20:.1f} seconds of driving data!")

if __name__ == "__main__":
    generate_large_dataset('data/large_training_data.csv', num_trips=100)
