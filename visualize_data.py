import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import sys
import os

def create_visualization():
    # 1. Generate Data
    if not os.path.exists('data'):
        os.makedirs('data')
        
    print("Generating synthetic data...")
    subprocess.run([sys.executable, 'python/simulate.py', '--out', 'data/preview_sim.csv'], check=True)
    
    # 2. Read Data
    df = pd.read_csv('data/preview_sim.csv')
    
    # 3. Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Path Plot
    ax1.plot(df['lon'], df['lat'], label='Vehicle Path', color='blue', linewidth=2)
    ax1.set_title('Simulated Driving Path (GPS)', fontsize=14)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.grid(True)
    ax1.legend()
    
    # Sensor Plot
    # Create time axis relative to start
    t = df['utc_ts'] - df['utc_ts'].iloc[0]
    
    ax2.plot(t, df['acc_y'], label='Lateral Accel (acc_y)', color='red', alpha=0.7)
    ax2.plot(t, df['gyro_z'], label='Yaw Rate (gyro_z)', color='green', alpha=0.7)
    ax2.set_title('Sensor Readings during Curves', fontsize=14)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('data_preview.png', dpi=300)
    print("✅ Visualization saved to data_preview.png")

if __name__ == "__main__":
    create_visualization()
