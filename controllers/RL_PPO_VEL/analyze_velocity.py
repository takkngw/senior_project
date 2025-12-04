import csv
import numpy as np
import sys

# Increase field size limit just in case
csv.field_size_limit(sys.maxsize)

filepath = "/Users/tduiisl/Library/CloudStorage/GoogleDrive-takkngw.08.05@gmail.com/マイドライブ/Study/TDU/4年/卒業研究/webots_18関節/controllers/RL_PPO_VEL/ppo_training_20251203_155955/step_data.csv"

vel_x_list = []
vel_y_list = []
vel_z_list = []
rewards = []

print(f"Reading {filepath}...")

try:
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            try:
                vx = float(row['Vel_X'])
                vy = float(row['Vel_Y'])
                vz = float(row['Vel_Z'])
                r = float(row['Reward'])
                
                vel_x_list.append(vx)
                vel_y_list.append(vy)
                vel_z_list.append(vz)
                rewards.append(r)
                
                count += 1
                if count % 1000000 == 0:
                    print(f"Processed {count} rows...")
            except ValueError:
                continue

    print(f"Total rows processed: {count}")

    if count > 0:
        print("-" * 30)
        print(f"Vel_Y (Forward):")
        print(f"  Max: {np.max(vel_y_list):.4f}")
        print(f"  Min: {np.min(vel_y_list):.4f}")
        print(f"  Mean: {np.mean(vel_y_list):.4f}")
        print(f"  Std: {np.std(vel_y_list):.4f}")
        
        print("-" * 30)
        print(f"Vel_X (Side):")
        print(f"  Max: {np.max(vel_x_list):.4f}")
        print(f"  Mean: {np.mean(vel_x_list):.4f}")

        print("-" * 30)
        print(f"Vel_Z (Vertical):")
        print(f"  Max: {np.max(vel_z_list):.4f}")
        print(f"  Mean: {np.mean(vel_z_list):.4f}")
        
        print("-" * 30)
        print(f"Reward:")
        print(f"  Max: {np.max(rewards):.4f}")
        print(f"  Min: {np.min(rewards):.4f}")
        print(f"  Mean: {np.mean(rewards):.4f}")

except Exception as e:
    print(f"Error: {e}")
