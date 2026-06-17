import numpy as np
import os

# Define the local path where your training data resides
DATA_DIR = r"D:\IIT Internship\Nasa Data\data\train"

print("==================================================")
print("     NASA TELEMETRY DATA EXPLORATION PHASE        ")
print("==================================================\n")

# List out the first few .npy files present
npy_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.npy')]
print(f"Total telemetry channel files found: {len(npy_files)}")
print(f"Sample file channels: {npy_files[:5]}\n")

# Pick one channel file to explore (e.g., E-1.npy or whichever is present)
sample_channel = npy_files[0] 
sample_path = os.path.join(DATA_DIR, sample_channel)

print(f"--- Investigating Channel: {sample_channel} ---")

# Load the file using numpy
data_matrix = np.load(sample_path)

print(f"Data Matrix Shape: {data_matrix.shape}")
print(f"Total Time-steps:  {data_matrix.shape[0]}")
print(f"Total Features:     {data_matrix.shape[1]}")

print("\n--- Statistical Snapshot of Continuous Telemetry Value (Column 0) ---")
telemetry_values = data_matrix[:, 0]
print(f"Mean Value:       {np.mean(telemetry_values):.4f}")
print(f"Standard Dev:     {np.std(telemetry_values):.4f}")
print(f"Minimum Value:    {np.min(telemetry_values):.4f}")
print(f"Maximum Value:    {np.max(telemetry_values):.4f}")
print("==================================================")