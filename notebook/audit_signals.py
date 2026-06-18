import os
import numpy as np
import pandas as pd

# The exact path you provided for the raw telemetry training data
DATA_DIR = r"D:\IIT Internship\Nasa Data\data\train"

print("="*85)
print("📡 TELEMETRY SIGNAL AUDIT & STATISTICAL SUMMARY 📡")
print("="*85)

# 1. Verify the directory exists
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Directory not found: {DATA_DIR}. Please check your path.")

# List to hold our statistical findings for each channel
audit_results = []

# 2. Get a list of all .npy files in the directory
# (Assuming the files are saved as numpy arrays, which is standard for this dataset)
files = [f for f in os.listdir(DATA_DIR) if f.endswith('.npy')]

if len(files) == 0:
    print("No .npy files found in the directory. (If your data is .csv, let me know!)")
else:
    print(f"Found {len(files)} telemetry files. Analyzing structure and statistics...\n")
    
    # 3. Loop through every file and calculate statistics
    for file_name in files:
        file_path = os.path.join(DATA_DIR, file_name)
        
        try:
            # Load the raw mathematical matrix
            data = np.load(file_path)
            
            # Extract basic structure
            shape = str(data.shape)
            dtype = str(data.dtype)
            
            # Count corrupt or missing data points
            # NaN = Not a Number, Inf = Infinity
            nan_count = np.isnan(data).sum()
            inf_count = np.isinf(data).sum()
            
            # Calculate signal statistics (ignoring NaNs so it doesn't crash)
            d_min = np.nanmin(data) if data.size > 0 else 0
            d_max = np.nanmax(data) if data.size > 0 else 0
            d_mean = np.nanmean(data) if data.size > 0 else 0
            d_std = np.nanstd(data) if data.size > 0 else 0
            
            # Save to our results list
            audit_results.append({
                "Channel Name": file_name.replace('.npy', ''),
                "Shape": shape,
                "Datatype": dtype,
                "Min": round(d_min, 4),
                "Max": round(d_max, 4),
                "Mean": round(d_mean, 4),
                "Std": round(d_std, 4),
                "NaN Count": nan_count,
                "Inf Count": inf_count
            })
            
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

    # 4. Convert the results into a pandas DataFrame for a beautiful table
    df_audit = pd.DataFrame(audit_results)
    
    # Print the first 15 rows to the terminal to verify
    print(df_audit.head(15).to_string(index=False))
    
    if len(df_audit) > 15:
        print(f"\n... and {len(df_audit) - 15} more channels analyzed.")
    
    # 5. Save the full audit to a CSV so you can view it anytime
    report_path = "notebook/signal_audit_report.csv"
    df_audit.to_csv(report_path, index=False)
    print(f"\n✅ Full statistical audit saved successfully to: {report_path}")