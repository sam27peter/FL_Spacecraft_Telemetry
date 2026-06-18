import os
import pandas as pd
import ast

# 1. Define paths (Assumes running from the project root directory)
csv_path = "labeled_anomalies.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Could not find {csv_path}. Please make sure it's in the root folder.")

print("="*60)
print("🛰️ SPACECRAFT TELEMETRY DATA EXPLORATION MODULE 🛰️")
print("="*60)

# 2. Load the dataset
df = pd.read_csv(csv_path)

# 3. Basic Shape Inspection
print(f"\n🔹 Dataset Matrix Dimensions:")
print(f"Total Telemetry Channels Indexed: {df.shape[0]}")
print(f"Total Feature Columns Available: {df.shape[1]}")
print(f"Column Names: {list(df.columns)}")

# 4. Analyze Spacecraft Distribution
print(f"\n🔹 Distribution by Spacecraft Entity:")
spacecraft_counts = df['spacecraft'].value_counts()
for craft, count in spacecraft_counts.items():
    print(f"  • {craft}: {count} individual telemetry channels")

# 5. Analyze Anomaly Classification Types
print(f"\n🔹 Anomaly Categories Found:")
# The class column stores string-represented lists, so we look at raw unique types
class_counts = df['class'].value_counts()
for cls_type, count in class_counts.items():
    print(f"  • {cls_type}: {count} channels")

# 6. Sample Row Analysis to See Sequences
print(f"\n🔹 Sample Metadata Breakdown (First 3 Channels):")
print("-" * 60)
for idx, row in df.head(3).iterrows():
    # Convert string representation of list to actual python list safely
    sequences = ast.literal_eval(row['anomaly_sequences'])
    print(f"Channel ID: {row['chan_id']} | Spacecraft: {row['spacecraft']}")
    print(f"  └─ Total Data Points Streamed: {row['num_values']}")
    print(f"  └─ Target Anomaly Indices (Start, End): {sequences}")
    print(f"  └─ Specific Anomaly Class Profile: {row['class']}")
    print("-" * 60)