import os
import ast
import numpy as np
import pandas as pd

def load_and_scale_features(file_path):
    """
    Loads a raw .npy file and applies Min-Max Normalization to scale 
    all features smoothly between 0.0 and 1.0.
    
    Input: file_path (str) - Path to the specific .npy telemetry channel
    Output: normalized_data (np.array) - Multi-column feature matrix scaled [0, 1]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Telemetry array not found at: {file_path}")
        
    # Load raw data matrix
    raw_data = np.load(file_path)
    
    # Min-Max Scaling implementation: (X - min) / (max - min)
    # axis=0 means we find min and max independently for each sensor column
    min_val = np.min(raw_data, axis=0)
    max_val = np.max(raw_data, axis=0)
    
    # Avoid zero division if max equals min for a stable feature channel
    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0
    
    normalized_data = (raw_data - min_val) / range_val
    return normalized_data

def generate_binary_labels(chan_id, sequence_length, csv_path="labeled_anomalies.csv"):
    """
    Cross-references the labeled_anomalies.csv master index to generate 
    a 1D array of labels where 0 is normal and 1 is an anomaly interval.
    
    Inputs:
        chan_id (str) - The target channel (e.g., 'A-1')
        sequence_length (int) - Total number of rows/timestamps in the data
        csv_path (str) - Path to the metadata CSV file
    Output:
        labels (np.array) - A binary array of shape (sequence_length,)
    """
    # 1. Initialize an array of zeros (assuming everything is normal by default)
    labels = np.zeros(sequence_length, dtype=np.int32)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata index missing: {csv_path}")
        
    df_index = pd.read_csv(csv_path)
    
    # 2. Extract row for the requested channel ID
    channel_row = df_index[df_index['chan_id'] == chan_id]
    
    if channel_row.empty:
        print(f"⚠️ Warning: Channel {chan_id} not indexed in CSV. Defaulting to all normal labels.")
        return labels
        
    # 3. Parse the anomaly index sequences out of the string representation
    anomaly_string = channel_row['anomaly_sequences'].values[0]
    anomaly_ranges = ast.literal_eval(anomaly_string) # Safe conversion to python list of lists
    
    # 4. Loop through the explicit anomaly window boundaries and set flags to 1
    for start, end in anomaly_ranges:
        # Secure boundary checks to avoid index errors outside array bounds
        start_idx = max(0, start)
        end_idx = min(sequence_length, end)
        
        labels[start_idx:end_idx] = 1
        
    return labels

# Test execution validation block
if __name__ == "__main__":
    print("="*60)
    print("🛠️ PREPROCESSING PIPELINE UNIT TEST RUNNER 🛠️")
    print("="*60)
    
    # Test with the first channel from your directory path
    test_chan = "A-1"
    data_dir = r"D:\IIT Internship\Nasa Data\data\train"
    sample_file = os.path.join(data_dir, f"{test_chan}.npy")
    
    if os.path.exists(sample_file):
        # Run normalization
        X_test = load_and_scale_features(sample_file)
        # Run label engineering
        y_test = generate_binary_labels(test_chan, sequence_length=len(X_test))
        
        print(f"✔ Successfully Processed Channel: {test_chan}")
        print(f"👉 Normalized Feature Matrix Shape (X): {X_test.shape}")
        print(f"👉 Generated Binary Labels Vector Shape (y): {y_test.shape}")
        print(f"👉 Value Range Check -> Min: {np.min(X_test)}, Max: {np.max(X_test)}")
        print(f"👉 Distribution of Ground Truth -> Normal points: {np.sum(y_test==0)}, Anomaly points: {np.sum(y_test==1)}")
    else:
        print(f"Could not locate sample file {sample_file} to verify unit run.")