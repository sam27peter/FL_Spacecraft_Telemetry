import os
import ast
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# CONFIGURATION (Where your folders live)
# ------------------------------------------------------------------
TRAIN_DIR = r"D:\IIT Internship\Nasa Data\data\train"
TEST_DIR = r"D:\IIT Internship\Nasa Data\data\test"
CSV_INDEX = "labeled_anomalies.csv"

# ------------------------------------------------------------------
# CORE HELPER BUTTONS (The cleaning and cutting tools)
# ------------------------------------------------------------------

def clean_and_scale(file_path):
    """
    SQUISHING TOOL: Takes a file and squishes all its wild numbers 
    smoothly between 0.0 and 1.0 so the robot brain doesn't get confused.
    """
    raw_data = np.load(file_path)
    # Min-max scaling formula
    min_val = np.min(raw_data, axis=0)
    max_val = np.max(raw_data, axis=0)
    denom = max_val - min_val
    denom[denom == 0] = 1.0 # Prevent crashing if max equals min
    return (raw_data - min_val) / denom

def load_truth_labels(chan_id, total_rows):
    """
    LABEL GENERATOR: Looks at the CSV file to create an array of answers.
    0 = Safe/Normal flight time.
    1 = Anomaly/Danger window.
    """
    labels = np.zeros(total_rows, dtype=np.int32)
    if not os.path.exists(CSV_INDEX):
        return labels
        
    df = pd.read_csv(CSV_INDEX)
    row = df[df['chan_id'] == chan_id]
    if row.empty:
        return labels
        
    # Read the text brackets like [[12, 50]] and turn them into a real list
    ranges = ast.literal_eval(row['anomaly_sequences'].values[0])
    for start, end in ranges:
        labels[max(0, start):min(total_rows, end)] = 1
    return labels

def slice_into_windows(X, y, window_size=100, alpha=1.0):
    """
    SLIDING WINDOW & ALPHA TOOL: Cuts long data into small blocks of history.
    Also handles DEMAND #3 (Alpha): If alpha is less than 1.0, we throw away
    the end of the data to save time/energy.
    """
    X_windows, y_windows = [], []
    total_len = len(X)
    
    # Slide a window across the timeline
    for i in range(0, total_len - window_size + 1):
        X_windows.append(X[i : i + window_size])
        y_windows.append(y[i + window_size - 1]) # Label of the last second in the window
        
    X_windows = np.array(X_windows, dtype=np.float32)
    y_windows = np.array(y_windows, dtype=np.int32)
    
    # --- Demand 3: Alpha cut-off logic ---
    if 0.0 < alpha < 1.0:
        keep_amount = int(len(X_windows) * alpha)
        X_windows = X_windows[:keep_amount]
        y_windows = y_windows[:keep_amount]
        
    return X_windows, y_windows


# ------------------------------------------------------------------
# MASTER MODES (Your 3 Demands)
# ------------------------------------------------------------------

def get_single_machine_data(chan_id, mode="train", window_size=100, alpha=1.0):
    """
    DEMAND 1: Grabs data for JUST ONE lonely channel.
    mode can be "train" (practice folder) or "test" (exam folder).
    """
    folder = TRAIN_DIR if mode == "train" else TEST_DIR
    file_path = os.path.join(folder, f"{chan_id}.npy")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find the file for channel {chan_id} in {folder}")
        
    X_clean = clean_and_scale(file_path)
    y_clean = load_truth_labels(chan_id, total_rows=len(X_clean))
    
    X_windows, y_windows = slice_into_windows(X_clean, y_clean, window_size, alpha)
    return X_windows, y_windows

def list_all_channels():
    """
    Returns a clean list of all 82 sensor names found in your train folder.
    """
    files = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.npy')]
    return [f.replace('.npy', '') for f in files]


# ------------------------------------------------------------------
# TEST RUNNER (Let's make sure our tool works perfectly)
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print("🛰️ MASTER DATA MANAGER TEST DRIVE 🛰️")
    print("="*60)
    
    # Let's find out how many channels we have total
    all_channels = list_all_channels()
    print(f"✨ Found {len(all_channels)} total spacecraft sensor files on your disk.")
    
    print("\n📦 Testing Demand 1: Single Machine (Loading A-1 for practice)...")
    X_a1, y_a1 = get_single_machine_data("A-1", mode="train", alpha=1.0)
    print(f"   A-1 Practice Windows Shape: {X_a1.shape}")
    print(f"   A-1 Practice Answers Shape: {y_a1.shape}")
    
    print("\n🔋 Testing Demand 3: Alpha Factor (Loading A-1 with Alpha=0.4)...")
    X_alpha, y_alpha = get_single_machine_data("A-1", mode="train", alpha=0.4)
    print(f"   Notice how the number of windows shrank from {len(X_a1)} down to {len(X_alpha)}!")
    print("="*60)