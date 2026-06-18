import sys
import os

# --- CRITICAL PATH FIX ---
# Find the main folder path (D:\IIT Internship\FL_Spacecraft_Telemetry)
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MAIN_DIR not in sys.path:
    sys.path.insert(0, MAIN_DIR)
# -------------------------

import json
import numpy as np
import tensorflow as tf

# Now Python can safely find both utils and models!
from utils.data_manager import get_single_machine_data, list_all_channels
from models.architecture import build_spacecraft_brain

# ... (rest of the script remains exactly the same!)

# ------------------------------------------------------------------
# 🚀 THE MASTER TRAINING ENGINE (CNN Upgraded)
# ------------------------------------------------------------------
def train_and_evaluate_channel(chan_id, alpha_value=1.0):
    print(f"\n📡 Starting CNN training on Sensor Channel: [{chan_id}] (Alpha = {alpha_value})")
    
    # 1. Grab our practice homework and answers
    try:
        X_train, y_train = get_single_machine_data(chan_id, mode="train", window_size=100, alpha=alpha_value)
        X_test, y_test = get_single_machine_data(chan_id, mode="test", window_size=100, alpha=1.0)
    except Exception as e:
        print(f"   ⚠️ Skipping {chan_id}: {e}")
        return

    # Look at the shapes of the sensors
    num_windows, window_size, num_features = X_train.shape
    
    # 2. Spawn our Upgraded CNN Brain
    model = build_spacecraft_brain(window_size, num_features)
    
    # 3. Practice! (Train for 3 rounds/epochs)
    model.fit(X_train, y_train, epochs=3, batch_size=64, verbose=0)
            
    # 4. Final Exam! (Evaluate using Test data)
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=64, verbose=0)
    accuracy_percentage = accuracy * 100
    
    print(f"   🎯 Final CNN Exam Score for {chan_id}: {accuracy_percentage:.2f}% Accuracy")
    
    # 5. Save results
    result_metrics = {
        "channel": chan_id,
        "alpha": alpha_value,
        "features_count": num_features,
        "training_windows": num_windows,
        "test_accuracy": round(accuracy_percentage, 2)
    }
    
    os.makedirs("results/single_machine", exist_ok=True)
    output_file = f"results/single_machine/{chan_id}_results.json"
    with open(output_file, 'w') as f:
        json.dump(result_metrics, f, indent=4)
    print(f"   💾 Saved results report to: {output_file}")

# ------------------------------------------------------------------
# RUNNER SCRIPT START ENTRY
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("="*70)
    print("🤖 CNN SINGLE MACHINE LOOPER RUNNING 🤖")
    print("="*70)
    
    channels_to_run = list_all_channels()
    print(f"Robot assistant will loop through ALL {len(channels_to_run)} sensors!")
    
    ALPHA_FACTOR = 1.0 
    
    for channel in channels_to_run:
        train_and_evaluate_channel(channel, alpha_value=ALPHA_FACTOR)
        
    print("\n="*70)
    print("✅ PROCESS COMPLETE: Check your results/single_machine/ folder!")
    print("="*70)