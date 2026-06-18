import sys
import os

# --- CRITICAL PATH FIX ---
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MAIN_DIR not in sys.path:
    sys.path.insert(0, MAIN_DIR)
# -------------------------

import json
import numpy as np
import tensorflow as tf

from utils.data_manager import get_single_machine_data, list_all_channels
from models.architecture import build_spacecraft_brain

# ... (rest of the script remains exactly the same!)

# ------------------------------------------------------------------
# 🚀 THE MASTER GLOBAL TRAINING ENGINE (CNN Upgraded)
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("="*70)
    print("🌍 CNN GLOBAL MACHINE (SUPER-BRAIN) TRAINING 🌍")
    print("="*70)
    
    all_channels = list_all_channels()
    print(f"CNN Super-Brain will sequentially learn from all {len(all_channels)} channels.")
    
    trained_brains = {}
    ALPHA_FACTOR = 1.0
    
    # --- PHASE 1: CENTRALIZED TRAINING ---
    print("\n📚 Step 1: Training the CNN Master Brain across all files...")
    for idx, channel in enumerate(all_channels):
        print(f"   [{idx+1}/{len(all_channels)}] Brain is studying Channel: {channel}...")
        
        try:
            X_train, y_train = get_single_machine_data(channel, mode="train", window_size=100, alpha=ALPHA_FACTOR)
        except Exception as e:
            print(f"   ⚠️ Skipping training for {channel}: {e}")
            continue
            
        num_windows, window_size, num_features = X_train.shape
        
        # Build the shared CNN if we haven't seen this shape yet
        if num_features not in trained_brains:
            trained_brains[num_features] = build_spacecraft_brain(window_size, num_features)
            
        master_model = trained_brains[num_features]
        master_model.fit(X_train, y_train, epochs=2, batch_size=64, verbose=0)
        
    # --- PHASE 2: GLOBAL EVALUATION ---
    print("\n📝 Step 2: Testing the CNN Super-Brain on the Final Exams...")
    os.makedirs("results/global_machine", exist_ok=True)
    
    for channel in all_channels:
        try:
            X_test, y_test = get_single_machine_data(channel, mode="test", window_size=100, alpha=1.0)
        except Exception as e:
            print(f"   ⚠️ Skipping test for {channel}: {e}")
            continue
            
        _, window_size, num_features = X_test.shape
        
        if num_features in trained_brains:
            master_model = trained_brains[num_features]
            loss, accuracy = master_model.evaluate(X_test, y_test, batch_size=64, verbose=0)
            accuracy_percentage = accuracy * 100
            
            print(f"   🎯 Global CNN Exam Score for {channel}: {accuracy_percentage:.2f}% Accuracy")
            
            result_metrics = {
                "channel": channel,
                "alpha": ALPHA_FACTOR,
                "features_count": num_features,
                "test_accuracy": round(accuracy_percentage, 2),
                "mode": "global_machine_cnn"
            }
            
            output_file = f"results/global_machine/{channel}_global_results.json"
            with open(output_file, 'w') as f:
                json.dump(result_metrics, f, indent=4)
        else:
            print(f"   ❌ No trained global CNN model found for feature size {num_features}")
            
    print("\n="*70)
    print("✅ GLOBAL CNN PROCESS COMPLETE: Check your results/global_machine/ folder!")
    print("="*70)