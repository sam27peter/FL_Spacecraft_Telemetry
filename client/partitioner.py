import os
import sys

# --- CRITICAL PATH FIX ---
# This ensures we can import our data manager no matter where we run this script from
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MAIN_DIR not in sys.path:
    sys.path.insert(0, MAIN_DIR)
# -------------------------

from utils.data_manager import list_all_channels

def get_non_iid_partitions():
    """
    STRATEGY 1: NON-IID (Subsystem Grouping)
    Groups sensors by their prefix (A, B, C...). Creates clients based on unique letters.
    """
    all_channels = list_all_channels()
    partitions = {}
    
    for channel in all_channels:
        prefix = channel.split('-')[0]
        if prefix not in partitions:
            partitions[prefix] = []
        partitions[prefix].append(channel)
        
    # Map to integer client IDs: 0, 1, 2...
    sorted_prefixes = sorted(list(partitions.keys()))
    client_mapping = {idx: partitions[prefix] for idx, prefix in enumerate(sorted_prefixes)}
    
    return client_mapping

def get_iid_partitions(num_clients):
    """
    STRATEGY 2: IID (Uniform Split)
    Evenly distributes the 82 channels across the exact same number of clients.
    """
    all_channels = sorted(list_all_channels())
    client_mapping = {i: [] for i in range(num_clients)}
    
    # Deal the channels out like a deck of cards
    for idx, channel in enumerate(all_channels):
        client_id = idx % num_clients
        client_mapping[client_id].append(channel)
        
    return client_mapping

# ------------------------------------------------------------------
# TEST RUNNER (Verify both maps are perfectly equal!)
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print("🛰️ FL CLIENT PARTITIONER TEST DRIVE 🛰️")
    print("="*60)
    
    # Generate the Non-IID Map first to see how many prefixes exist
    non_iid_map = get_non_iid_partitions()
    num_target_clients = len(non_iid_map)
    
    print(f"\n📦 Testing Non-IID Strategy (Grouped by Subsystem Prefix):")
    print(f"✨ Created {num_target_clients} total clients.")
    for client_id, channels in non_iid_map.items():
        print(f"   🔹 Client {client_id} -> {len(channels)} sensors (e.g., {channels[:3]}...)")
        
    print(f"\n🔋 Testing IID Strategy (Forcing exactly {num_target_clients} clients to match!):")
    iid_map = get_iid_partitions(num_clients=num_target_clients)
    print(f"✨ Created {len(iid_map)} total clients.")
    for client_id, channels in iid_map.items():
        print(f"   🔸 Client {client_id} -> {len(channels)} sensors (e.g., {channels[:3]}...)")
    print("="*60)