import tensorflow as tf
from tensorflow.keras import layers, models

def build_spacecraft_brain(window_size, num_features):
    """
    Upgraded 1D-CNN Architecture for Time-Series Anomaly Detection.
    Shared by Single, Global, and Federated clients.
    """
    model = models.Sequential([
        # The CNN slides across the 100 timesteps looking for spikes/drops
        layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(window_size, num_features)),
        layers.MaxPooling1D(pool_size=2), # Compresses the timeline to keep the strongest signals
        
        # A second pass to find deeper, more complex patterns
        layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        
        # Now we flatten the extracted clues, not the raw data!
        layers.Flatten(),
        
        # The Detective Room
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2), # Prevents memorizing the data (overfitting)
        
        # The Announcer: Shouting a danger probability (0 to 1)
        layers.Dense(1, activation='sigmoid')
    ])
    
    # We compile it here so EVERY script uses the exact same grading rules
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model