# label_dataset.py - ENHANCED RULE-BASED LABELING
import pandas as pd
import numpy as np
import os

# --- Configuration ---
INPUT_LOG_FILE = 'supervisor_log.csv'
OUTPUT_LABELED_FILE = 'labeled_dataset.csv'

# Define the optimal/worst-case parameters for trust scoring
# These values define the *rules* for calculating trust.

# 1. GPS Quality Rules (Satellites & HDOP)
# Higher satellites and lower HDOP (Horizontal Dilution of Precision) mean higher trust.
SATELLITE_MIN_TRUST_THRESHOLD = 8  # Below this, trust drops significantly
HDOP_MAX_TRUST_THRESHOLD = 2.0     # Above this, trust drops significantly

# 2. Vibration Quality Rules
# Lower vibration_x means higher trust.
VIBRATION_MAX_TRUST_THRESHOLD = 30 # Above this, trust drops significantly

# Define the influence (weight) of each sensor on the final score (must sum to 1.0)
WEIGHT_GPS = 0.50
WEIGHT_VIBRATION = 0.50

# --- Trust Calculation Function ---
def calculate_trust_score(row):
    # This is a temporary function to force high variance for debugging!
    # It assigns a very low score if HDOP is above 3.0, otherwise a high score,
    # and adds a random component.
    
    score = 0.9 + np.random.uniform(-0.1, 0.1) # Start around 0.9 with random noise

    # Aggressively penalize bad HDOP (e.g., if HDOP > 3.0, drop score significantly)
    if row['hdop'] > 3.0:
        score -= (row['hdop'] - 3.0) * 0.2
        
    # Clamp to ensure the range is wide
    return np.clip(score, 0.1, 0.99)


# --- Script Logic ---
if not os.path.exists(INPUT_LOG_FILE):
    print(f"🛑 ERROR: Input file not found! Please run the supervisor script first to create '{INPUT_LOG_FILE}'.")
else:
    print(f"Reading data from {INPUT_LOG_FILE}...")
    df = pd.read_csv(INPUT_LOG_FILE)

    # Ensure the required columns for calculation exist
    required_cols = ['satellites', 'hdop', 'vibration_x']
    if not all(col in df.columns for col in required_cols):
        print(f"🛑 ERROR: Input file must contain columns: {required_cols}")
    else:
        print("Calculating dynamic target trust scores (0.1 - 0.99)...")
        
        # Apply the calculation function row by row
        df['target_trust_score'] = df.apply(calculate_trust_score, axis=1)

        # Drop the original 'scenario' column if it exists, as it's no longer needed for labeling
        if 'scenario' in df.columns:
            df.drop(columns=['scenario'], inplace=True)
            
        print(f"✅ Labeling complete! Processed {len(df)} rows.")
        print(f"Saving dynamically labeled dataset to {OUTPUT_LABELED_FILE}...")
        df.to_csv(OUTPUT_LABELED_FILE, index=False)
        print("File saved successfully.")
