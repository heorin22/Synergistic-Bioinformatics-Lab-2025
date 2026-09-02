import torch
import os
import numpy as np

# Define the paths to your generated final feature files
linear_features_path = "Data/Feature/AggregatedWeightedESM/aggregated_weighted_features_linear.pt"
softmax_features_path = "Data/Feature/AggregatedWeightedESM/aggregated_weighted_features_softmax.pt"

def inspect_features_file(file_path, weighting_type):
    print(f"\n--- Inspecting {weighting_type} features file: {file_path} ---")
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        data = torch.load(file_path, map_location='cpu')

        print(f"Successfully loaded data from: {file_path}")
        print(f"Type of loaded data: {type(data)}")

        if isinstance(data, dict):
            num_gpcrs = len(data)
            print(f"Number of GPCRs (UniProt IDs) with aggregated features: {num_gpcrs}")
            
            # Check a few random or the first few GPCRs to confirm vector dimensions and validity
            print("\n--- Sample GPCR Feature Details ---")
            sample_uids = list(data.keys())[:5] # Take first 5 UIDs
            if not sample_uids:
                print("No GPCRs found in this file.")
                return

            for i, uid in enumerate(sample_uids):
                feature_vector = data[uid]
                print(f"UniProt ID: {uid}")
                print(f"  Feature vector type: {type(feature_vector)}")
                print(f"  Feature vector shape: {feature_vector.shape}")
                print(f"  Expected shape: (1286,)") # Your target dimension
                
                # Check for NaNs or all zeros (excluding cases where it might be intentional, e.g., for non-processable GPCRs)
                if np.isnan(feature_vector).any():
                    print(f"  WARNING: Contains NaN values!")
                if np.all(feature_vector == 0):
                    print(f"  WARNING: Contains all zero values (may indicate no valid data processed for this GPCR)!")
                
                # Optionally, print first few elements to check values
                print(f"  First 5 elements: {feature_vector[:5]}")
                print(f"  Last 5 elements: {feature_vector[-5:]}")
        else:
            print(f"Loaded data is not a dictionary of features. Type: {type(data)}")

    except Exception as e:
        print(f"An error occurred while loading or inspecting {file_path}: {e}")

if __name__ == "__main__":
    inspect_features_file(linear_features_path, "Linear-Weighted")
    inspect_features_file(softmax_features_path, "Softmax-Weighted")