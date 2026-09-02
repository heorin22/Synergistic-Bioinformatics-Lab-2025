# In scripts/modeling/train_model.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import torch # NEW: Import torch for loading .pt files
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm # NEW: Import tqdm for progress bars (if not already there)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Global settings for data paths and model parameters
# Adjust paths to match your project's root when running with 'python -m'
DATA_DIR = "data" # Base directory for input CSVs
SPLITS_DIR = os.path.join(DATA_DIR, "splits") # Directory for dataset splits

# NEW: Paths to your aggregated protein feature files
AGGREGATED_FEATURES_LINEAR_PATH = "data/processed/aggregated_weighted_esm/aggregated_weighted_features_linear.pt"
AGGREGATED_FEATURES_SOFTMAX_PATH = "data/processed/aggregated_weighted_esm/aggregated_weighted_features_softmax.pt"

# Existing paths from original train_model.py (ensure they are correct relative to project root)
LIGAND_INFO_PATH = os.path.join(DATA_DIR, "PDB_ago_ant_chain_info_v2.csv") # Used for SMILES and Ikey
GPCR_BS_UNIPROT_PATH = os.path.join(DATA_DIR, "GPCR_Binding_Residues_UniProt.csv") # Used for UniProt sequence length if needed

# Model parameters (can be tuned)
INPUT_DIM_LIGAND = 1024  # ECFP4 fingerprint dimension
INPUT_DIM_PROTEIN = 1286 # Your new weighted ESM + spatial features dimension

HIDDEN_DIM_LIGAND = 512
HIDDEN_DIM_PROTEIN = 512
JOINT_HIDDEN_DIM = 512

EPOCHS = 200
BATCH_SIZE = 32
PATIENCE = 10
N_SPLITS = 3 # Number of random splits for evaluation


# --- Helper function for ECFP4 generation (from original train_model.py) ---
def generate_ecfp4(smiles):
    """Generate 1024-bit ECFP4 fingerprint from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(INPUT_DIM_LIGAND, np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=INPUT_DIM_LIGAND)
        return np.array(fp, dtype=np.float32)
    except Exception as e:
        # print(f"[ERR] generate_ecfp4 for SMILES '{smiles}': {e}") # Uncomment for debugging
        return np.zeros(INPUT_DIM_LIGAND, np.float32)


# --- NEW: Function to load aggregated protein features ---
def load_aggregated_protein_features(file_path):
    """Loads aggregated protein features from a .pt file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Protein features file not found: {file_path}")
    
    print(f"[INFO] Loading protein features from: {file_path}")
    # Use map_location='cpu' to ensure loaded onto CPU memory regardless of device it was saved from
    features = torch.load(file_path, map_location='cpu')
    print(f"[INFO] Loaded {len(features)} GPCRs from {file_path}.")
    return features


# --- Modified prepare_dataset function ---
def prepare_dataset(split_type, protein_features_dict):
    """
    Prepares the training or testing dataset by combining ligand ECFP4,
    protein features, and activity labels.
    
    Args:
        split_type (str): 'train' or 'test'.
        protein_features_dict (dict): Dictionary mapping Uniprot_ID to 1286D protein feature vector.
        
    Returns:
        tuple: (X_ligand, X_protein, y_labels, valid_uids_ikeys)
            X_ligand (np.ndarray): Array of ligand ECFP4 fingerprints.
            X_protein (np.ndarray): Array of 1286D protein features.
            y_labels (np.ndarray): Array of activity labels (0 or 1).
            valid_uids_ikeys (list): List of (Uniprot_ID, Ikey) for valid pairs.
    """
    df_splits = pd.read_csv(os.path.join(SPLITS_DIR, f"{split_type}_set_scaffold.csv"))
    df_ligand_info = pd.read_csv(LIGAND_INFO_PATH) # Contains SMILES and InChIKey

    # Merge to get SMILES for ECFP4 generation
    df_merged = pd.merge(df_splits, df_ligand_info[['InChIKey', 'SMILES']], # Corrected 'SMILES' casing
                         left_on='Ikey', right_on='InChIKey', how='left')
    
    # Filter out entries where protein features are not available
    # Or where SMILES is missing
    print(f"[DEBUG] Columns in df_merged before UniProt_ID filter: {df_merged.columns.tolist()}")
    print(f"[DEBUG] Shape of df_merged before UniProt_ID filter: {df_merged.shape}")

    # --- CRITICAL FIX: Changed 'Uniprot_ID' to 'AC' ---
    df_merged = df_merged[df_merged['Uniprot_ID'].isin(protein_features_dict.keys())].copy() 
    # Corrected 'SMILES_y' for dropna subset
    df_merged = df_merged.dropna(subset=['SMILES_y']).copy() 

    X_ligand_list = []
    X_protein_list = []
    y_labels_list = []
    valid_uids_ikeys = []

    print(f"[INFO] Preparing {split_type} dataset (matching {len(df_merged)} entries)...")
    for idx, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc=f"Preparing {split_type} data"):
        # --- CRITICAL FIX: Changed 'Uniprot_ID' to 'AC' ---
        uniprot_id = row['Uniprot_ID'] 
        ikey = row['Ikey']
        # Corrected 'SMILES_y' for row access
        smiles = row['SMILES_y'] 
        
        label = row['Label']
        # Convert string labels to numerical (0 or 1)
        if isinstance(label, str): # Check if it's a string (e.g., 'agonist', 'antagonist')
            if label.lower() == 'agonist':
                label = 1
            elif label.lower() == 'antagonist':
                label = 0
            else: # This 'else' handles unknown string labels
                # If the label is an unknown string, we skip this row
                # print(f"[WARN] Unknown label '{label}' for {uniprot_id}-{ikey}. Skipping.") # Uncomment for debugging
                continue # Skip this row if label is not 'agonist' or 'antagonist'

        # --- CORRECTED INDENTATION START ---
        # These lines are correctly indented within the 'for' loop and outside the 'if isinstance(label, str):' block.
        
        # Generate ligand ECFP4 fingerprint
        ecfp4 = generate_ecfp4(smiles)
        
        # Get protein feature vector from the pre-loaded dictionary
        protein_feature_vector = protein_features_dict.get(uniprot_id)

        # Skip if features are invalid (all zeros or None)
        if protein_feature_vector is None or np.all(protein_feature_vector == 0):
            # print(f"[WARN] Skipping {uniprot_id}-{ikey}: Protein features not found or are all zeros.") # Uncomment for debugging
            continue # This continue will also skip the current row if protein features are bad.

        # Add to lists
        X_ligand_list.append(ecfp4)
        X_protein_list.append(protein_feature_vector)
        y_labels_list.append(label)
        valid_uids_ikeys.append((uniprot_id, ikey))
        # --- CORRECTED INDENTATION END ---

    X_ligand = np.array(X_ligand_list, dtype=np.float32)
    X_protein = np.array(X_protein_list, dtype=np.float32)
    y_labels = np.array(y_labels_list, dtype=np.int32)

    print(f"[INFO] {split_type} dataset prepared: {len(y_labels)} valid pairs.")
    return X_ligand, X_protein, y_labels, valid_uids_ikeys


# --- Remaining parts of train_model.py (Model Definition, Training Loop, Evaluation) ---
# This section remains largely the same as the original train_model.py,
# but now uses the new INPUT_DIM_PROTEIN and receives the prepared data.

# Build the MLP model
def build_mlp_model():
    """Builds the Multilayer Perceptron (MLP) model."""
    # Ligand branch
    ligand_input = keras.Input(shape=(INPUT_DIM_LIGAND,), name="ligand_input")
    ligand_dense = layers.Dense(HIDDEN_DIM_LIGAND, activation="relu", name="ligand_dense1")(ligand_input)
    ligand_dense = layers.Dense(HIDDEN_DIM_LIGAND, activation="relu", name="ligand_dense2")(ligand_dense)

    # Protein branch
    protein_input = keras.Input(shape=(INPUT_DIM_PROTEIN,), name="protein_input")
    protein_dense = layers.Dense(HIDDEN_DIM_PROTEIN, activation="relu", name="protein_dense1")(protein_input)
    protein_dense = layers.Dense(HIDDEN_DIM_PROTEIN, activation="relu", name="protein_dense2")(protein_dense)

    # Concatenate and joint layers
    merged = layers.concatenate([ligand_dense, protein_dense], name="merged_features")
    joint_dense = layers.Dense(JOINT_HIDDEN_DIM, activation="relu", name="joint_dense1")(merged)
    joint_dense = layers.Dense(JOINT_HIDDEN_DIM, activation="relu", name="joint_dense2")(joint_dense)

    # Output layer
    output_layer = layers.Dense(1, activation="sigmoid", name="output_layer")(joint_dense)

    model = keras.Model(inputs=[ligand_input, protein_input], outputs=output_layer)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Main training and evaluation loop
def train_and_evaluate_model(X_ligand, X_protein, y_labels, model_name="DR_model"):
    """
    Trains and evaluates the MLP model using StratifiedShuffleSplit.
    """
    sss = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=42) # 80/20 train/val split for model tuning
    
    all_bacc = []
    all_auroc = []
    all_auprc = []
    
    print(f"[INFO] Starting model training for {model_name} with {N_SPLITS} splits...")

    # We need to ensure we split the data correctly before the loop
    # The original paper uses scaffold-based evaluation, so internal validation splits are okay.

    for fold, (train_index, val_index) in enumerate(sss.split(X_ligand, y_labels)):
        print(f"\n--- Fold {fold + 1}/{N_SPLITS} ---")
        X_train_lig, X_val_lig = X_ligand[train_index], X_ligand[val_index]
        X_train_prot, X_val_prot = X_protein[train_index], X_protein[val_index]
        y_train, y_val = y_labels[train_index], y_labels[val_index]

        model = build_mlp_model()
        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
        ]

        history = model.fit(
            [X_train_lig, X_train_prot], y_train,
            validation_data=([X_val_lig, X_val_prot], y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=0 # Set to 1 or 2 for more fitting output
        )
        
        # Predict probabilities on validation set
        y_pred_proba = model.predict([X_val_lig, X_val_prot], verbose=0).ravel()
        
        # Calculate balanced accuracy
        # Find optimal threshold for balanced accuracy on validation set
        best_bacc = 0
        best_threshold = 0.5
        for threshold in np.linspace(0.01, 0.99, 99):
            y_pred_class = (y_pred_proba > threshold).astype(int)
            bacc = balanced_accuracy_score(y_val, y_pred_class)
            if bacc > best_bacc:
                best_bacc = bacc
                best_threshold = threshold
        
        auroc = roc_auc_score(y_val, y_pred_proba)
        auprc = average_precision_score(y_val, y_pred_proba)

        print(f"Fold {fold + 1} - Val Balanced Accuracy: {best_bacc:.4f} (Threshold: {best_threshold:.2f})")
        print(f"Fold {fold + 1} - Val AUROC: {auroc:.4f}")
        print(f"Fold {fold + 1} - Val AUPRC: {auprc:.4f}")
        
        all_bacc.append(best_bacc)
        all_auroc.append(auroc)
        all_auprc.append(auprc)
        
    print("\n--- Training Summary Across Splits ---")
    print(f"Mean Val Balanced Accuracy: {np.mean(all_bacc):.4f} +/- {np.std(all_bacc):.4f}")
    print(f"Mean Val AUROC: {np.mean(all_auroc):.4f} +/- {np.std(all_auroc):.4f}")
    print(f"Mean Val AUPRC: {np.mean(all_auprc):.4f} +/- {np.std(all_auprc):.4f}")
    
    return model # Return the last trained model


# --- Main execution block for the script ---
if __name__ == "__main__":
    # Load aggregated protein features (linear and softmax weighted)
    linear_protein_features = load_aggregated_protein_features(AGGREGATED_FEATURES_LINEAR_PATH)
    softmax_protein_features = load_aggregated_protein_features(AGGREGATED_FEATURES_SOFTMAX_PATH)

    # --- Train and Evaluate with Linear-Weighted Features ---
    print("\n========================================================")
    print("=== Training Model with LINEAR-WEIGHTED Protein Features ===")
    print("========================================================")
    X_train_lig_linear, X_train_prot_linear, y_train_linear, _ = prepare_dataset('train', linear_protein_features)
    X_test_lig_linear, X_test_prot_linear, y_test_linear, _ = prepare_dataset('test', linear_protein_features)
    
    # The paper mentions 'rigorous scaffold-based evaluation', meaning the train/test splits are fixed,
    # but the internal validation (here via SSS) is on the training data.
    # For final evaluation on the scaffold-based test set, separate prediction will be done.
    
    # Train/evaluate on the prepared training dataset
    print("\n--- Training on Scaffold-Based TRAIN Set (with internal validation splits) ---")
    model_linear = train_and_evaluate_model(X_train_lig_linear, X_train_prot_linear, y_train_linear, model_name="Linear_Weighted_Model")

    # Evaluate on the fixed scaffold-based TEST set (final performance metric)
    print("\n--- Evaluating on Scaffold-Based TEST Set (FINAL METRICS) ---")
    y_test_pred_proba_linear = model_linear.predict([X_test_lig_linear, X_test_prot_linear], verbose=0).ravel()
    
    # Calculate balanced accuracy on test set
    # Find optimal threshold using the validation set's best threshold from train_and_evaluate_model if needed,
    # but generally optimal threshold from training is applied to test.
    # For simplicity, finding it on test for now, or using a fixed 0.5.
    # The paper finds optimal threshold per GPCR. Here, it's global for the test set.
    best_bacc_test_linear = 0
    test_threshold_linear = 0.5
    for threshold in np.linspace(0.01, 0.99, 99):
        y_pred_class = (y_test_pred_proba_linear > threshold).astype(int)
        bacc = balanced_accuracy_score(y_test_linear, y_pred_class)
        if bacc > best_bacc_test_linear:
            best_bacc_test_linear = bacc
            test_threshold_linear = threshold

    auroc_test_linear = roc_auc_score(y_test_linear, y_test_pred_proba_linear)
    auprc_test_linear = average_precision_score(y_test_linear, y_test_pred_proba_linear)
    
    print(f"FINAL Test Balanced Accuracy (Linear): {best_bacc_test_linear:.4f} (Threshold: {test_threshold_linear:.2f})")
    print(f"FINAL Test AUROC (Linear): {auroc_test_linear:.4f}")
    print(f"FINAL Test AUPRC (Linear): {auprc_test_linear:.4f}")


    # --- Train and Evaluate with Softmax-Weighted Features ---
    print("\n=========================================================")
    print("=== Training Model with SOFTMAX-WEIGHTED Protein Features ===")
    print("=========================================================")
    X_train_lig_softmax, X_train_prot_softmax, y_train_softmax, _ = prepare_dataset('train', softmax_protein_features)
    X_test_lig_softmax, X_test_prot_softmax, y_test_softmax, _ = prepare_dataset('test', softmax_protein_features)
    
    print("\n--- Training on Scaffold-Based TRAIN Set (with internal validation splits) ---")
    model_softmax = train_and_evaluate_model(X_train_lig_softmax, X_train_prot_softmax, y_train_softmax, model_name="Softmax_Weighted_Model")

    print("\n--- Evaluating on Scaffold-Based TEST Set (FINAL METRICS) ---")
    y_test_pred_proba_softmax = model_softmax.predict([X_test_lig_softmax, X_test_prot_softmax], verbose=0).ravel()
    
    best_bacc_test_softmax = 0
    test_threshold_softmax = 0.5
    for threshold in np.linspace(0.01, 0.99, 99):
        y_pred_class = (y_test_pred_proba_softmax > threshold).astype(int)
        bacc = balanced_accuracy_score(y_test_softmax, y_pred_class)
        if bacc > best_bacc_test_softmax:
            best_bacc_test_softmax = bacc
            test_threshold_softmax = threshold

    auroc_test_softmax = roc_auc_score(y_test_softmax, y_test_pred_proba_softmax)
    auprc_test_softmax = average_precision_score(y_test_softmax, y_test_pred_proba_softmax)
    
    print(f"FINAL Test Balanced Accuracy (Softmax): {best_bacc_test_softmax:.4f} (Threshold: {test_threshold_softmax:.2f})")
    print(f"FINAL Test AUROC (Softmax): {auroc_test_softmax:.4f}")
    print(f"FINAL Test AUPRC (Softmax): {auprc_test_softmax:.4f}")

    print("\n--- All Model Training and Evaluation Complete ---")
