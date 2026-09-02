#!/usr/bin/env python
# coding: utf-8

"""
extract_dr.py: Modified to extract per-residue features (ESM-1b embeddings, displacements, coordinates)
for all common binding residues in GPCR structural comparisons (Apo-Ago, Apo-Ant, Ago-Ant).
This version does NOT filter for Differential Residues (DRs) at this stage; instead,
it collects data for all conformational shifts in the binding site for subsequent weighted averaging.

Integrates with parse_gpcr_structures.py for structure processing and supports AlphaFold2 (AF2) structures.
Generates new PyTorch dictionaries for detailed per-pair, per-residue feature data.
Original DR summary CSVs are still generated for backward compatibility/logging.

IMPORTANT: This script requires 'process_structure' function to be added to 'parse_gpcr_structures.py',
and that 'process_structure' returns the 'full_sequence' of the protein chain, 'uni_idxs', and 'coords'.
Additionally, 'compute_rmsd_and_disp' must be defined in 'utils_structure.py'.
"""

import os
import numpy as np 
import pandas as pd
import torch
import ast
import esm # New import for ESM model
from tqdm import tqdm # New import for progress bars

# --- CORRECTED IMPORT STATEMENTS FOR YOUR PROJECT STRUCTURE ---
# Import process_structure and fetch_uniprot_sequence from parse_gpcr_structures.py (in the same directory)
from ..structure.parse_gpcr_structures import process_structure, fetch_uniprot_sequence
# Import compute_rmsd_and_disp from utils_structure.py (now in the same directory as extract_dr.py)
from ..structure.utils_structure import compute_rmsd_and_disp
# --- END CORRECTED IMPORT STATEMENTS ---

# Global settings
# Adjust these paths as per your project setup. Using absolute paths for CIF/AF for robustness.
CIF_DIR = "data/cif"
AF_DIR = "data/alphafold"
OUTPUT_FEATURE_DIR = "data/processed/weighted_esm" # Relative to project root
OUTPUT_FINAL_DIR = "results/weighted_esm"   # Relative to project root
os.makedirs(OUTPUT_FEATURE_DIR, exist_ok=True)
os.makedirs(OUTPUT_FINAL_DIR, exist_ok=True)

# DR thresholds (kept for original summary/logging, not directly used for new feature selection)
DISP_THRESHOLD_AGO = 2.0    # Å for DR-Ago
DISP_THRESHOLD_ANT = 2.0    # Å for DR-Ant
DISP_THRESHOLD_STATE = 0.5  # Å for DR-State

# Output paths for original DR summaries (for reference/logging)
CSV_OUTPUT_PATH_AGO = os.path.join(OUTPUT_FINAL_DIR, f"Apo_Ago_Binding_Site_RMSD_Comparison_OriginalDR_{DISP_THRESHOLD_AGO}.csv")
CSV_OUTPUT_PATH_ANT = os.path.join(OUTPUT_FINAL_DIR, f"Apo_Ant_Binding_Site_RMSD_Comparison_OriginalDR_{DISP_THRESHOLD_ANT}.csv")
CSV_OUTPUT_PATH_STATE = os.path.join(OUTPUT_FINAL_DIR, f"Ago_Ant_Binding_Site_RMSD_Comparison_OriginalDR_{DISP_THRESHOLD_STATE}.csv")

# New output paths for detailed per-pair, per-residue features
OUTPUT_PER_PAIR_FEATURES_AGO = os.path.join(OUTPUT_FEATURE_DIR, "per_pair_residue_features_apo_ago.pt")
OUTPUT_PER_PAIR_FEATURES_ANT = os.path.join(OUTPUT_FEATURE_DIR, "per_pair_residue_features_apo_ant.pt")
OUTPUT_PER_PAIR_FEATURES_STATE = os.path.join(OUTPUT_FEATURE_DIR, "per_pair_residue_features_ago_ant.pt")


# --- ESM-1b Model Loading and Helper Function ---
print("[INFO] Loading ESM-1b model... This may take a moment.")
model_esm, alphabet_esm = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter_esm = alphabet_esm.get_batch_converter() # Corrected from alphabet_converter
model_esm.eval() # Set model to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_esm = model_esm.to(device)
print(f"[INFO] ESM-1b model loaded successfully on {device}.")

MAX_TOKENS_ESM = model_esm.embed_positions.max_positions

@torch.no_grad()
def get_per_residue_esm_embeddings(sequence: str, residue_numbers_1based: list) -> dict:
    """
    Compute 1280-d ESM vectors for specific 1-based residue numbers within a context window.
    Ensures subsequences fit ESM's max_positions.
    Returns a dictionary mapping residue_number (UniProt index) to its embedding.
    """
    if not sequence or not residue_numbers_1based:
        return {}

    esm_embeddings = {}
    usable_context_len = MAX_TOKENS_ESM - 2  # This is 1022, the strict max AA length for ESM-1b

    for res_num_1based in residue_numbers_1based:
        pos_0based = res_num_1based - 1 # Convert 1-based UniProt index to 0-based sequence index

        if not (0 <= pos_0based < len(sequence)):
            # Residue index out of bounds for the sequence, assign zero vector
            esm_embeddings[res_num_1based] = np.zeros(1280, dtype=np.float32)
            # print(f"[WARNING] Residue {res_num_1based} out of bounds for sequence of length {len(sequence)}.") # Debug print (uncomment if needed)
            continue

        # Define an initial window centered around the residue
        half_window = usable_context_len // 2
        
        start_0based = pos_0based - half_window
        end_0based = pos_0based + half_window + 1 # +1 because slicing end is exclusive

        # Adjust window boundaries to be within [0, len(sequence)]
        start_0based = max(0, start_0based)
        end_0based = min(len(sequence), end_0based)

        # If the window is still too short after initial boundary adjustments (e.g., at sequence ends),
        # expand it to fill the usable_context_len if possible.
        # This part ensures we get the maximum possible context.
        current_len_after_boundary_adjust = end_0based - start_0based
        if current_len_after_boundary_adjust < usable_context_len:
            if start_0based == 0: # If window starts at beginning, expand end
                end_0based = min(len(sequence), usable_context_len)
            elif end_0based == len(sequence): # If window ends at sequence end, expand start
                start_0based = max(0, len(sequence) - usable_context_len)
        
        # Final safeguard: Ensure the subsequence length never exceeds usable_context_len
        # This is the critical part to prevent the ValueError
        if (end_0based - start_0based) > usable_context_len:
            # If for any reason it's still too long, strictly cap it.
            # This prioritizes avoiding the error over perfect centering at the very edge cases.
            end_0based = start_0based + usable_context_len
            
        subsequence_for_esm = sequence[start_0based:end_0based]

        if not subsequence_for_esm:
            esm_embeddings[res_num_1based] = np.zeros(1280, dtype=np.float32)
            # print(f"[WARNING] Empty subsequence for residue {res_num_1based}.") # Debug print (uncomment if needed)
            continue
        
        # Final double check - this print should ideally never fire now
        if len(subsequence_for_esm) + 2 > MAX_TOKENS_ESM:
            print(f"[CRITICAL ERROR] Subsequence length {len(subsequence_for_esm)} for residue {res_num_1based} (orig len {len(sequence)}) is still too long for ESM max_tokens {MAX_TOKENS_ESM}. This indicates a bug in windowing logic, assign zero vector.")
            esm_embeddings[res_num_1based] = np.zeros(1280, dtype=np.float32)
            continue


        # Prepare subsequence for ESM model
        data = [("protein_subseq", subsequence_for_esm)]
        batch_labels, batch_strs, batch_tokens = batch_converter_esm(data)
        batch_tokens = batch_tokens.to(device)

        # Get representations for layer 33 (ESM's output layer)
        # Slicing [0, 1:-1, :] removes batch dimension, <bos> and <eos> tokens from ESM output
        token_representations = model_esm(batch_tokens, repr_layers=[33], return_contacts=False)["representations"][33][0, 1:-1, :]

        # Calculate the relative index of our residue within this subsequence (0-based)
        relative_pos_0based = pos_0based - start_0based

        if 0 <= relative_pos_0based < token_representations.size(0):
            embedding = token_representations[relative_pos_0based].cpu().numpy().astype(np.float32)
            esm_embeddings[res_num_1based] = embedding
        else:
            # Fallback if for some unexpected reason the residue is not in the extracted token_representations
            esm_embeddings[res_num_1based] = np.zeros(1280, dtype=np.float32)
            # print(f"[WARNING] Failed to extract ESM embedding for residue {res_num_1based} within context window.") # Debug print (uncomment if needed)

    return esm_embeddings

def extract_dr_ago():
    """
    Extract per-residue features (ESM-1b embeddings, displacements, coordinates)
    for common binding residues when comparing Apo (or AF2) vs. Agonist-bound structures.
    """
    # Paths for input CSVs are relative to the project root
    rep_apo_df = pd.read_csv("data/Representative_Apo_Structures.csv", dtype={'PDB_ID': str}) 
    rep_apo_df = rep_apo_df[rep_apo_df.Binding_Coverage.astype(float) >= 50] # Filter by binding coverage
    
    rep_map = dict(zip(rep_apo_df.UniProt_ID, rep_apo_df.PDB_ID))
    cls_df = pd.read_csv("data/GPCR_PDB_classification.csv")[['UniProt_ID', 'Ago_PDB']] 
    lig_info = pd.read_csv("data/PDB_ago_ant_chain_info_v2.csv", dtype={'PDB_ID': str}) 
    rep_chain = pd.read_csv("data/Rep_GPCR_chain.csv", dtype={'PDB_ID': str}) 

    per_gpcr_pair_features = {} 
    original_summary = [] 

    # Wrap the iteration with tqdm to see progress for each GPCR
    for _, row in tqdm(cls_df[cls_df.Ago_PDB > 0].iterrows(), # Corrected filter for Ago_PDB for this function
                       total=len(cls_df[cls_df.Ago_PDB > 0]),
                       desc="Processing Apo-Agonist GPCRs"):
        uid = row['UniProt_ID']
        print(f"\n[DEBUG] Starting processing for UniProt_ID: {uid}") # Debug print (uncomment if needed)
        
        apo_source_pdb = rep_map.get(uid) # PDB ID if experimental Apo exists
        apo_info = None
        
        # Try to process experimental Apo structure
        if apo_source_pdb:
            sub = rep_chain[(rep_chain.UniProt_ID == uid) & (rep_chain.PDB_ID == apo_source_pdb)]
            if not sub.empty:
                apo_chain = sub.loc[sub['score'].astype(float).idxmax(), 'chain_id']
                apo_info = process_structure(uid, pdb_id=apo_source_pdb, chain_id=apo_chain, is_apo=True)
        
        # Fallback to AlphaFold2 structure if experimental Apo is not found or processed
        if apo_info is None:
            af_file_path = os.path.join(AF_DIR, f"AF-{uid}-F1-model_v3.pdb")
            apo_info = process_structure(uid, af_path=af_file_path, chain_id='A', is_apo=True)
        
        if apo_info is None or 'uni_idxs' not in apo_info or 'coords' not in apo_info:
            print(f"[SKIP] {uid}: Failed to process Apo/AF2 structure or get coordinates for comparison.") # Debug print (uncomment if needed)
            continue

        # Fetch the canonical UniProt sequence for ESM embeddings for consistency
        uni_full_sequence = fetch_uniprot_sequence(uid)
        if not uni_full_sequence:
            print(f"[SKIP] {uid}: Failed to fetch canonical UniProt sequence for ESM embeddings.") # Debug print (uncomment if needed)
            continue
        
        current_gpcr_comparisons = [] 

        for _, ag_lig in lig_info[(lig_info.Entry == uid) & (lig_info.MoA.str.lower() == 'agonist')].iterrows():
            ag_pdb_id = ag_lig.PDB_ID
            
            sub2 = rep_chain[(rep_chain.UniProt_ID == uid) & (rep_chain.PDB_ID == ag_pdb_id)]
            if sub2.empty:
                print(f"[SKIP] {uid} {ag_pdb_id}: No chain info found in Rep_GPCR_chain.csv for agonist PDB.") # Debug print (uncomment if needed)
                continue
            
            ag_chain = sub2.loc[sub2['score'].astype(float).idxmax(), 'chain_id']
            ag_info = process_structure(uid, pdb_id=ag_pdb_id, chain_id=ag_chain, ligand_id=ag_lig.LIGAND_ID, is_apo=False)
            
            if ag_info is None or 'uni_idxs' not in ag_info or 'coords' not in ag_info:
                print(f"[SKIP] {uid} {ag_pdb_id}: Failed to process agonist structure or get coordinates.") # Debug print (uncomment if needed)
                continue
            
            # Identify common UniProt residue indices between Apo and Agonist structures
            common_uni_idxs = np.intersect1d(apo_info['uni_idxs'], ag_info['uni_idxs'])
            
            if len(common_uni_idxs) == 0:
                print(f"[SKIP] {uid} {ag_pdb_id}: No common binding residues between Apo and Agonist structure.") # Debug print (uncomment if needed)
                continue
            
            # Map UniProt indices to their coordinates in Apo and Agonist structures
            map_apo_coords = dict(zip(apo_info['uni_idxs'], apo_info['coords']))
            map_ag_coords = dict(zip(ag_info['uni_idxs'], ag_info['coords']))

            # Get coordinates for common residues in correct order for compute_rmsd_and_disp
            coords_apo_common = np.array([map_apo_coords[u] for u in common_uni_idxs])
            coords_ag_common = np.array([map_ag_coords[u] for u in common_uni_idxs])

            # Calculate RMSD and per-residue displacements
            overall_rmsd, displacements, transformed_coords_ag = compute_rmsd_and_disp(
                coords_apo_common,
                coords_ag_common
            )
            
            # Original DRs (for logging/summary, not used in new feature creation directly)
            diffs_original = [int(u) for u, d in zip(common_uni_idxs, displacements) if d >= DISP_THRESHOLD_AGO]
            
            original_summary.append({
                'UniProt_ID': uid,
                'Apo_Source': apo_info.get('pdb_id', 'Unknown'), # Use pdb_id from apo_info dict
                'Agonist_PDB': ag_pdb_id,
                'Common_Residues': list(common_uni_idxs),
                'Differential_Binding_Residues': diffs_original, # Original DRs
                'Overall_RMSD': overall_rmsd
            })

            # --- New Feature Collection for Weighted ESM and Spatial Features ---
            # Use the canonical UniProt sequence for ESM context
            common_esm_embeddings_map = get_per_residue_esm_embeddings(uni_full_sequence, list(common_uni_idxs))
            
            per_residue_data_for_this_pair = []
            for i, u_idx in enumerate(common_uni_idxs):
                esm_embedding_vector = common_esm_embeddings_map.get(u_idx, np.zeros(1280, dtype=np.float32))
                
                per_residue_data_for_this_pair.append({
                    'UniProt_Res_Idx': int(u_idx),
                    'Displacement': float(displacements[i]),
                    'ESM1b_Embedding': esm_embedding_vector.tolist(), # Store as list for saving
                    'Coordinate_Apo': map_apo_coords[u_idx].tolist(), # Coordinate from Apo structure
                    'Coordinate_Holo_Transformed': transformed_coords_ag[i].tolist() # Transformed coordinate from Holo (Agonist) structure
                })
            
            # Store this detailed data for the current comparison (Apo vs Agonist PDB)
            current_gpcr_comparisons.append({
                'Apo_PDB_ID': apo_info.get('pdb_id', 'Unknown'),
                'Holo_PDB_ID': ag_pdb_id,
                'Comparison_Type': 'ApoAgo',
                'Per_Residue_Data': per_residue_data_for_this_pair
            })
        
        if current_gpcr_comparisons:
            # Aggregate all comparisons for this UniProt_ID
            per_gpcr_pair_features.setdefault(uid, []).extend(current_gpcr_comparisons)

    # Save original summary (optional, for compatibility/logging)
    pd.DataFrame(original_summary).to_csv(CSV_OUTPUT_PATH_AGO, index=False)
    print(f"[INFO] Original DR-Ago summary saved to {CSV_OUTPUT_PATH_AGO}")

    # Save the new detailed per-pair, per-residue features
    torch.save(per_gpcr_pair_features, OUTPUT_PER_PAIR_FEATURES_AGO)
    print(f"[INFO] New per-pair, per-residue features for ApoAgo saved to {OUTPUT_PER_PAIR_FEATURES_AGO}")
    
    return per_gpcr_pair_features

def extract_dr_ant():
    """
    Extract per-residue features (ESM-1b embeddings, displacements, coordinates)
    for common binding residues when comparing Apo (or AF2) vs. Antagonist-bound structures.
    """
    # Paths for input CSVs are relative to the project root
    rep_apo_df = pd.read_csv("data/Representative_Apo_Structures.csv", dtype={'PDB_ID': str}) 
    rep_apo_df = rep_apo_df[rep_apo_df.Binding_Coverage.astype(float) >= 50] # Filter by binding coverage
    
    rep_map = dict(zip(rep_apo_df.UniProt_ID, rep_apo_df.PDB_ID))
    cls_df = pd.read_csv("data/GPCR_PDB_classification.csv")[['UniProt_ID', 'Ant_PDB']] # Corrected path
    lig_info = pd.read_csv("data/PDB_ago_ant_chain_info_v2.csv", dtype={'PDB_ID': str}) 
    rep_chain = pd.read_csv("data/Rep_GPCR_chain.csv", dtype={'PDB_ID': str}) 

    per_gpcr_pair_features = {} 
    original_summary = [] 

    # Wrap the iteration with tqdm to see progress for each GPCR
    for _, row in tqdm(cls_df[cls_df.Ant_PDB > 0].iterrows(), # Corrected filter for Ant_PDB
                       total=len(cls_df[cls_df.Ant_PDB > 0]),
                       desc="Processing Apo-Antagonist GPCRs"):
        uid = row['UniProt_ID']
        print(f"\n[DEBUG] Starting processing for UniProt_ID: {uid}") # Debug print (uncomment if needed)
        
        apo_source_pdb = rep_map.get(uid)
        apo_info = None
        
        if apo_source_pdb:
            sub = rep_chain[(rep_chain.UniProt_ID == uid) & (rep_chain.PDB_ID == apo_source_pdb)]
            if not sub.empty:
                apo_chain = sub.loc[sub['score'].astype(float).idxmax(), 'chain_id']
                apo_info = process_structure(uid, pdb_id=apo_source_pdb, chain_id=apo_chain, is_apo=True)
        
        if apo_info is None:
            af_file_path = os.path.join(AF_DIR, f"AF-{uid}-F1-model_v3.pdb")
            apo_info = process_structure(uid, af_path=af_file_path, chain_id='A', is_apo=True)
        
        if apo_info is None or 'uni_idxs' not in apo_info or 'coords' not in apo_info:
            print(f"[SKIP] {uid}: Failed to process Apo/AF2 structure or get coordinates for comparison.") # Debug print (uncomment if needed)
            continue

        # Fetch the canonical UniProt sequence for ESM embeddings for consistency
        uni_full_sequence = fetch_uniprot_sequence(uid)
        if not uni_full_sequence:
            print(f"[SKIP] {uid}: Failed to fetch canonical UniProt sequence for ESM embeddings.") # Debug print (uncomment if needed)
            continue

        current_gpcr_comparisons = [] 

        for _, ant_lig in lig_info[(lig_info.Entry == uid) & (lig_info.MoA.str.lower() == 'antagonist')].iterrows():
            ant_pdb_id = ant_lig.PDB_ID
            
            sub2 = rep_chain[(rep_chain.UniProt_ID == uid) & (rep_chain.PDB_ID == ant_pdb_id)]
            if sub2.empty:
                print(f"[SKIP] {uid} {ant_pdb_id}: No chain info found in Rep_GPCR_chain.csv for antagonist PDB.") # Debug print (uncomment if needed)
                continue
            
            ant_chain = sub2.loc[sub2['score'].astype(float).idxmax(), 'chain_id']
            ant_info = process_structure(uid, pdb_id=ant_pdb_id, chain_id=ant_chain, ligand_id=ant_lig.LIGAND_ID, is_apo=False)
            
            if ant_info is None or 'uni_idxs' not in ant_info or 'coords' not in ant_info:
                print(f"[SKIP] {uid} {ant_pdb_id}: Failed to process antagonist structure or get coordinates.") # Debug print (uncomment if needed)
                continue
            
            common_uni_idxs = np.intersect1d(apo_info['uni_idxs'], ant_info['uni_idxs'])
            
            if len(common_uni_idxs) == 0:
                print(f"[SKIP] {uid} {ant_pdb_id}: No common binding residues between Apo and Antagonist structure.") # Debug print (uncomment if needed)
                continue
            
            map_apo_coords = dict(zip(apo_info['uni_idxs'], apo_info['coords']))
            map_ant_coords = dict(zip(ant_info['uni_idxs'], ant_info['coords']))

            coords_apo_common = np.array([map_apo_coords[u] for u in common_uni_idxs])
            coords_ant_common = np.array([map_ant_coords[u] for u in common_uni_idxs])

            overall_rmsd, displacements, transformed_coords_ant = compute_rmsd_and_disp(
                coords_apo_common,
                coords_ant_common
            )
            
            diffs_original = [int(u) for u, d in zip(common_uni_idxs, displacements) if d >= DISP_THRESHOLD_ANT]
            
            original_summary.append({
                'UniProt_ID': uid,
                'Apo_Source': apo_info.get('pdb_id', 'Unknown'),
                'Antagonist_PDB': ant_pdb_id,
                'Common_Residues': list(common_uni_idxs),
                'Differential_Binding_Residues': diffs_original,
                'Overall_RMSD': overall_rmsd
            })

            # --- New Feature Collection ---
            common_esm_embeddings_map = get_per_residue_esm_embeddings(uni_full_sequence, list(common_uni_idxs))
            
            per_residue_data_for_this_pair = []
            for i, u_idx in enumerate(common_uni_idxs):
                esm_embedding_vector = common_esm_embeddings_map.get(u_idx, np.zeros(1280, dtype=np.float32))
                
                per_residue_data_for_this_pair.append({
                    'UniProt_Res_Idx': int(u_idx),
                    'Displacement': float(displacements[i]),
                    'ESM1b_Embedding': esm_embedding_vector.tolist(),
                    'Coordinate_Apo': map_apo_coords[u_idx].tolist(),
                    'Coordinate_Holo_Transformed': transformed_coords_ant[i].tolist() # Transformed coordinate from Holo (Antagonist) structure
                })
            
            current_gpcr_comparisons.append({
                'Apo_PDB_ID': apo_info.get('pdb_id', 'Unknown'),
                'Holo_PDB_ID': ant_pdb_id,
                'Comparison_Type': 'ApoAnt',
                'Per_Residue_Data': per_residue_data_for_this_pair
            })
        
        if current_gpcr_comparisons:
            per_gpcr_pair_features.setdefault(uid, []).extend(current_gpcr_comparisons)

    pd.DataFrame(original_summary).to_csv(CSV_OUTPUT_PATH_ANT, index=False)
    print(f"[INFO] Original DR-Ant summary saved to {CSV_OUTPUT_PATH_ANT}")

    torch.save(per_gpcr_pair_features, OUTPUT_PER_PAIR_FEATURES_ANT)
    print(f"[INFO] New per-pair, per-residue features for ApoAnt saved to {OUTPUT_PER_PAIR_FEATURES_ANT}")
    
    return per_gpcr_pair_features


def extract_dr_state():
    """
    Extract per-residue features (ESM-1b embeddings, displacements, coordinates)
    for common binding residues when comparing Agonist-bound vs. Antagonist-bound structures.
    """
    # Paths for input CSVs are relative to the project root
    cls_df = pd.read_csv("data/GPCR_PDB_classification.csv")[['UniProt_ID', 'Ago_PDB', 'Ant_PDB']] 
    lig_info = pd.read_csv("data/PDB_ago_ant_chain_info_v2.csv", dtype={'PDB_ID': str}) 
    rep_chain = pd.read_csv("data/Rep_GPCR_chain.csv", dtype={'PDB_ID': str}) 

    per_gpcr_pair_features = {} 
    original_summary = [] 

    # Wrap the iteration with tqdm to see progress for each GPCR
    for _, row in tqdm(cls_df[(cls_df.Ago_PDB > 0) & (cls_df.Ant_PDB > 0)].iterrows(),
                       total=len(cls_df[(cls_df.Ago_PDB > 0) & (cls_df.Ant_PDB > 0)]),
                       desc="Processing Agonist-Antagonist GPCRs"):
        uid = row['UniProt_ID']
        print(f"\n[DEBUG] Starting processing for UniProt_ID: {uid}") # Debug print (uncomment if needed)
        
        ago_pdbs_info = lig_info[(lig_info.Entry == uid) & (lig_info.MoA.str.lower() == 'agonist')]
        ant_pdbs_info = lig_info[(lig_info.Entry == uid) & (lig_info.MoA.str.lower() == 'antagonist')]

        if ago_pdbs_info.empty or ant_pdbs_info.empty:
            print(f"[SKIP] {uid}: Missing agonist or antagonist PDBs for state comparison.") # Debug print (uncomment if needed)
            continue

        # Fetch the canonical UniProt sequence for ESM embeddings for state comparison
        uni_full_sequence = fetch_uniprot_sequence(uid)
        if not uni_full_sequence:
            print(f"[SKIP] {uid}: Failed to fetch canonical UniProt sequence for ESM embeddings for state comparison.") # Debug print (uncomment if needed)
            continue

        current_gpcr_comparisons = [] 

        for _, ago_lig in ago_pdbs_info.iterrows():
            pdb_ago = ago_lig.PDB_ID
            sub_ago = rep_chain[(rep_chain.UniProt_ID == uid) & (rep_chain.PDB_ID == pdb_ago)]
            if sub_ago.empty:
                print(f"[SKIP] {uid} {pdb_ago}: No chain info for agonist PDB for state comparison.") # Debug print (uncomment if needed)
                continue
            chain_ago = sub_ago.loc[sub_ago['score'].astype(float).idxmax(), 'chain_id']
            ago_info = process_structure(uid, pdb_id=pdb_ago, chain_id=chain_ago, ligand_id=ago_lig.LIGAND_ID, is_apo=False)
            if ago_info is None or 'uni_idxs' not in ago_info or 'coords' not in ago_info:
                print(f"[SKIP] {uid} {pdb_ago}: Failed to process agonist structure or get coordinates for state comparison.") # Debug print (uncomment if needed)
                continue

            for _, ant_lig in ant_pdbs_info.iterrows():
                pdb_ant = ant_lig.PDB_ID
                sub_ant = rep_chain[(rep_chain.UniProt_ID == uid) & (rep_chain.PDB_ID == pdb_ant)]
                if sub_ant.empty:
                    print(f"[SKIP] {uid} {pdb_ago} vs {pdb_ant}: No chain info for antagonist PDB for state comparison.") # Debug print (uncomment if needed)
                    continue
                chain_ant = sub_ant.loc[sub_ant['score'].astype(float).idxmax(), 'chain_id']
                ant_info = process_structure(uid, pdb_id=pdb_ant, chain_id=chain_ant, ligand_id=ant_lig.LIGAND_ID, is_apo=False)
                if ant_info is None or 'uni_idxs' not in ant_info or 'coords' not in ant_info:
                    print(f"[SKIP] {uid} {pdb_ago} vs {pdb_ant}: Failed to process antagonist structure or get coordinates for state comparison.") # Debug print (uncomment if needed)
                    continue
                
                common_uni_idxs = np.intersect1d(ago_info['uni_idxs'], ant_info['uni_idxs'])
                
                if len(common_uni_idxs) < 3: # Keep the original minimum common residues check
                    print(f"[SKIP] {uid} {pdb_ago} vs {pdb_ant}: Too few common residues ({len(common_uni_idxs)}) for state comparison.") # Debug print (uncomment if needed)
                    continue
                
                map_ago_coords = dict(zip(ago_info['uni_idxs'], ago_info['coords']))
                map_ant_coords = dict(zip(ant_info['uni_idxs'], ant_info['coords']))
                
                coords_ago_common = np.array([map_ago_coords[u] for u in common_uni_idxs])
                coords_ant_common = np.array([map_ant_coords[u] for u in common_uni_idxs])

                overall_rmsd, displacements, transformed_coords_ant = compute_rmsd_and_disp(
                    coords_ago_common,
                    coords_ant_common
                )
                
                diffs_original = [int(u) for u, d in zip(common_uni_idxs, displacements) if d >= DISP_THRESHOLD_STATE]
                
                original_summary.append({
                    'UniProt_ID': uid,
                    'Agonist_PDB': pdb_ago,
                    'Antagonist_PDB': pdb_ant,
                    'Common_Residues': list(common_uni_idxs),
                    'Differential_Binding_Residues': diffs_original,
                    'Overall_RMSD': overall_rmsd
                })

                # --- New Feature Collection ---
                common_esm_embeddings_map = get_per_residue_esm_embeddings(uni_full_sequence, list(common_uni_idxs))
                
                per_residue_data_for_this_pair = []
                for i, u_idx in enumerate(common_uni_idxs):
                    esm_embedding_vector = common_esm_embeddings_map.get(u_idx, np.zeros(1280, dtype=np.float32))
                    
                    per_residue_data_for_this_pair.append({
                        'UniProt_Res_Idx': int(u_idx),
                        'Displacement': float(displacements[i]),
                        'ESM1b_Embedding': esm_embedding_vector.tolist(),
                        'Coordinate_Ago': map_ago_coords[u_idx].tolist(), # Coordinate from Agonist structure
                        'Coordinate_Ant_Transformed': transformed_coords_ant[i].tolist() # Transformed coordinate from Antagonist structure
                    })
                
                current_gpcr_comparisons.append({
                    'Agonist_PDB_ID': pdb_ago,
                    'Antagonist_PDB_ID': pdb_ant,
                    'Comparison_Type': 'AgoAnt',
                    'Per_Residue_Data': per_residue_data_for_this_pair
                })
        
        if current_gpcr_comparisons:
            per_gpcr_pair_features.setdefault(uid, []).extend(current_gpcr_comparisons)

    pd.DataFrame(original_summary).to_csv(CSV_OUTPUT_PATH_STATE, index=False)
    print(f"[INFO] Original DR-State summary saved to {CSV_OUTPUT_PATH_STATE}")

    torch.save(per_gpcr_pair_features, OUTPUT_PER_PAIR_FEATURES_STATE)
    print(f"[INFO] New per-pair, per-residue features for AgoAnt saved to {OUTPUT_PER_PAIR_FEATURES_STATE}")
    
    return per_gpcr_pair_features


# --- Main Execution Block (Modified) ---
if __name__ == "__main__":
    print("--- Starting Per-Pair, Per-Residue Feature Extraction ---")
    
    print("\nExtracting features for Apo-Agonist comparisons...")
    #extract_dr_ago()
    
    print("\nExtracting features for Apo-Antagonist comparisons...")
    extract_dr_ant()
    
    print("\nExtracting features for Agonist-Antagonist comparisons...")
    extract_dr_state()
    
    print("\n--- Feature Extraction Complete ---")
    print("New per-pair, per-residue feature files saved to:")
    print(f"- {OUTPUT_PER_PAIR_FEATURES_AGO}")
    print(f"- {OUTPUT_PER_PAIR_FEATURES_ANT}")
    print(f"- {OUTPUT_PER_PAIR_FEATURES_STATE}")
    
    # The original aggregate_dr_sets() and merge_dr_coordinates()
    # are removed from this main block as they relate to the old DR definition.
    # A new script will handle the aggregation of these new features.
