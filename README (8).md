# GPCR Ligand Activity Prediction

A research pipeline for predicting **agonist vs antagonist activity** of ligand–GPCR pairs by integrating ligand chemical fingerprints, protein language model embeddings, and GPCR structural features.

> **Research context:** This repository is a portfolio-oriented reorganisation of code developed during a bioinformatics research internship. The original research code has been cleaned and restructured here for readability. Large datasets, intermediate tensors, and lab-specific files are not included.

## Overview

The pipeline combines three information sources:

1. **Ligand chemistry** — 1024-bit ECFP4 fingerprints generated from SMILES using RDKit.
2. **Protein sequence representation** — residue-level 1280-dimensional ESM-1b embeddings.
3. **GPCR structural information** — conformational displacement and spatial features derived from structural comparisons.

These features are combined in a dual-branch neural network to classify ligand–GPCR pairs as agonist or antagonist.

## Pipeline

```text
PDB / AlphaFold2 structures
            |
            v
   GPCR structure processing
            |
            v
Apo / Agonist / Antagonist comparisons
            |
            +----> residue displacement + geometry
            |
            +----> ESM-1b residue embeddings
                         |
                         v
              1286-d GPCR feature
                         |
                         +-------------------+
                                             |
SMILES --> ECFP4 (1024-d) --> ligand branch  |
                                             v
                                  feature concatenation
                                             |
                                             v
                                             MLP
                                             |
                                             v
                                  Agonist / Antagonist
```

## Repository structure

```text
gpcr-ligand-activity-prediction/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── data/
│   ├── README.md
│   └── sample/
├── docs/
├── results/
│   ├── figures/
│   └── metrics/
├── scripts/
│   └── inspect_features.py
└── src/
    └── gpcr_activity/
        ├── structure/
        │   ├── parse_gpcr_structures.py
        │   └── utils_structure.py
        ├── features/
        │   ├── extract_structural_features.py
        │   └── aggregate_features.py
        └── modeling/
            ├── train_model.py
            └── predict.py
```

## Methods

### 1. GPCR structure processing

The structural pipeline downloads and parses PDB/mmCIF structures, supports AlphaFold structures when required, identifies relevant chains and ligand-binding residues, and maps structural residue numbering to canonical UniProt positions.

### 2. Structural comparison and ESM-1b features

Common binding-site residues are identified across:

- apo vs agonist-bound structures,
- apo vs antagonist-bound structures,
- agonist-bound vs antagonist-bound structures.

After structural superposition, per-residue displacement is calculated. Residue-level ESM-1b embeddings are generated for the corresponding UniProt positions.

### 3. Protein-level feature aggregation

Residue-level information is aggregated into a fixed-length GPCR representation.

The resulting protein vector contains:

- **1280 ESM-1b dimensions**
- **6 spatial / structural dimensions**

for a total of **1286 protein features**.

The code includes both uniform/linear and softmax-based displacement weighting strategies.

### 4. Ligand representation

Ligands are represented using **1024-bit ECFP4 (Morgan) fingerprints** generated from SMILES with RDKit.

### 5. Agonist / antagonist prediction

The model contains separate ligand and protein branches:

```text
ECFP4 (1024)  -> Dense -> Dense --+
                                   |
                                   +-> Concatenate -> Dense -> Dense -> Sigmoid
                                   |
GPCR (1286)   -> Dense -> Dense --+
```

The training code evaluates the model using metrics including:

- balanced accuracy
- ROC-AUC
- average precision / PR-AUC

## Installation

```bash
git clone <your-repository-url>
cd gpcr-ligand-activity-prediction

python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
pip install -e .
```

## Example workflow

The complete research workflow depends on structural and processed input datasets that are not distributed in this portfolio repository.

With the required local data available, the main stages are organised as:

```bash
python -m gpcr_activity.structure.parse_gpcr_structures
python -m gpcr_activity.features.extract_structural_features
python -m gpcr_activity.features.aggregate_features
python -m gpcr_activity.modeling.train_model
python -m gpcr_activity.modeling.predict
```

## Data availability

Large structural files, PyTorch feature tensors, trained model checkpoints, and lab-specific datasets are excluded from GitHub.

See [`data/README.md`](data/README.md) for the expected local layout.

## Results

Add the final experimental metrics before making the repository public.

A useful summary table would be:

| Protein representation | Balanced accuracy | ROC-AUC | PR-AUC |
|---|---:|---:|---:|
| Uniform-weighted ESM + structure | TBD | TBD | TBD |
| Softmax-weighted ESM + structure | TBD | TBD | TBD |

Recommended figures:

- ROC curve
- precision-recall curve
- simplified model architecture
- one representative GPCR structural comparison

## Tools

Python · PyTorch · TensorFlow/Keras · ESM-1b · RDKit · Biopython · scikit-learn · NumPy · pandas

## Research context and contribution

This code was developed as part of a computational biology research project on ligand-dependent GPCR activity prediction. The work explored the integration of structural conformational information with protein language model representations and ligand chemical fingerprints.

This public repository is intended as a **research portfolio** rather than a complete reproduction package. Lab-specific datasets and large generated artifacts have therefore been omitted.

## License

Before adding an open-source license, confirm that the underlying research code and data can be shared publicly under your lab or institution's policies.
