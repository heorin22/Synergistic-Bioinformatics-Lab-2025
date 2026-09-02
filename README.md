# Data

Large raw and derived data files are intentionally excluded from this repository.

The original research pipeline used GPCR structural information from sources such as
the Protein Data Bank (PDB), AlphaFold DB, and UniProt-derived sequence/annotation data.

Expected local directories include:

- `data/cif/` — mmCIF structures
- `data/alphafold/` — AlphaFold structures
- `data/splits/` — train/test split files
- `data/processed/weighted_esm/` — per-residue ESM/structural features
- `data/processed/aggregated_weighted_esm/` — aggregated 1286-dimensional GPCR features

Only small example inputs should be committed to `data/sample/`.
