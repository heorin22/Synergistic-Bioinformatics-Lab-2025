# Original-to-portfolio file mapping

| Original file | Portfolio location |
|---|---|
| `parse_gpcr_structures.py` | `src/gpcr_activity/structure/parse_gpcr_structures.py` |
| `utils_structure.py` | `src/gpcr_activity/structure/utils_structure.py` |
| `extract_dr.py` | `src/gpcr_activity/features/extract_structural_features.py` |
| `aggregate_weighted_features.py` | `src/gpcr_activity/features/aggregate_features.py` |
| `train_model.py` | `src/gpcr_activity/modeling/train_model.py` |
| `predict_gpcr_activity.py` | `src/gpcr_activity/modeling/predict.py` |
| `inspect_final_features.py` | `scripts/inspect_features.py` |

Not included in this GPCR repository:

- `dd.py` — CUDA/debug helper
- `fun.py` — tensor test
- `df.py` — CFTR/ESM2 analysis
- `esm2_analysis.py` — CFTR/ESM2 analysis
- `run_analysis.py` — separate alignment/ESM analysis
- `inspect_pt_file.py` — general development inspection helper
