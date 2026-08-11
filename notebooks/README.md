# Notebooks

This repository is command-first so it remains reproducible in CI. Suggested
notebooks to add after running the pipeline:

1. `01_data_generation_audit.ipynb` - inspect synthetic distributions.
2. `02_model_comparison.ipynb` - visualize CV metrics and calibration.
3. `03_decisioning_review.ipynb` - review tiers, premium lift, and diagnostics.

The source of truth remains the package code in `src/underwriting_engine`.
