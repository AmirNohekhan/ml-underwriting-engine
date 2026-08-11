# Model Card

## Intended Use

Estimate expected loss for synthetic property/rental underwriting examples and
convert the estimate into risk tiers and technical premiums. The repository is a
portfolio demonstration and must not be used for real underwriting decisions.

## Data

All records are generated synthetically. The generator includes geography-level
correlation, exposure weighting, property characteristics, applicant attributes,
macroeconomic variables, and a compound claim frequency/severity target process.

## Evaluation

Primary metrics are exposure-weighted Tweedie deviance, exposure-weighted MAE,
and calibration bias ratio. Validation uses geography-grouped folds to reduce
leakage from correlated local risk factors.

## Limitations

Synthetic data cannot establish regulatory acceptability, causal validity, or
real-world fairness. The fairness report is included to demonstrate a review
pattern, not to certify that a real model is compliant.
