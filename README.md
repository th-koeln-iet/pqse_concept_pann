# pqse_concept_pann
Code for the concept paper presenting physics-aware neural networks for power quality state estimation

## Preparation
Download the following files:
- `y_mats_per_frequency.pic` (Admittance matrices per frequency in a range from 50Hz to 1000Hz in 50Hz steps)
- `y_train.pic` (Training set)
- `y_test.pic` (Test set)
- `y_validation.pic` (Validation set)

Move those files to `pqse_concept_pann/data/CigreLVDist/`

Optionally download the weight files, in that case set load_weights to True and epochs to zero in `experiments.py`.
If no weights are downloaded, you can train the model yourself. 
