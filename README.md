# Power Quality State Estimation for Distribution Grids based on Physics-Aware Neural Networks - Harmonic State Estimation [Code Repository]
Code for the concept paper presenting physics-aware neural networks for power quality state estimation

## Preparation
To replicate the results of the paper follow these steps:

- clone the linked repository
- Download the following files:
  - `y_mats_per_frequency.pic` (Admittance matrices per frequency in a range from 50Hz to 1000Hz in 50Hz steps)
  - `y_train.pic` (Training set)
  - `y_test.pic` (Test set)
  - `y_validation.pic` (Validation set)
- Move those files to `pqse_concept_pann/data/CigreLVDist/`
- Optionally download the weight files, in that case set load_weights to True and epochs to zero in `experiments.py`.
- If no pre-trained weights are used, you can train the model yourself. 

The code provides methods for reading in and looking at the original data sets. Transformations such as scaling and conversion to other complex representations are done at a later stage.

## Harmonic injections
Harmonic injections were modeled using spectra injecting different amounts of harmonic currents at various frequencies.
The harmonic injections scale based on load or generation profiles for a full year and the configured rated loads.

### Training set:
- "2015 Mercedes B-Class charged at 15.4A, 240V"
  - source: https://avt.inl.gov/sites/default/files/pdf/fsev/SteadyStateVehicleChargingFactSheet_2015MercedesBClass.pdf
  - nodes: 16, 37, 42
- "2015 Nissan Leaf charged at 16A, 208V"
  - source: https://avt.inl.gov/sites/default/files/pdf/fsev/SteadyStateLoadCharacterization2015Leaf.pdf
  - nodes: 17, 41
- "PV Inverter 33kW"
  - source: R. F. Arritt and R. C. Dugan, ‘Validation of Harmonic Models for PV Inverters: PV-MOD Milestone 2.8.2’, Electric Power Research Institute (EPRI), Washington, D.C. (United States), 2.8.2, Sep. 2022. doi: 10.2172/1894588
  - nodes: 9, 32, 35
  
### Test/Validation set: 
- "2014 BMW i3 charged at 15.5A, 240V", source: https://avt.inl.gov/sites/default/files/pdf/fsev/SteadyStateLoadCharacterization2014BMWi3.pdf
  - nodes: 16, 17, 37, 41, 42
- "PV Inverter 33kW", source: doi: 10.2172/1894588
  - nodes: 9, 32, 35
