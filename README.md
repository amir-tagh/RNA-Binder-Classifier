# RNA Binder vs Non-Binder Random Forest Classifier

A cheminformatics and machine-learning workflow for classifying small molecules as **RNA binders** or **non-binders** from molecular fingerprints using a Random Forest classifier.

The current implementation validates SMILES, generates one or more molecular fingerprint representations, performs Random-Forest-based feature selection, trains and tunes a classifier using stratified cross-validation, evaluates predictive performance, and generates model-interpretability and chemical-space visualizations.

## Overview

The workflow is designed for binary classification datasets containing molecular structures encoded as SMILES and a binary activity label:

- `1` = binder
- `0` = non-binder

The main analysis pipeline is:

```text
Input CSV
   |
   v
SMILES validation
   |
   v
Molecular fingerprints
   |
   +-- Morgan
   +-- MACCS
   +-- RDKit
   +-- AtomPair
   +-- fingerprint combinations
   |
   v
Random-Forest feature selection
   |
   v
Train/test split
   |
   v
Stratified K-fold Random Forest training
   |
   +-- SMOTE within CV training folds
   +-- randomized hyperparameter search
   |
   v
Model evaluation
   |
   +-- ROC-AUC
   +-- confidence-threshold tuning
   +-- confusion matrix
   +-- average cross-validated ROC curve
   |
   v
Model interpretation / visualization
   |
   +-- SHAP analysis
   +-- t-SNE chemical-space visualization
   |
   v
Output files and figures
```

## Repository files

### `binder_vs_nonbinder_rf_REVIEWER_MINPLOTS_V3.py`

Main Python workflow for preprocessing, fingerprint generation, feature selection, Random Forest training, hyperparameter optimization, evaluation, SHAP interpretation, and visualization.

### `sub-dual-mode.bash`

Example launcher showing how to run the classifier with a combined fingerprint representation:

```bash
python binder_vs_nonbinder_rf_REVIEWER_MINPLOTS_V3.py \
    --input combined_smiles_with_MF_enhanced.csv \
    --output results_dual_mode_12_18_2025 \
    --use_smote \
    --fingerprint morgan+rdkit+atompair \
    --kfolds 10
```

## Requirements

The script uses the following Python packages:

- Python
- pandas
- NumPy
- RDKit
- scikit-learn
- imbalanced-learn
- SHAP
- matplotlib
- seaborn
- tqdm

A typical Conda installation is:

```bash
conda create -n rna_rf python=3.10 -y
conda activate rna_rf

conda install -c conda-forge \
    rdkit pandas numpy scikit-learn imbalanced-learn \
    matplotlib seaborn tqdm shap -y
```

Exact package versions are not pinned by the current repository. For reproducible analyses, creating an `environment.yml` or pinned requirements file is recommended.

## Input format

The program expects a CSV file containing at least these two columns:

```text
SMILES,Label
CCOC(=O)c1ccccc1,0
CN1CCC(c2ccccc2)CC1,1
...
```

### Required columns

| Column | Description |
|---|---|
| `SMILES` | Molecular structure encoded as a SMILES string |
| `Label` | Binary class label: `0` for non-binder and `1` for binder |

SMILES are parsed with RDKit. Invalid structures are excluded from subsequent analysis and written to:

```text
invalid_smiles.log
```

## Usage

Basic syntax:

```bash
python binder_vs_nonbinder_rf_REVIEWER_MINPLOTS_V3.py \
    --input INPUT.csv \
    --output OUTPUT_DIRECTORY \
    --fingerprint FINGERPRINT \
    --kfolds 5
```

Example:

```bash
python binder_vs_nonbinder_rf_REVIEWER_MINPLOTS_V3.py \
    --input combined_smiles_with_MF_enhanced.csv \
    --output results \
    --fingerprint morgan \
    --kfolds 5
```

### Combined fingerprints

Multiple fingerprints can be concatenated by separating their names with `+`:

```bash
python binder_vs_nonbinder_rf_REVIEWER_MINPLOTS_V3.py \
    --input combined_smiles_with_MF_enhanced.csv \
    --output results \
    --fingerprint morgan+rdkit+atompair \
    --kfolds 10
```

The supplied `sub-dual-mode.bash` uses this combined representation.

## Command-line arguments

| Argument | Required | Default | Description |
|---|---:|---:|---|
| `--input` | Yes | — | Input CSV containing `SMILES` and `Label` |
| `--output` | Yes | — | Root output directory |
| `--fingerprint` | No | `morgan` | Fingerprint type or `+`-separated combination |
| `--kfolds` | No | `5` | Number of stratified cross-validation folds |
| `--use_smote` | No | False | Accepted by the current CLI; see implementation note below |

## Molecular fingerprints

The current script implements four fingerprint types.

### Morgan fingerprint

```text
radius = 3
nBits = 512
```

Selected with:

```bash
--fingerprint morgan
```

### MACCS keys

RDKit MACCS structural keys:

```bash
--fingerprint maccs
```

### RDKit fingerprint

RDKit topological/path fingerprint:

```bash
--fingerprint rdkit
```

### AtomPair fingerprint

Hashed AtomPair bit vector:

```text
nBits = 512
```

Selected with:

```bash
--fingerprint atompair
```

### Fingerprint ensembles

Fingerprints can be concatenated into a single feature matrix:

```bash
--fingerprint morgan+rdkit+atompair
```

For example, the supplied launcher uses:

```text
Morgan + RDKit + AtomPair
```

before feature selection.

## Feature selection

Before model training, the script fits a Random Forest classifier to the fingerprint matrix and applies:

```python
SelectFromModel(..., threshold="median")
```

Features with sufficient Random Forest importance are retained, reducing the dimensionality of the fingerprint representation before the train/test split and downstream modeling.

## Train/test split

The selected feature matrix is divided into:

```text
80% training
20% testing
```

using:

```python
train_test_split(..., test_size=0.2, random_state=42)
```

## Class imbalance handling

The code contains utilities for both **SMOTE + Tomek Links** and SMOTE.

Within `train_model_kfold()`, SMOTE is applied to each cross-validation training fold using:

```python
SMOTE(
    sampling_strategy=0.75,
    random_state=42
)
```

This means the minority class is synthetically oversampled within the training partition before fitting the Random Forest.

### Important note about `--use_smote`

The current command-line parser accepts:

```bash
--use_smote
```

but the current training function applies SMOTE internally regardless of whether this flag is supplied. Therefore, in this version the flag does **not** toggle the fold-level SMOTE behavior.

This should be considered when reproducing analyses or extending the code.

## Random Forest model

The base estimator is:

```python
RandomForestClassifier(
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
    n_estimators=100
)
```

The code evaluates the model with stratified K-fold cross-validation.

## Hyperparameter optimization

A randomized search explores combinations of:

- number of estimators
- maximum tree depth
- minimum samples required to split
- minimum samples per leaf
- number of features considered at a split
- bootstrap sampling

The search space is currently:

```python
{
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [2, 4, 6],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True]
}
```

Optimization uses ROC-AUC as the scoring metric.

The resulting parameters and average cross-validation AUC are written to:

```text
best_params.txt
```

## Classification threshold optimization

Instead of assuming a fixed probability cutoff of `0.5`, the workflow searches thresholds from approximately `0.05` to `0.94` and selects the threshold maximizing the F1 score on the held-out test set.

This threshold is subsequently used for the confusion-matrix analysis.

The source code also defines two LOBO-oriented thresholds:

```text
Screening threshold    = 0.25
Confirmation threshold = 0.37
```

but the dual-mode LOBO execution block is disabled in the current reviewer-minimal main workflow.

## Model evaluation

### ROC-AUC

ROC-AUC is calculated during stratified cross-validation.

### Average ROC curve

The script constructs an average ROC curve across stratified folds.

The plot includes:

- mean ROC curve
- chance diagonal
- pointwise standard-error ribbon
- mean AUC
- AUC variability

Output:

```text
average_auc_roc.png
```

### Confusion matrix

Predictions are converted to binary classes using the optimized confidence threshold.

Output:

```text
confusion_matrix.png
```

## SHAP model interpretation

The workflow uses `shap.TreeExplainer` to interpret the trained Random Forest.

Generated outputs include:

```text
shap_values.csv
mean_shap_importance.csv
shap_summary_bar.png
shap_summary_beeswarm.png
```

The mean absolute SHAP magnitude is used to rank fingerprint features by contribution to model predictions.

Because the input features are fingerprint bits rather than named physicochemical descriptors, the generated feature names follow the form:

```text
Feature_0
Feature_1
Feature_2
...
```

For combined fingerprints, these indices refer to the post-feature-selection feature matrix and are not automatically mapped back to specific chemical substructures by the current script.

## t-SNE chemical-space visualization

The selected fingerprint representation is projected into two dimensions with t-SNE.

Binders and non-binders are displayed separately to visualize their distribution in fingerprint space.

Output:

```text
tsne_plot.png
```

## LOBO evaluation

The code contains a **Leave-One-Binder-Out (LOBO)** evaluation function.

For each known binder, the function:

1. removes that binder from training,
2. retrains the model,
3. predicts the held-out binder,
4. records its predicted probability,
5. applies a specified decision threshold.

Two modes are defined:

```text
Screening mode
Confirmation mode
```

and LOBO results can be written to separate directories containing:

```text
lobo_results.csv
lobo_confusion_matrix.png
lobo_roc_curve.png
```

However, the LOBO execution block in the current main script is intentionally commented out. Therefore, these files are **not generated by default**.

To use LOBO, the corresponding block near the end of the Python script must be enabled.

## Output structure

For a command such as:

```bash
python binder_vs_nonbinder_rf_REVIEWER_MINPLOTS_V3.py \
    --input dataset.csv \
    --output results \
    --fingerprint morgan+rdkit+atompair \
    --kfolds 10
```

the program creates a fingerprint-specific directory:

```text
results/
└── morgan+rdkit+atompair/
    ├── average_auc_roc.png
    ├── best_params.txt
    ├── confusion_matrix.png
    ├── mean_shap_importance.csv
    ├── shap_summary_bar.png
    ├── shap_summary_beeswarm.png
    ├── shap_values.csv
    └── tsne_plot.png
```

If invalid SMILES are encountered, `invalid_smiles.log` is written to the root output directory.

Exact outputs can vary if optional or currently disabled analysis blocks are modified.

## Example launcher

Make the supplied shell script executable:

```bash
chmod +x sub-dual-mode.bash
```

Then run:

```bash
./sub-dual-mode.bash
```

Its current configuration is:

```bash
python binder_vs_nonbinder_rf_REVIEWER_MINPLOTS_V3.py \
    --input combined_smiles_with_MF_enhanced.csv \
    --output results_dual_mode_12_18_2025 \
    --use_smote \
    --fingerprint morgan+rdkit+atompair \
    --kfolds 10
```

The input CSV must be available in the working directory unless an absolute or relative path is supplied.

## Reproducibility

Several parts of the workflow use:

```text
random_state = 42
```

including the train/test split, Random Forest, SMOTE, stratified cross-validation, randomized hyperparameter search, and t-SNE initialization.

This improves reproducibility across repeated runs using the same software environment and input data.

## Current implementation notes

Users should be aware of several details in the present version:

1. `--use_smote` is parsed but does not currently control whether fold-level SMOTE is executed.
2. Feature selection is performed before the train/test split in the current workflow. For a strictly leakage-controlled benchmark, feature selection should ideally be fit only on training data, preferably within a scikit-learn/imblearn pipeline inside each cross-validation fold.
3. The randomized hyperparameter search is fit on the data supplied to `train_model_kfold()` without the same explicit per-fold SMOTE transformation used in the preceding manual CV loop.
4. LOBO functionality exists but is disabled in the default main execution block.
5. The current code uses fixed fingerprint settings for Morgan and AtomPair fingerprints rather than exposing radius/bit length as command-line options.
6. Fingerprint feature indices in SHAP outputs are not currently translated back into chemically interpretable substructures.

These points are useful targets for future development.

## Suggested future improvements

Potential extensions include:

- leakage-safe preprocessing using `imblearn.pipeline.Pipeline`
- command-line control of Morgan radius and fingerprint bit length
- nested cross-validation for hyperparameter optimization
- explicit precision-recall/AUPRC reporting for imbalanced datasets
- Matthews correlation coefficient, balanced accuracy, sensitivity, specificity, and F1 reporting
- scaffold-based or cluster-based train/test splitting
- external validation datasets
- probability calibration
- automated fingerprint benchmarking
- explicit LOBO command-line mode
- model serialization for inference on external libraries
- mapping important Morgan bits back to molecular environments
- UMAP chemical-space visualization
- pinned Conda environment for reproducibility
- automated unit tests and continuous integration

## Citation

If this repository supports a publication, please cite the corresponding manuscript once citation information is available.

## License

No software license is specified by the current source files. Add an explicit `LICENSE` before public distribution so users know how the code may be used, modified, and redistributed.

## Disclaimer

This code is intended for research use. Predictive classification scores should not be interpreted as experimental evidence of RNA binding without appropriate validation.
