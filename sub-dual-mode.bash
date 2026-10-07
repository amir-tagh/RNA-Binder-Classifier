#!/bin/bash

python binder_vs_nonbinder_rf_REVIEWER_MINPLOTS_V3.py \
	--input combined_smiles_with_MF_enhanced.csv \
	--output results_dual_mode_12_18_2025 \
	--use_smote \
	--fingerprint morgan+rdkit+atompair \
	--kfolds 10


#binder_vs_nonbinder_rf_REVIEWER_MINPLOTS_V2.py
#binder_vs_nonbinder_rf_LOBO_SMOTE_PERFOLD_V4_ENSEMBLE_V4.py
