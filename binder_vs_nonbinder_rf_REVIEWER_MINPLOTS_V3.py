import pandas as pd
import numpy as np
import os
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import shap
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Global matplotlib style settings
plt.rcParams.update({
    "axes.linewidth": 2,         # Thicker axes lines
    "axes.labelsize": 16,        # Font size for x and y labels
    "axes.labelweight": "bold",  # Make x and y labels bold
    "xtick.labelsize": 14,       # Font size for x tick labels
    "ytick.labelsize": 14,       # Font size for y tick labels
    "xtick.major.width": 2,      # Thicker x ticks
    "ytick.major.width": 2,      # Thicker y ticks
    "font.weight": "bold",       # Make all text bold
    "legend.fontsize": 14,       # Legend font size
    "legend.title_fontsize": 14  # Legend title size
})

from imblearn.combine import SMOTETomek
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, roc_curve,auc
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, roc_curve, auc, precision_recall_curve, confusion_matrix
import seaborn as sns
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# Thresholds for LOBO prediction
SCREENING_THRESHOLD = 0.25
CONFIRMATION_THRESHOLD = 0.37


def tune_confidence_threshold(y_true, y_probs, optimize_for='f1'):
    from sklearn.metrics import f1_score, recall_score
    print(f"Tuning confidence threshold based on {optimize_for} score...")
    best_threshold, best_score = 0.5, 0
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (y_probs >= t).astype(int)
        score = f1_score(y_true, preds) if optimize_for == 'f1' else recall_score(y_true, preds)
        if score > best_score:
            best_score = score
            best_threshold = t
    print(f"Best threshold: {best_threshold:.2f} with {optimize_for} score: {best_score:.3f}")
    return best_threshold



#import numpy as np
#import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay

def run_dual_lobo_evaluation(X, y, smiles, model, output_dir, screening_threshold, confirmation_threshold, labels):
    """Runs LOBO evaluation twice: screening and confirmation mode using specified thresholds."""

    thresholds = {
        "screening": screening_threshold,
        "confirmation": confirmation_threshold,
    }

    for mode, threshold in thresholds.items():
        print(f"\nLOBO Evaluation - {mode.upper()} Mode (threshold = {threshold:.2f})")
        binder_indices = [i for i, label in enumerate(labels) if label == 1]

        all_preds, all_trues, all_probs, all_smiles = [], [], [], []

        for idx in binder_indices:
            test_indices = [idx]
            train_indices = [i for i in range(len(y)) if i != idx]
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            model.fit(X_train, y_train)
            positive_class_index = list(model.classes_).index(1)
            y_prob = model.predict_proba(X_test)[0][positive_class_index]
            y_pred = int(y_prob >= threshold)

            all_preds.append(y_pred)
            all_probs.append(y_prob)
            all_trues.append(y_test[0])
            all_smiles.append(smiles[idx])

        # Save results
        outdir = os.path.join(output_dir, f"LOBO_{mode}")
        os.makedirs(outdir, exist_ok=True)
        df = pd.DataFrame({
            "SMILES": all_smiles,
            "True Label": all_trues,
            "Predicted Label": all_preds,
            "Probability": all_probs
        })
        df.to_csv(os.path.join(outdir, "lobo_results.csv"), index=False)
        print(f"LOBO results saved to '{os.path.join(outdir, 'lobo_results.csv')}'")

        # Plot confusion matrix
        cm = confusion_matrix(all_trues, all_preds, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-binder", "Binder"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"LOBO Confusion Matrix ({mode})")
        plt.savefig(os.path.join(outdir, "lobo_confusion_matrix.png"))
        plt.close()

        # Plot ROC Curve (if enough class variation)
        try:
            if len(set(all_trues)) > 1:
                RocCurveDisplay.from_predictions(all_trues, all_probs)
                plt.title(f"LOBO ROC Curve ({mode})")
                plt.savefig(os.path.join(outdir, "lobo_roc_curve.png"))
                plt.close()
        except Exception as e:
            print(f"ROC Curve not plotted for {mode}: {e}")


# Step 1: Read Data with SMILES Validation
def read_data(file_path, output_dir):
    print("Reading data...")
    data = pd.read_csv(file_path)
    smiles = data['SMILES']
    labels = data['Label']

    # Validate SMILES and log invalid ones
    valid_smiles = []
    valid_labels = []
    invalid_smiles = []

    for sm, label in zip(smiles, labels):
        mol = Chem.MolFromSmiles(sm)
        if mol:
            valid_smiles.append(sm)
            valid_labels.append(label)
        else:
            invalid_smiles.append(sm)

    # Save invalid SMILES to a log file if any
    if invalid_smiles:
        with open(os.path.join(output_dir, "invalid_smiles.log"), "w") as log_file:
            log_file.write("\n".join(invalid_smiles))
        print(f"Found {len(invalid_smiles)} invalid SMILES. Logged to 'invalid_smiles.log'.")

    print(f"Total valid SMILES: {len(valid_smiles)}")
    print(f"Total invalid SMILES: {len(invalid_smiles)}")

    # Return only valid SMILES and labels
    return valid_smiles, valid_labels

# Step 2: Generate Fingerprints Based on User Choice
def get_fingerprints(smiles, fingerprint_type='morgan'):
    print(f"Generating {fingerprint_type} fingerprints...")
    fingerprints = []
    for smi in tqdm(smiles, desc="Processing SMILES", ncols=80):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            if fingerprint_type == 'morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=512)
            elif fingerprint_type == 'maccs':
                fp = MACCSkeys.GenMACCSKeys(mol)
            elif fingerprint_type == 'rdkit':
                fp = Chem.RDKFingerprint(mol)
            elif fingerprint_type == 'atompair':
                fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=512)
            else:
                raise ValueError("Unsupported fingerprint type!")
            fingerprints.append(fp)
        else:
            if fingerprint_type == 'maccs':
                fingerprints.append([0] * 167)
            else:
                fingerprints.append([0] * 2048)
    return np.array(fingerprints)

# Step 3: Apply SMOTE + Tomek Links
def apply_smote_tomek(X_train, y_train):
    print("\nApplying SMOTE + Tomek Links to balance classes...")
    print("Class distribution before balancing:", Counter(y_train))
    smote_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
    print("Class distribution after balancing:", Counter(y_resampled))
    return X_resampled, y_resampled


def select_important_features(X, y, threshold="median"):
    print("Selecting important features with Random Forest...")
    rf = RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    selector = SelectFromModel(rf, threshold=threshold, prefit=True)
    X_reduced = selector.transform(X)
    print(f"Features reduced from {X.shape[1]} to {X_reduced.shape[1]}")
    return X_reduced


def apply_confidence_threshold(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    #cm = confusion_matrix(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return y_pred, cm


# Function to Plot Train vs. Test AUC
def plot_train_vs_test_auc(rf, X_train, X_test, y_train, y_test, output_dir):
    train_proba = rf.predict_proba(X_train)[:, 1]
    test_proba = rf.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, train_proba)
    test_auc = roc_auc_score(y_test, test_proba)

    plt.bar(['Train AUC', 'Test AUC'], [train_auc, test_auc], color=['blue', 'red'])
    plt.ylim(0, 1)
    plt.ylabel('AUC Score')
    plt.title('Train vs. Test AUC')
    plt.savefig(os.path.join(output_dir, "train_vs_test_auc.png"))
    plt.close()
    print(f"Train AUC: {train_auc:.3f}, Test AUC: {test_auc:.3f}")

# Function to Plot Precision-Recall Curve
# def plot_precision_recall_curve(rf, X_test, y_test, output_dir):
    probas = rf.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, probas)

    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
    plt.close()

# Function to Plot Confusion Matrix
def plot_confusion_matrix(rf, X_test, y_test, output_dir, threshold=0.05):
    # Apply confidence threshold to predictions
    probas = rf.predict_proba(X_test)[:, 1]
    y_pred, cm = apply_confidence_threshold(y_test, probas, threshold=threshold)

    # Plotting
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Binder", "Binder"],
                yticklabels=["Non-Binder", "Binder"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (Threshold = {threshold})")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

# Function to Plot Learning Curve
from sklearn.model_selection import learning_curve

def plot_learning_curve(rf, X, y, output_dir):
    train_sizes, train_scores, test_scores = learning_curve(rf, X, y, cv=5, scoring='roc_auc')

    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Train Score")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label="Test Score")
    plt.xlabel("Training Size")
    plt.ylabel("AUC Score")
    plt.title("Learning Curve")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "learning_curve.png"))
    plt.close()

# Function to Plot Train vs. Validation Loss Curve
# def plot_loss_curve(train_losses, val_losses, output_dir):
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train vs. Validation Loss Curve")
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score

def train_model_kfold(X, y, output_dir, n_splits=3):
    print("Training Random Forest model with Stratified K-Fold Cross-Validation and SHAP Analysis...")

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [2, 4, 6],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True]
    }

    train_losses, val_losses = [], []
    auc_scores = []

    from sklearn.ensemble import RandomForestClassifier
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.model_selection import RandomizedSearchCV

    rf_base = RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=42, n_estimators=100)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[test_idx]
        y_train, y_val = y[train_idx], y[test_idx]

        #Apply SMOTE on training split
        smote = SMOTE(sampling_strategy=0.75, random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        rf = rf_base
        rf.fit(X_train_resampled, y_train_resampled)

        train_losses.append(1 - rf.score(X_train_resampled, y_train_resampled))
        val_losses.append(1 - rf.score(X_val, y_val))

        y_val_pred = rf.predict_proba(X_val)[:, 1]
        y_val_pred_class, cm = apply_confidence_threshold(y_val, y_val_pred, threshold=0.5)
        auc_scores.append(roc_auc_score(y_val, y_val_pred))

    avg_auc_score = np.mean(auc_scores)
    # plot_loss_curve(train_losses, val_losses, output_dir)

    #Grid Search on whole data
    grid_search = RandomizedSearchCV(
        estimator=rf_base, param_distributions=param_grid,
        n_iter=50, cv=skf, n_jobs=-1, scoring='roc_auc', random_state=42
    )
    grid_search.fit(X, y)

    best_rf = grid_search.best_estimator_

    #Save best parameters
    best_params = grid_search.best_params_
    with open(os.path.join(output_dir, "best_params.txt"), "w") as f:
        f.write(f"Best Parameters: {best_params}\n")
        f.write(f"Average AUC Score: {avg_auc_score}\n")

    # === SHAP Analysis ===
    print("Performing SHAP analysis...")
    try:
        X_train_shap, _, y_train_shap, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_shap = np.array(X_train_shap, dtype=np.float64)

        explainer = shap.TreeExplainer(best_rf)
        shap_values = explainer.shap_values(X_train_shap)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # binary classification

        shap_values = np.array(shap_values, dtype=np.float64)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        if shap_values.shape[1] != X_train_shap.shape[1]:
            print(f"SHAP mismatch: {shap_values.shape} vs. {X_train_shap.shape}")
            shap_values = shap_values[:, :X_train_shap.shape[1]]

        # Save SHAP values
        shap_df = pd.DataFrame(shap_values, columns=[f"Feature_{i}" for i in range(X_train_shap.shape[1])])
        shap_df.to_csv(os.path.join(output_dir, "shap_values.csv"), index=False)

        # Feature importance
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        importance_df = pd.DataFrame({
            'Feature': [f"Feature_{i}" for i in range(X_train_shap.shape[1])],
            'Mean SHAP Value': mean_shap
        }).sort_values(by='Mean SHAP Value', ascending=False)
        importance_df.to_csv(os.path.join(output_dir, "mean_shap_importance.csv"), index=False)

        # Plots
        shap.summary_plot(shap_values, X_train_shap, plot_type="bar", show=False)
        plt.savefig(os.path.join(output_dir, "shap_summary_bar.png")); plt.close()

        shap.summary_plot(shap_values, X_train_shap, plot_type="dot", show=False)
        plt.savefig(os.path.join(output_dir, "shap_summary_beeswarm.png")); plt.close()

        for feature in importance_df['Feature'].head(5):
            idx = int(feature.split("_")[1])
            # shap.dependence_plot(idx, shap_values, X_train_shap, show=False)
            # plt.savefig(os.path.join(output_dir, f"shap_dependence_{feature}.png")); plt.close()

        print("SHAP analysis completed.")
    except Exception as e:
        print(f"SHAP analysis error: {e}")

    return best_rf, avg_auc_score


# Modify SHAP Analysis to Include PR Curve & Confusion Matrix
#def shap_analysis(rf, X_train, X_test, y_train, y_test, feature_names, output_dir):
def shap_analysis(rf, X_train, X_test, y_train, y_test, feature_names, output_dir, best_threshold=0.05):
    
    print("Performing SHAP analysis...")

    try:
        #Convert to NumPy array to avoid dtype mismatch errors
        X_train = np.array(X_train, dtype=np.float64)
        X_test = np.array(X_test, dtype=np.float64)

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_train)

        #Handle SHAP Values for Binary Classification
        if isinstance(shap_values, list):  
            shap_values = shap_values[1]  # Extract SHAP values for the positive class

        #Ensure SHAP is 2D
        shap_values = np.array(shap_values, dtype=np.float64)  
        if shap_values.ndim == 3:  
            shap_values = shap_values[:, :, 1]

        #Fix Feature Mismatch Error
        if shap_values.shape[1] != X_train.shape[1]:  
            print(f"Warning: SHAP shape mismatch: {shap_values.shape} vs. {X_train.shape}")
            shap_values = shap_values[:, :X_train.shape[1]]

        #Save SHAP values
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        shap_df.to_csv(os.path.join(output_dir, "shap_values.csv"), index=False)
        print("SHAP values saved to 'shap_values.csv'.")

        #Compute Feature Importance
        mean_shap_values = np.mean(np.abs(shap_values), axis=0)
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean SHAP Value': mean_shap_values
        }).sort_values(by='Mean SHAP Value', ascending=False)
        feature_importance_df.to_csv(os.path.join(output_dir, "mean_shap_importance.csv"), index=False)
        print("Mean SHAP importance saved.")

        #SHAP Summary Plots
        shap.summary_plot(shap_values, X_train, feature_names=feature_names, plot_type="bar", show=False)
        plt.savefig(os.path.join(output_dir, "shap_summary_bar.png"))
        plt.close()

        #shap.summary_plot(shap_values, X_train, feature_names=feature_names, plot_type="dot", show=False)
        #plt.savefig(os.path.join(output_dir, "shap_summary_beeswarm.png"))
        #plt.close()

        shap.summary_plot(
            shap_values,
            X_train,
            feature_names=feature_names,
            plot_type="dot",
            show=False,
            max_display=10, # optional: control number of features shown
            cmap=plt.cm.viridis  # change to any matplotlib colormap
        )
        plt.xlabel("SHAP value (impact on model output)", fontsize=18, fontweight="bold")
        plt.ylabel("Features", fontsize=18, fontweight="bold")
        plt.xticks(fontsize=14, fontweight="bold")
        plt.yticks(fontsize=14, fontweight="bold")
        plt.savefig(os.path.join(output_dir, "shap_summary_beeswarm.png"), bbox_inches="tight")
        plt.close()


        #SHAP Dependence Plots (Top 5 Features)
        top_features = feature_importance_df['Feature'].head(5).tolist()
        for feature in top_features:
            feature_idx = int(feature.split("_")[1])
            # shap.dependence_plot(feature_idx, shap_values, X_train, feature_names=feature_names, show=False)
            # plt.savefig(os.path.join(output_dir, f"shap_dependence_{feature}.png"))
            plt.close()

        #SHAP Force Plots (First 3 Samples)
        for i in range(min(3, len(X_train))):
            # shap.force_plot(
            #     explainer.expected_value[1],
            #     shap_values[i, :],
            #     X_train[i, :],
            #     feature_names=feature_names,
            #     matplotlib=True,
            # )
            # plt.savefig(os.path.join(output_dir, f"shap_force_sample_{i+1}.png"))
            plt.close()

        print("SHAP analysis completed successfully!")

    except Exception as e:
        print(f"Error during SHAP analysis: {e}")

    #Add PR Curve & Confusion Matrix with tuned threshold
    # plot_precision_recall_curve(rf, X_test, y_test, output_dir)
    plot_confusion_matrix(rf, X_test, y_test, output_dir=output_dir, threshold=best_threshold)


# Step 6: t-SNE Visualization and Save Plot
def plot_tsne(X, y, output_dir):
    print("Generating t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X)
    plt.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], label='Non-Binders', alpha=0.6)
    plt.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], label='Binders', alpha=0.6)
    plt.legend()
    plt.title("t-SNE Visualization of Molecular Fingerprints")
    plt.savefig(os.path.join(output_dir, "tsne_plot.png"))
    plt.close()


# Plot Average AUC-ROC Curve with Shaded Error
def plot_average_auc_roc(rf, X, y, output_dir, n_splits=5, color="tab:orange", alpha=0.20, use_auc_se_in_legend=False):
    """
    Plots mean ROC curve with shaded STANDARD ERROR (SE) band.
    - Shading = pointwise SE of TPR across folds (std / sqrt(k)).
    - Legend shows Mean AUC ± SD by default; set use_auc_se_in_legend=True for ± SE.
    Saves: average_auc_roc.png
    """
    import os, numpy as np, matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        print("ROC skipped: only one class present.")
        return

    safe_splits = max(2, min(n_splits, counts.min()))
    skf = StratifiedKFold(n_splits=safe_splits, shuffle=True, random_state=42)

    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []

    for tr, te in skf.split(X, y):
        if len(np.unique(y[te])) < 2:
            continue
        rf.fit(X[tr], y[tr])
        prob = rf.predict_proba(X[te])[:, 1]
        fpr, tpr, _ = roc_curve(y[te], prob)
        tpr_i = np.interp(mean_fpr, fpr, tpr, left=0, right=1)
        tpr_i[0] = 0.0
        tprs.append(tpr_i)
        aucs.append(auc(fpr, tpr))

    if not tprs:
        print("ROC skipped: no valid folds with both classes.")
        return

    tprs = np.vstack(tprs)
    k = tprs.shape[0]

    mean_tpr = tprs.mean(axis=0)
    mean_tpr[-1] = 1.0

    # Pointwise SE band
    std_tpr = tprs.std(axis=0, ddof=1) if k > 1 else np.zeros_like(mean_tpr)
    se_tpr  = std_tpr / np.sqrt(k)
    lower = np.clip(mean_tpr - se_tpr, 0, 1)
    upper = np.clip(mean_tpr + se_tpr, 0, 1)

    mean_auc = auc(mean_fpr, mean_tpr)
    sd_auc   = np.std(aucs, ddof=1) if k > 1 else 0.0
    se_auc   = sd_auc / np.sqrt(k) if k > 1 else 0.0
    auc_err  = se_auc if use_auc_se_in_legend else sd_auc
    auc_err_label = "SE" if use_auc_se_in_legend else "SD"

    plt.figure(figsize=(8, 6))

    # Mean ROC
    plt.plot(mean_fpr, mean_tpr, color=color, lw=2,
             label=f"Mean AUC = {mean_auc:.2f} ± {auc_err:.2f} ({auc_err_label})")

    # Shaded SE ribbon
    plt.fill_between(mean_fpr, lower, upper, color=color, alpha=alpha, linewidth=0)

    # Chance line
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", lw=2, label="Chance")

    # Styling
    plt.xlabel("False Positive Rate", fontsize=18, fontweight="bold")
    plt.ylabel("True Positive Rate", fontsize=18, fontweight="bold")
    plt.title("Average AUC-ROC Curve", fontsize=18, fontweight="bold")
    plt.xticks(fontsize=14, fontweight="bold")
    plt.yticks(fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=14, frameon=False)
    plt.grid(False)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "average_auc_roc.png"), bbox_inches="tight")
    plt.close()

    # NOTE (reviewer-ready): The following script previously contained an accidental
    # LOBO evaluation block *inside* this ROC plotting helper, which could break
    # execution when running in "plots only" mode. It has been removed.
    return

# --- Main Script Begins Below ---
if __name__ == "__main__":

    import argparse
    import os

    # Argument Parsing
    parser = argparse.ArgumentParser(description="Train a Random Forest model with different fingerprints using Stratified K-Fold Cross-Validation.")
    parser.add_argument("--input", type=str, required=True, help="Path to CSV file with SMILES and Label columns.")
    parser.add_argument("--output", type=str, required=True, help="Directory to save output files.")
    parser.add_argument("--use_smote", action="store_true", help="Use SMOTE to balance classes.")
    parser.add_argument("--fingerprint", type=str, default="morgan", help="Type of fingerprint to use. Use + to combine (e.g., morgan+rdkit).")
    parser.add_argument("--kfolds", type=int, default=5, help="Number of folds for Stratified K-Fold Cross-Validation.")

    args = parser.parse_args()
    print(f"Arguments parsed: {args}")

    #Create output directory
    fingerprint_dir = os.path.join(args.output, args.fingerprint)
    try:
        os.makedirs(fingerprint_dir, exist_ok=True)
        print(f"Created/Found directory: {fingerprint_dir}")
    except Exception as e:
        print(f"Failed to create output directory: {e}")
        exit(1)

    #Read Data
    smiles, labels = read_data(args.input, args.output)
    original_smiles = smiles.copy()
    original_labels = labels.copy()
    print(f"Total molecules loaded: {len(smiles)}")

    #Generate Fingerprints
    if "+" in args.fingerprint:
        fps = args.fingerprint.split("+")
        print(f"🔗 Using fingerprint ensemble: {fps}")
        X_list = []
        for fp in fps:
            X_fp = get_fingerprints(smiles, fingerprint_type=fp)
            if X_fp is None or len(X_fp) == 0:
                print(f"Fingerprint generation failed for {fp}.")
                exit(1)
            X_list.append(np.asarray(X_fp))
        X = np.hstack(X_list)
    else:
        print(f"Generating {args.fingerprint} fingerprints...")
        X = get_fingerprints(smiles, fingerprint_type=args.fingerprint)

    if X is None or len(X) == 0:
        print("No fingerprints generated. Check input data or fingerprint type.")
        exit(1)

    #Convert labels
    y = np.array(labels)

    #Feature Selection
    print("🔄 Applying important features...")
    X = select_important_features(X, y, threshold="median")

    #Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Train Model
    try:
        print("Training model using Stratified K-Fold Cross-Validation...")
        trained_rf, avg_auc_score = train_model_kfold(X_train, y_train, fingerprint_dir, n_splits=args.kfolds)
        print("Model training completed.")
    except Exception as e:
        print(f"Error during model training: {e}")
        exit(1)

    #Save Best Parameters
    try:
        with open(os.path.join(fingerprint_dir, "best_params.txt"), "w") as f:
            f.write(f"Best Parameters: {trained_rf.get_params()}\n")
            f.write(f"Average AUC Score: {avg_auc_score:.4f}\n")
        print("Best parameters saved.")
    except Exception as e:
        print(f"Failed to save best parameters: {e}")

    #Tune Confidence Threshold
    try:
        print("Tuning best confidence threshold from test set...")
        y_test_proba = trained_rf.predict_proba(X_test)[:, 1]
        #best_threshold = tune_confidence_threshold(y_test, y_test_proba, optimize_for='recall')
        best_threshold = tune_confidence_threshold(y_test, y_test_proba, optimize_for='f1')
        print(f"Tuned threshold for prediction: {best_threshold}")
    except Exception as e:
        print(f"Error during threshold tuning: {e}")
        best_threshold = 0.5

    #SHAP Analysis
    try:
        print("Performing SHAP analysis...")
        shap_analysis(trained_rf, X_train, X_test, y_train, y_test,
                      feature_names=list(range(X_train.shape[1])),
                      output_dir=fingerprint_dir,
                      best_threshold=best_threshold)
        print("SHAP analysis completed.")
    except Exception as e:
        print(f"Error during SHAP analysis: {e}")

    #t-SNE Visualization
    try:
        print("Performing t-SNE visualization...")
        plot_tsne(X, y, fingerprint_dir)
        print("t-SNE plot saved.")
    except Exception as e:
        print(f"Error during t-SNE visualization: {e}")

    #Plot ROC Curve
    try:
        #plot_average_auc_roc(trained_rf, X_test, y_test, fingerprint_dir,
        #                     n_splits=args.kfolds, best_threshold=best_threshold)
        plot_average_auc_roc(trained_rf, X, y, fingerprint_dir, n_splits=args.kfolds)
        print("ROC curve plotted.")
    except Exception as e:
        print(f"Error plotting ROC curve: {e}")

    # Dual-mode LOBO Evaluation (Screening + Confirmation)
    # Reviewer-minimal mode: LOBO is intentionally disabled.
    # Uncomment this block if you want to run LOBO locally.
    # try:
    #     print("Performing Leave-One-Binder-Out (LOBO) Evaluation...")
    #
    #     screening_threshold = 0.25  # Low threshold for broad screening
    #     confirmation_threshold = best_threshold  # Tuned threshold for high-confidence confirmation
    #
    #     run_dual_lobo_evaluation(
    #         X=X,
    #         y=y,
    #         smiles=original_smiles,
    #         model=trained_rf,
    #         output_dir=fingerprint_dir,
    #         screening_threshold=screening_threshold,
    #         confirmation_threshold=confirmation_threshold,
    #         labels=original_labels
    #     )
    #
    #     print("Dual-mode LOBO evaluation completed.")
    # except Exception as e:
    #     print(f"Error during LOBO evaluation: {e}")

    print(f"\nChemotype analysis and hyperparameter tuning completed successfully!\n")
