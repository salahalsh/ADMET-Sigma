#!/usr/bin/env python3
"""
ADMET-Σ Model Training Script
==============================
Supplementary Script S1 for: "ADMET-Σ: A Multi-Engine ADMET Prediction Platform
with Prediction Reliability Quantification, Clinical Developability Scoring,
and Adverse Outcome Pathway Mapping"

Trains per-task Random Forest + Gradient Boosting ensemble classifiers
on MoleculeNet TOX21 (12 tasks) and ClinTox (2 tasks) benchmark datasets
using scaffold-based splitting and ECFP4 fingerprints.

Requirements:
    pip install deepchem rdkit scikit-learn numpy joblib

Usage:
    python 01_train_models.py --output_dir ./trained_models

Output:
    trained_models/
    ├── tox21/
    │   ├── checkpoints/
    │   │   └── tox21_sklearn_ensemble.joblib
    │   └── metadata.json
    └── clintox/
        ├── checkpoints/
        │   └── clintox_sklearn_ensemble.joblib
        └── metadata.json
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score,
    f1_score, matthews_corrcoef, precision_score, recall_score,
    roc_curve, precision_recall_curve, average_precision_score
)

from rdkit import Chem
from rdkit.Chem import AllChem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

TOX21_TASKS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
    'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

CLINTOX_TASKS = ['FDA_APPROVED', 'CT_TOX']

RF_PARAMS = {
    'n_estimators': 500,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'class_weight': 'balanced',
    'n_jobs': -1,
    'random_state': 42,
}

GB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'min_samples_split': 5,
    'random_state': 42,
}

FP_PARAMS = {
    'radius': 2,
    'n_bits': 2048,
}


# ============================================================================
# Featurization
# ============================================================================

def smiles_to_ecfp4(smiles_list, radius=2, n_bits=2048):
    """Convert SMILES to ECFP4 fingerprints (Morgan, radius=2, 2048 bits)."""
    fps = []
    valid_indices = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=radius, nBits=n_bits
            )
            fps.append(np.array(fp))
            valid_indices.append(i)
        else:
            logger.warning(f"Invalid SMILES at index {i}: {smi}")
    return np.array(fps), valid_indices


# ============================================================================
# Dataset Loading
# ============================================================================

def load_dataset(dataset_name):
    """Load MoleculeNet dataset using DeepChem with scaffold splitting."""
    import deepchem as dc

    logger.info(f"Loading {dataset_name} dataset with scaffold split...")

    if dataset_name == 'tox21':
        tasks, datasets, transformers = dc.molnet.load_tox21(
            featurizer='ECFP',
            splitter='scaffold',
            reload=True
        )
    elif dataset_name == 'clintox':
        tasks, datasets, transformers = dc.molnet.load_clintox(
            featurizer='ECFP',
            splitter='scaffold',
            reload=True
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_ds, valid_ds, test_ds = datasets

    logger.info(f"  Tasks: {tasks}")
    logger.info(f"  Train: {len(train_ds)} | Valid: {len(valid_ds)} | Test: {len(test_ds)}")

    return tasks, train_ds, valid_ds, test_ds


# ============================================================================
# Training
# ============================================================================

def train_ensemble(X_train, y_train, X_valid, y_valid, task_name):
    """Train RF + GB ensemble for a single task."""
    logger.info(f"  Training {task_name}: {len(y_train)} train, {len(y_valid)} valid")
    logger.info(f"    Class distribution: {np.bincount(y_train.astype(int))}")

    # Random Forest
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)

    # Gradient Boosting
    gb = GradientBoostingClassifier(**GB_PARAMS)
    gb.fit(X_train, y_train)

    # Validate
    rf_prob = rf.predict_proba(X_valid)[:, 1]
    gb_prob = gb.predict_proba(X_valid)[:, 1]
    ensemble_prob = (rf_prob + gb_prob) / 2.0
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)

    metrics = {
        'auc_roc': roc_auc_score(y_valid, ensemble_prob),
        'auc_pr': average_precision_score(y_valid, ensemble_prob),
        'accuracy': accuracy_score(y_valid, ensemble_pred),
        'balanced_accuracy': balanced_accuracy_score(y_valid, ensemble_pred),
        'f1': f1_score(y_valid, ensemble_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_valid, ensemble_pred),
        'sensitivity': recall_score(y_valid, ensemble_pred, zero_division=0),
        'specificity': recall_score(
            y_valid, ensemble_pred, pos_label=0, zero_division=0
        ),
        'rf_auc': roc_auc_score(y_valid, rf_prob),
        'gb_auc': roc_auc_score(y_valid, gb_prob),
        'n_train': len(y_train),
        'n_valid': len(y_valid),
        'positive_rate_train': float(y_train.mean()),
        'positive_rate_valid': float(y_valid.mean()),
    }

    # ROC curve points for figure generation
    fpr, tpr, _ = roc_curve(y_valid, ensemble_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_valid, ensemble_prob)

    curves = {
        'roc_fpr': fpr.tolist(),
        'roc_tpr': tpr.tolist(),
        'pr_precision': precision_curve.tolist(),
        'pr_recall': recall_curve.tolist(),
    }

    logger.info(f"    AUC-ROC: {metrics['auc_roc']:.3f} "
                f"(RF={metrics['rf_auc']:.3f}, GB={metrics['gb_auc']:.3f})")
    logger.info(f"    Balanced Acc: {metrics['balanced_accuracy']:.3f} | "
                f"MCC: {metrics['mcc']:.3f} | F1: {metrics['f1']:.3f}")

    return {'rf': rf, 'gb': gb}, metrics, curves


def train_dataset(dataset_name, output_dir):
    """Train all tasks for a dataset."""
    tasks, train_ds, valid_ds, test_ds = load_dataset(dataset_name)

    # Featurize
    logger.info("Featurizing with ECFP4 (Morgan radius=2, 2048 bits)...")
    X_train, train_valid_idx = smiles_to_ecfp4(
        train_ds.ids, **FP_PARAMS
    )
    X_valid, valid_valid_idx = smiles_to_ecfp4(
        valid_ds.ids, **FP_PARAMS
    )
    X_test, test_valid_idx = smiles_to_ecfp4(
        test_ds.ids, **FP_PARAMS
    )

    models = {}
    all_metrics = {}
    all_curves = {}
    test_metrics = {}

    for i, task in enumerate(tasks):
        logger.info(f"\n{'='*60}")
        logger.info(f"Task {i+1}/{len(tasks)}: {task}")
        logger.info(f"{'='*60}")

        # Extract labels (handle missing data with weights)
        y_train_full = train_ds.y[:, i]
        w_train_full = train_ds.w[:, i]
        y_valid_full = valid_ds.y[:, i]
        w_valid_full = valid_ds.w[:, i]
        y_test_full = test_ds.y[:, i]
        w_test_full = test_ds.w[:, i]

        # Intersect valid SMILES indices with non-missing label indices
        train_mask = np.zeros(len(y_train_full), dtype=bool)
        train_mask[train_valid_idx] = True
        train_mask &= (w_train_full > 0)

        valid_mask = np.zeros(len(y_valid_full), dtype=bool)
        valid_mask[valid_valid_idx] = True
        valid_mask &= (w_valid_full > 0)

        test_mask = np.zeros(len(y_test_full), dtype=bool)
        test_mask[test_valid_idx] = True
        test_mask &= (w_test_full > 0)

        if train_mask.sum() < 50:
            logger.warning(f"  Skipping {task}: only {train_mask.sum()} training samples")
            continue

        # Map from full index to featurized index
        train_fp_idx = np.array([
            train_valid_idx.index(j) for j in range(len(y_train_full))
            if j in train_valid_idx and train_mask[j]
        ])
        valid_fp_idx = np.array([
            valid_valid_idx.index(j) for j in range(len(y_valid_full))
            if j in valid_valid_idx and valid_mask[j]
        ])
        test_fp_idx = np.array([
            test_valid_idx.index(j) for j in range(len(y_test_full))
            if j in test_valid_idx and test_mask[j]
        ])

        X_tr = X_train[train_fp_idx]
        y_tr = y_train_full[train_mask]
        X_va = X_valid[valid_fp_idx]
        y_va = y_valid_full[valid_mask]
        X_te = X_test[test_fp_idx]
        y_te = y_test_full[test_mask]

        # Train
        model, metrics, curves = train_ensemble(X_tr, y_tr, X_va, y_va, task)
        models[task] = model
        all_metrics[task] = metrics
        all_curves[task] = curves

        # Test set evaluation
        rf_prob_test = model['rf'].predict_proba(X_te)[:, 1]
        gb_prob_test = model['gb'].predict_proba(X_te)[:, 1]
        ens_prob_test = (rf_prob_test + gb_prob_test) / 2.0
        ens_pred_test = (ens_prob_test >= 0.5).astype(int)

        test_metrics[task] = {
            'auc_roc': roc_auc_score(y_te, ens_prob_test),
            'balanced_accuracy': balanced_accuracy_score(y_te, ens_pred_test),
            'f1': f1_score(y_te, ens_pred_test, zero_division=0),
            'mcc': matthews_corrcoef(y_te, ens_pred_test),
            'sensitivity': recall_score(y_te, ens_pred_test, zero_division=0),
            'specificity': recall_score(y_te, ens_pred_test, pos_label=0, zero_division=0),
            'n_test': len(y_te),
        }
        logger.info(f"  Test AUC-ROC: {test_metrics[task]['auc_roc']:.3f}")

    # Save
    save_dir = Path(output_dir) / dataset_name
    checkpoint_dir = save_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_path = checkpoint_dir / f'{dataset_name}_sklearn_ensemble.joblib'
    joblib.dump(models, model_path)
    logger.info(f"\nModels saved to: {model_path}")

    # Compute overall metrics
    val_aucs = [m['auc_roc'] for m in all_metrics.values()]
    test_aucs = [m['auc_roc'] for m in test_metrics.values()]

    metadata = {
        'model_type': 'sklearn_ensemble',
        'version': '1.0.0',
        'training_date': datetime.now().isoformat(),
        'dataset': dataset_name,
        'tasks': list(models.keys()),
        'n_tasks': len(models),
        'fingerprint': {
            'type': 'ECFP4 (Morgan)',
            'radius': FP_PARAMS['radius'],
            'n_bits': FP_PARAMS['n_bits'],
        },
        'split_method': 'scaffold (Bemis-Murcko)',
        'split_ratios': '80/10/10',
        'rf_params': RF_PARAMS,
        'gb_params': GB_PARAMS,
        'validation_metrics': all_metrics,
        'test_metrics': test_metrics,
        'auc_roc': float(np.mean(val_aucs)),
        'auc_roc_std': float(np.std(val_aucs)),
        'test_auc_roc': float(np.mean(test_aucs)),
        'test_auc_roc_std': float(np.std(test_aucs)),
        'per_task_auc': {k: v['auc_roc'] for k, v in all_metrics.items()},
        'per_task_test_auc': {k: v['auc_roc'] for k, v in test_metrics.items()},
    }

    meta_path = save_dir / 'metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Metadata saved to: {meta_path}")

    # Save ROC/PR curves for figure generation
    curves_path = save_dir / 'curves.json'
    with open(curves_path, 'w') as f:
        json.dump(all_curves, f)
    logger.info(f"Curves saved to: {curves_path}")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY: {dataset_name.upper()}")
    logger.info(f"{'='*60}")
    logger.info(f"Tasks trained: {len(models)}")
    logger.info(f"Mean Validation AUC-ROC: {np.mean(val_aucs):.3f} +/- {np.std(val_aucs):.3f}")
    logger.info(f"Mean Test AUC-ROC:       {np.mean(test_aucs):.3f} +/- {np.std(test_aucs):.3f}")
    for task in models:
        logger.info(f"  {task:20s}  Val={all_metrics[task]['auc_roc']:.3f}  "
                     f"Test={test_metrics[task]['auc_roc']:.3f}")

    return models, metadata


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ADMET-Σ Model Training Script'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./trained_models',
        help='Output directory for trained models'
    )
    parser.add_argument(
        '--dataset', type=str, default='both',
        choices=['tox21', 'clintox', 'both'],
        help='Dataset to train on'
    )
    args = parser.parse_args()

    start_time = time.time()

    if args.dataset in ('tox21', 'both'):
        logger.info("\n" + "="*60)
        logger.info("TRAINING TOX21 MODELS")
        logger.info("="*60)
        train_dataset('tox21', args.output_dir)

    if args.dataset in ('clintox', 'both'):
        logger.info("\n" + "="*60)
        logger.info("TRAINING CLINTOX MODELS")
        logger.info("="*60)
        train_dataset('clintox', args.output_dir)

    elapsed = time.time() - start_time
    logger.info(f"\nTotal training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
