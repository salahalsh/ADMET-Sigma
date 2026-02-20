# ADMET-X: Supplementary Scripts and Data

Reproducibility scripts for:

> **ADMET-X: A Multi-Engine ADMET Prediction Platform with Prediction Reliability Quantification, Clinical Developability Scoring, and Adverse Outcome Pathway Mapping**
>
> Salah A. Alsharif
>
> *Journal of Chemical Information and Modeling* (2026)

## Overview

ADMET-X integrates three prediction tiers (RDKit rule-based, Deep-PK deep learning, and locally trained ML ensembles) with ten translational feature modules into a unified ADMET prediction platform that generates ~250 properties per compound.

**Web application:** [https://insilicox-lab.com/](https://insilicox-lab.com/)

## Repository Structure

```
admet-x/
├── 01_train_models.py              # Script S1: Train RF+GB ensembles on TOX21/ClinTox
├── 02_validate_pri_cds.py          # Script S2: Validate PRI and CDS metrics
├── 03_case_studies.py              # Script S3: Run case studies (38 reference drugs)
├── 04_generate_figures.py          # Script S4: Generate manuscript figures 2-4, 6-8
├── 05_production_validation.py     # Script S5: Production pipeline validation
├── generate_fig1_fig5.py           # Generate architecture (Fig 1) and AOP (Fig 5) diagrams
├── trained_models/
│   ├── tox21/
│   │   ├── checkpoints/            # Trained ensemble model (joblib)
│   │   ├── metadata.json           # Training metadata and metrics
│   │   └── curves.json             # ROC curve data
│   └── clintox/
│       ├── checkpoints/            # Trained ensemble model (joblib)
│       ├── metadata.json           # Training metadata and metrics
│       └── curves.json             # ROC curve data
├── validation_results/
│   ├── pri_calibration_results.csv # PRI validation on 38 reference drugs
│   ├── cds_validation_results.csv  # CDS approved vs withdrawn comparison
│   └── cds_validation_summary.json # Statistical summary
├── production_results/
│   ├── validation_production.csv   # Production pipeline validation data
│   ├── production_summary.json     # Summary statistics
│   ├── case_study_production.json  # Case study production outputs
│   └── case_study_raw.json         # Raw pipeline outputs
├── case_study_results/
│   ├── case_study_profiles.json    # Complete ADMET profiles
│   ├── case_study_summary.csv      # Summary table
│   └── case_study_report.txt       # Human-readable report
├── figures/                        # Generated manuscript figures (PDF + PNG)
├── LICENSE                         # CC BY-NC 4.0
└── README.md
```

## Scripts

### Script S1: Model Training (`01_train_models.py`)

Trains per-task Random Forest (500 estimators) and Gradient Boosting (200 estimators) ensemble classifiers on MoleculeNet TOX21 (12 tasks, 7,831 compounds) and ClinTox (2 tasks, 1,491 compounds) using scaffold-based splitting and ECFP4 2048-bit fingerprints.

```bash
python 01_train_models.py --output_dir ./trained_models
```

### Script S2: PRI and CDS Validation (`02_validate_pri_cds.py`)

Validates the Prediction Reliability Index (PRI) and Clinical Developability Score (CDS) using a panel of 38 well-characterized drugs (25 FDA-approved, 13 withdrawn).

```bash
python 02_validate_pri_cds.py --models_dir ./trained_models --output_dir ./validation_results
```

### Script S3: Case Studies (`03_case_studies.py`)

Runs comprehensive ADMET-X profiling on reference compounds including aspirin, simvastatin, and troglitazone. Generates all case study data reported in the manuscript.

```bash
python 03_case_studies.py --models_dir ./trained_models --output_dir ./case_study_results
```

### Script S4: Figure Generation (`04_generate_figures.py`)

Generates publication-quality figures 2-4 and 6-8 using matplotlib and seaborn.

```bash
python 04_generate_figures.py --models_dir ./trained_models --validation_dir ./validation_results --output_dir ./figures
```

### Script S5: Production Validation (`05_production_validation.py`)

Validates the production ADMET-X pipeline by running all 38 reference drugs through the live system and comparing outputs against expected values.

```bash
python 05_production_validation.py --output_dir ./production_results
```

## Requirements

```
python >= 3.10
numpy >= 1.24
scikit-learn >= 1.2
rdkit >= 2023.03
deepchem >= 2.5.0
matplotlib >= 3.7
seaborn >= 0.12
pandas >= 2.0
scipy >= 1.10
joblib >= 1.2
```

Install all dependencies:

```bash
pip install numpy scikit-learn rdkit deepchem matplotlib seaborn pandas scipy joblib
```

## Trained Models

> **Note:** Model checkpoint files (`.joblib`) are not included in this repository due to size constraints (~600 MB). To generate them, run `python 01_train_models.py`. Training metadata and ROC curve data are included.

Pre-trained model performance on `trained_models/`:

| Dataset | Tasks | Architecture | Split | Mean AUC-ROC |
|---------|-------|-------------|-------|-------------|
| TOX21 | 12 | RF(500) + GB(200) | Scaffold | 0.710 |
| ClinTox | 2 | RF(500) + GB(200) | Scaffold | 0.777 |

Models use ECFP4 Morgan fingerprints (radius=2, 2048 bits) as input features.

## Reproducing Manuscript Results

To reproduce all results from the manuscript, run the scripts in order:

```bash
# 1. Train models (or use provided pre-trained models)
python 01_train_models.py --output_dir ./trained_models

# 2. Validate PRI and CDS
python 02_validate_pri_cds.py --models_dir ./trained_models --output_dir ./validation_results

# 3. Run case studies
python 03_case_studies.py --models_dir ./trained_models --output_dir ./case_study_results

# 4. Generate figures
python 04_generate_figures.py --models_dir ./trained_models --validation_dir ./validation_results --output_dir ./figures

# 5. Validate production pipeline (requires running ADMET-X web application)
python 05_production_validation.py --output_dir ./production_results
```

## Citation

If you use ADMET-X or any part of this repository in your research, please cite:

```bibtex
@article{alsharif2026admetx,
  title={ADMET-X: A Multi-Engine ADMET Prediction Platform with Prediction Reliability Quantification, Clinical Developability Scoring, and Adverse Outcome Pathway Mapping},
  author={Alsharif, Salah A.},
  journal={Journal of Chemical Information and Modeling},
  year={2026},
  publisher={American Chemical Society}
}
```

## License

This project is licensed under the [CC BY-NC 4.0 License](LICENSE) - non-commercial use with attribution required.
