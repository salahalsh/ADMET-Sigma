#!/usr/bin/env python3
"""
ADMET-Σ Production Validation Script
======================================
Runs all case study and validation compounds through the actual production
ADMET-Σ aggregator to ensure paper numbers match tool output.

This script imports the real ADMETAggregator from the Django project and
runs compounds with engine='RDKIT' (which adds Trained ML, enhanced features).

Usage:
    cd "D:/myTools/Tool - InsilicoX Project/insilicox_web_app"
    python "C:/Users/salah/Downloads/ADMET-Sigma paper/scripts/05_production_validation.py"
"""

import os
import sys
import json
import csv
import logging
from pathlib import Path
from datetime import datetime

# ============================================================================
# Django Setup (minimal — needed for trained model loading)
# ============================================================================
WEBAPP_DIR = r"D:\myTools\Tool - InsilicoX Project\insilicox_web_app"
sys.path.insert(0, WEBAPP_DIR)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'insilicox.settings')

import django
django.setup()

# Now safe to import ADMET-Σ services
from admet_x.services import ADMETAggregator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Compound Libraries
# ============================================================================

CASE_STUDIES = {
    "Aspirin": {
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "category": "classical_oral_drug",
        "known_bcs": "I",
    },
    "Simvastatin": {
        "smiles": "CCC(C)(C)C(=O)OC1CC(O)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C21",
        "category": "cyp3a4_victim",
        "known_bcs": "II",
    },
    "Cyclosporine A": {
        "smiles": "CCC1NC(=O)C(CC(C)C)N(C)C(=O)C(CC(C)C)N(C)C(=O)C(C(O)C(C)CC=CC)N(C)C(=O)C(C(C)C)NC(=O)C(C(C)C)NC(=O)C(CC(C)C)N(C)C(=O)C(C(C)CC)N(C)C(=O)CN(C)C(=O)C(C(C)C)N(C)C1=O",
        "category": "bro5_macrocycle",
        "known_bcs": "IV",
    },
    "ARV-110": {
        "smiles": "CC(C)c1cc(NC(=O)COc2ccc(-c3cnc4cc(F)c(OCC5CCN(CC6CC(=O)NC(=O)C6)C5)cc4n3)cc2)ccc1F",
        "category": "protac",
        "known_bcs": "IV",
    },
}

FDA_APPROVED_DRUGS = [
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("Metformin", "CN(C)C(=N)NC(=N)N"),
    ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("Amlodipine", "CCOC(=O)C1=C(COCCN)NC(C)=C(C1c1ccccc1Cl)C(=O)OC"),
    ("Metoprolol", "COCCc1ccc(OCC(O)CNC(C)C)cc1"),
    ("Simvastatin", "CCC(C)(C)C(=O)OC1CC(O)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C21"),
    ("Losartan", "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1"),
    ("Omeprazole", "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1"),
    ("Fluoxetine", "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1"),
    ("Warfarin", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"),
    ("Diazepam", "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21"),
    ("Carbamazepine", "NC(=O)N1c2ccccc2C=Cc2ccccc21"),
    ("Acetaminophen", "CC(=O)Nc1ccc(O)cc1"),
    ("Caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
    ("Propranolol", "CC(C)NCC(O)COc1cccc2ccccc12"),
    ("Amoxicillin", "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O"),
    ("Naproxen", "COc1ccc2cc(C(C)C(=O)O)ccc2c1"),
    ("Fluconazole", "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F"),
    ("Gabapentin", "NCC1(CC(=O)O)CCCCC1"),
    ("Atenolol", "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"),
    ("Pioglitazone", "CCc1ccc(CCOc2ccc(CC3SC(=O)NC3=O)cc2)nc1"),
    ("Tamoxifen", "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1"),
    ("Verapamil", "COc1ccc(CCN(C)CCCC(C#N)(C(C)C)c2ccc(OC)c(OC)c2)cc1OC"),
    ("Imatinib", "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"),
    ("Phenytoin", "O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1"),
]

WITHDRAWN_DRUGS = [
    ("Troglitazone", "Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O", "Hepatotoxicity"),
    ("Rofecoxib", "CS(=O)(=O)c1ccc(C2=C(c3ccccc3)C(=O)OC2)cc1", "Cardiovascular"),
    ("Valdecoxib", "CS(=O)(=O)c1ccc(-c2c(-c3ccccc3)noc2N)cc1", "Cardiovascular/Skin"),
    ("Tegaserod", "CCCCC(=N/NC1=NC(=O)NC2=CC=CC=C21)c1ccncc1", "Cardiovascular"),
    ("Sibutramine", "CC(C1(c2ccc(Cl)cc2)CCC1)N(C)C", "Cardiovascular"),
    ("Rosiglitazone", "CN(CCOc1ccc(CC2SC(=O)NC2=O)cc1)c1ccccn1", "Cardiovascular"),
    ("Cisapride", "COc1cc(N)c(Cl)cc1C(=O)NC1CC(OC)CCN1CCCC(=O)OC", "QT prolongation"),
    ("Terfenadine", "OC(CCN1CCC(C(O)(c2ccccc2)c2ccccc2)CC1)(C(C)(C)C)c1ccc(cc1)C(C)(C)C", "QT prolongation"),
    ("Nefazodone", "CCc1nn(CCCN2CCN(c3cccc(Cl)c3)CC2)c(=O)n1CCOc1ccccc1", "Hepatotoxicity"),
    ("Bromfenac", "OC(=O)Cc1cc(Br)ccc1Nc1ccccc1C=O", "Hepatotoxicity"),
    ("Benoxaprofen", "OC(=O)C(C)c1ccc2oc(-c3ccc(Cl)cc3)nc2c1", "Hepatotoxicity/Phototox"),
    ("Pemoline", "NC1(c2ccccc2)OC(=O)N=C1O", "Hepatotoxicity"),
    ("Dexfenfluramine", "CCN(C)C(C)c1ccc(C(F)(F)F)cc1", "Cardiac valvulopathy"),
]


def extract_key(result, keys, default=None):
    """Try multiple possible keys, return first found."""
    if isinstance(keys, str):
        keys = [keys]
    for k in keys:
        val = result.get(k)
        if val is not None:
            return val
    return default


def extract_cds(result):
    """Extract CDS data from production result."""
    return {
        'cds_overall': extract_key(result, ['cds_score', 'cds_overall_score']),
        'cds_category': extract_key(result, ['cds_category']),
        'cds_drug_likeness': extract_key(result, ['cds_drug_likeness']),
        'cds_metabolic_stability': extract_key(result, ['cds_metabolic_stability']),
        'cds_safety_profile': extract_key(result, ['cds_safety_profile']),
        'cds_permeability': extract_key(result, ['cds_permeability']),
        'cds_physicochemical': extract_key(result, ['cds_physicochemical']),
    }


def extract_pri(result):
    """Extract PRI data from production result."""
    return {
        'pri_overall': extract_key(result, ['pri_score']),
        'pri_category': extract_key(result, ['pri_category']),
        'pri_ad': extract_key(result, ['pri_ad_component']),
        'pri_ensemble': extract_key(result, ['pri_ensemble_component']),
        'pri_conformal': extract_key(result, ['pri_conformal_component']),
        'pri_model_perf': extract_key(result, ['pri_performance_component']),
    }


def extract_ddi(result):
    """Extract DDI data from production result."""
    return {
        'ddi_risk_score': extract_key(result, ['ddi_risk_score']),
        'ddi_risk_category': extract_key(result, ['ddi_risk_category']),
        'ddi_n_cyp_interactions': extract_key(result, ['ddi_n_cyp_interactions']),
    }


def extract_aop(result):
    """Extract AOP data from production result."""
    return {
        'aop_n_triggered': extract_key(result, ['aop_n_triggered']),
        'aop_risk_score': extract_key(result, ['aop_risk_score']),
        'aop_risk_category': extract_key(result, ['aop_risk_category']),
    }


def extract_bcs(result):
    """Extract BCS data from production result."""
    return {
        'bcs_class': extract_key(result, ['bcs_class']),
        'bcs_solubility': extract_key(result, ['bcs_solubility_category']),
        'bcs_permeability': extract_key(result, ['bcs_permeability_category']),
    }


def extract_bro5(result):
    """Extract bRo5/PROTAC data from production result."""
    return {
        'compound_class': extract_key(result, ['bro5_compound_class']),
        'chameleonicity': extract_key(result, ['bro5_chameleonicity']),
        'ro5_violations': extract_key(result, ['bro5_ro5_violations', 'lipinski_violations']),
        'is_macrocycle': extract_key(result, ['bro5_is_macrocycle'], False),
    }


def extract_comparative(result):
    """Extract comparative profiling data."""
    return {
        'most_similar_drug': extract_key(result, ['comparative_most_similar']),
        'most_similar_score': extract_key(result, ['comparative_similarity']),
    }


def run_compound(agg, name, smiles):
    """Run a single compound through the production aggregator."""
    logger.info(f"Processing: {name}")
    try:
        result = agg.calculate(
            smiles=smiles,
            compound_id=name,
            engine='RDKIT',  # Tier 1 + Trained ML + all enhanced features
            include_ad=True,
            include_uncertainty=True,
            include_shap=False,
        )
        if not result.get('success', True):
            logger.warning(f"  Failed: {result.get('error', 'Unknown')}")
        return result
    except Exception as e:
        logger.error(f"  Exception: {e}")
        return {'success': False, 'error': str(e)}


def main():
    output_dir = Path(r"C:\Users\salah\Downloads\ADMET-Sigma paper\scripts\production_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("ADMET-Σ Production Validation")
    logger.info("=" * 70)

    # Initialize aggregator
    logger.info("Initializing ADMETAggregator...")
    agg = ADMETAggregator(pkcm_api_key=None)
    engines = agg.get_available_engines()
    logger.info(f"Available engines: {json.dumps(engines, indent=2)}")

    # ============================================================
    # Part 1: Case Study Compounds
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("PART 1: Case Study Compounds (n=4)")
    logger.info("=" * 70)

    case_results = {}
    for name, info in CASE_STUDIES.items():
        result = run_compound(agg, name, info['smiles'])
        case_results[name] = {
            'raw': result,
            'cds': extract_cds(result),
            'pri': extract_pri(result),
            'ddi': extract_ddi(result),
            'aop': extract_aop(result),
            'bcs': extract_bcs(result),
            'bro5': extract_bro5(result),
            'comparative': extract_comparative(result),
            'mw': result.get('molecular_weight'),
            'logp': result.get('logp'),
            'tpsa': result.get('tpsa'),
            'prediction_engine': result.get('prediction_engine'),
        }

    # Print case study summary
    logger.info("\n--- Case Study Summary ---")
    for name, data in case_results.items():
        cds = data['cds']
        pri = data['pri']
        ddi = data['ddi']
        aop = data['aop']
        bcs = data['bcs']
        bro5 = data['bro5']
        comp = data['comparative']
        logger.info(f"\n{name}:")
        logger.info(f"  Engine: {data['prediction_engine']}")
        logger.info(f"  MW={data['mw']}, LogP={data['logp']}, TPSA={data['tpsa']}")
        logger.info(f"  CDS={cds['cds_overall']} ({cds['cds_category']})")
        logger.info(f"    DL={cds['cds_drug_likeness']}, MS={cds['cds_metabolic_stability']}, "
                     f"SP={cds['cds_safety_profile']}, PD={cds['cds_permeability']}, "
                     f"PC={cds['cds_physicochemical']}")
        logger.info(f"  PRI={pri['pri_overall']} ({pri['pri_category']})")
        logger.info(f"    AD={pri['pri_ad']}, Ens={pri['pri_ensemble']}, "
                     f"Conf={pri['pri_conformal']}, Model={pri['pri_model_perf']}")
        logger.info(f"  DDI: risk_score={ddi['ddi_risk_score']} ({ddi['ddi_risk_category']})")
        logger.info(f"  AOP: n_triggered={aop['aop_n_triggered']}, risk={aop['aop_risk_score']} ({aop['aop_risk_category']})")
        logger.info(f"  BCS: Class {bcs['bcs_class']}")
        logger.info(f"  bRo5: {bro5['compound_class']}, violations={bro5['ro5_violations']}, chameleonicity={bro5['chameleonicity']}")
        logger.info(f"  Most similar: {comp['most_similar_drug']} (Tc={comp['most_similar_score']})")

    # Save case study results
    case_summary_path = output_dir / "case_study_production.json"
    # Clean raw results for JSON serialization
    case_export = {}
    for name, data in case_results.items():
        case_export[name] = {k: v for k, v in data.items() if k != 'raw'}
    with open(case_summary_path, 'w') as f:
        json.dump(case_export, f, indent=2, default=str)
    logger.info(f"\nCase study results saved to {case_summary_path}")

    # Save full raw results for inspection
    raw_path = output_dir / "case_study_raw.json"
    raw_export = {}
    for name, data in case_results.items():
        raw_export[name] = {k: str(v) if not isinstance(v, (str, int, float, bool, type(None), list, dict)) else v
                           for k, v in data['raw'].items()}
    with open(raw_path, 'w') as f:
        json.dump(raw_export, f, indent=2, default=str)
    logger.info(f"Raw results saved to {raw_path}")

    # ============================================================
    # Part 2: Validation Compounds (25 approved + 13 withdrawn)
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("PART 2: Validation Compounds (n=38)")
    logger.info("=" * 70)

    validation_results = []

    # Process approved drugs
    for name, smiles in FDA_APPROVED_DRUGS:
        result = run_compound(agg, name, smiles)
        cds = extract_cds(result)
        pri = extract_pri(result)
        ddi = extract_ddi(result)
        aop = extract_aop(result)
        bcs = extract_bcs(result)
        validation_results.append({
            'name': name,
            'status': 'approved',
            'cds_overall': cds['cds_overall'],
            'cds_category': cds['cds_category'],
            'cds_drug_likeness': cds['cds_drug_likeness'],
            'cds_metabolic_stability': cds['cds_metabolic_stability'],
            'cds_safety_profile': cds['cds_safety_profile'],
            'cds_permeability': cds['cds_permeability'],
            'cds_physicochemical': cds['cds_physicochemical'],
            'pri_overall': pri['pri_overall'],
            'pri_category': pri['pri_category'],
            'pri_ad': pri['pri_ad'],
            'pri_ensemble': pri['pri_ensemble'],
            'pri_conformal': pri['pri_conformal'],
            'pri_model_perf': pri['pri_model_perf'],
            'ddi_risk_score': ddi['ddi_risk_score'],
            'ddi_risk_category': ddi['ddi_risk_category'],
            'aop_n_triggered': aop['aop_n_triggered'],
            'aop_risk_score': aop['aop_risk_score'],
            'aop_risk_category': aop['aop_risk_category'],
            'bcs_class': bcs['bcs_class'],
        })

    # Process withdrawn drugs
    for name, smiles, reason in WITHDRAWN_DRUGS:
        result = run_compound(agg, name, smiles)
        cds = extract_cds(result)
        pri = extract_pri(result)
        ddi = extract_ddi(result)
        aop = extract_aop(result)
        bcs = extract_bcs(result)
        validation_results.append({
            'name': name,
            'status': 'withdrawn',
            'withdrawal_reason': reason,
            'cds_overall': cds['cds_overall'],
            'cds_category': cds['cds_category'],
            'cds_drug_likeness': cds['cds_drug_likeness'],
            'cds_metabolic_stability': cds['cds_metabolic_stability'],
            'cds_safety_profile': cds['cds_safety_profile'],
            'cds_permeability': cds['cds_permeability'],
            'cds_physicochemical': cds['cds_physicochemical'],
            'pri_overall': pri['pri_overall'],
            'pri_category': pri['pri_category'],
            'pri_ad': pri['pri_ad'],
            'pri_ensemble': pri['pri_ensemble'],
            'pri_conformal': pri['pri_conformal'],
            'pri_model_perf': pri['pri_model_perf'],
            'ddi_risk_score': ddi['ddi_risk_score'],
            'ddi_risk_category': ddi['ddi_risk_category'],
            'aop_n_triggered': aop['aop_n_triggered'],
            'aop_risk_score': aop['aop_risk_score'],
            'aop_risk_category': aop['aop_risk_category'],
            'bcs_class': bcs['bcs_class'],
        })

    # Save validation CSV
    val_csv_path = output_dir / "validation_production.csv"
    if validation_results:
        # Collect all unique keys across all records
        all_keys = []
        seen = set()
        for r in validation_results:
            for k in r.keys():
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)
        with open(val_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(validation_results)
        logger.info(f"\nValidation results saved to {val_csv_path}")

    # ============================================================
    # Part 3: Statistical Analysis
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("PART 3: Statistical Analysis")
    logger.info("=" * 70)

    import numpy as np
    from scipy import stats

    approved = [r for r in validation_results if r['status'] == 'approved']
    withdrawn = [r for r in validation_results if r['status'] == 'withdrawn']

    def safe_values(records, key):
        return [float(r[key]) for r in records if r[key] is not None]

    # CDS Analysis
    app_cds = safe_values(approved, 'cds_overall')
    wit_cds = safe_values(withdrawn, 'cds_overall')

    if app_cds and wit_cds:
        app_median = np.median(app_cds)
        wit_median = np.median(wit_cds)
        app_mean = np.mean(app_cds)
        wit_mean = np.mean(wit_cds)
        app_std = np.std(app_cds, ddof=1)
        wit_std = np.std(wit_cds, ddof=1)

        u_stat, p_val = stats.mannwhitneyu(app_cds, wit_cds, alternative='greater')

        # Cohen's d
        pooled_std = np.sqrt(((len(app_cds)-1)*app_std**2 + (len(wit_cds)-1)*wit_std**2) /
                             (len(app_cds) + len(wit_cds) - 2))
        cohens_d = (app_mean - wit_mean) / pooled_std if pooled_std > 0 else 0

        logger.info(f"\nCDS Overall:")
        logger.info(f"  Approved (n={len(app_cds)}): median={app_median:.1f}, mean={app_mean:.1f} ± {app_std:.1f}")
        logger.info(f"  Withdrawn (n={len(wit_cds)}): median={wit_median:.1f}, mean={wit_mean:.1f} ± {wit_std:.1f}")
        logger.info(f"  Mann-Whitney U={u_stat:.1f}, p={p_val:.6f}")
        logger.info(f"  Cohen's d={cohens_d:.2f}")

        # Per-dimension analysis
        for dim in ['cds_drug_likeness', 'cds_metabolic_stability', 'cds_safety_profile',
                     'cds_permeability', 'cds_physicochemical']:
            app_dim = safe_values(approved, dim)
            wit_dim = safe_values(withdrawn, dim)
            if app_dim and wit_dim:
                logger.info(f"\n  {dim}:")
                logger.info(f"    Approved: mean={np.mean(app_dim):.1f} ± {np.std(app_dim, ddof=1):.1f}")
                logger.info(f"    Withdrawn: mean={np.mean(wit_dim):.1f} ± {np.std(wit_dim, ddof=1):.1f}")

    # PRI Analysis
    all_pri = safe_values(validation_results, 'pri_overall')
    if all_pri:
        high = sum(1 for p in all_pri if p >= 0.70)
        moderate = sum(1 for p in all_pri if 0.45 <= p < 0.70)
        low = sum(1 for p in all_pri if 0.25 <= p < 0.45)
        unreliable = sum(1 for p in all_pri if p < 0.25)
        n = len(all_pri)

        logger.info(f"\nPRI Distribution (n={n}):")
        logger.info(f"  High (>=0.70): {high} ({100*high/n:.0f}%)")
        logger.info(f"  Moderate (0.45-0.70): {moderate} ({100*moderate/n:.0f}%)")
        logger.info(f"  Low (0.25-0.45): {low} ({100*low/n:.0f}%)")
        logger.info(f"  Unreliable (<0.25): {unreliable} ({100*unreliable/n:.0f}%)")
        logger.info(f"  Mean PRI: {np.mean(all_pri):.3f} ± {np.std(all_pri, ddof=1):.3f}")

        # PRI component stats
        for comp in ['pri_ad', 'pri_ensemble', 'pri_conformal', 'pri_model_perf']:
            vals = safe_values(validation_results, comp)
            if vals:
                logger.info(f"  {comp}: mean={np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}, range={min(vals):.2f}-{max(vals):.2f}")

    # Save statistical summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'engine_availability': engines,
        'n_approved': len(approved),
        'n_withdrawn': len(withdrawn),
    }
    if app_cds and wit_cds:
        summary.update({
            'cds_approved_median': float(app_median),
            'cds_approved_mean': float(app_mean),
            'cds_approved_std': float(app_std),
            'cds_withdrawn_median': float(wit_median),
            'cds_withdrawn_mean': float(wit_mean),
            'cds_withdrawn_std': float(wit_std),
            'mann_whitney_U': float(u_stat),
            'p_value': float(p_val),
            'cohens_d': float(cohens_d),
        })
    if all_pri:
        summary.update({
            'pri_mean': float(np.mean(all_pri)),
            'pri_std': float(np.std(all_pri, ddof=1)),
            'pri_high_pct': 100 * high / n,
            'pri_moderate_pct': 100 * moderate / n,
            'pri_low_pct': 100 * low / n,
            'pri_unreliable_pct': 100 * unreliable / n,
        })

    summary_path = output_dir / "production_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nStatistical summary saved to {summary_path}")

    logger.info("\n" + "=" * 70)
    logger.info("DONE. All results in: " + str(output_dir))
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
