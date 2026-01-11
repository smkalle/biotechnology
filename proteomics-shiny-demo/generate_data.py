#!/usr/bin/env python3
"""
Synthetic Proteomics Data Generator
Generates realistic cross-study proteomics data for demo purposes
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# --- Configuration ---
n_proteins = 500
n_samples_per_study = {'Trial_A': 60, 'Trial_B': 45, 'Trial_C': 35}
n_total_samples = sum(n_samples_per_study.values())

# --- Generate Protein IDs ---
uniprot_ids = [f"P{np.random.randint(10000, 99999):05d}" for _ in range(int(n_proteins * 0.6))]

gene_symbols = [
    "ALB", "TNF", "IL6", "IL10", "VEGFA", "EGFR", "HER2", "TP53", "BRCA1", "MYC",
    "KRAS", "BRAF", "AKT1", "PTEN", "CDK4", "MTOR", "PIK3CA", "MAPK1", "JAK2", "STAT3",
    "BCL2", "BAX", "CASP3", "FAS", "APAF1", "CYCS", "BID", "MCL1", "XIAP", "SURVIVIN",
    "CD4", "CD8A", "FOXP3", "CD25", "PD1", "PDL1", "CTLA4", "LAG3", "TIM3", "TIGIT",
    "GAPDH", "ACTB", "TUBB", "HSP90", "HSP70", "GRP78", "CALR", "CANX", "PDI", "ERO1",
    "COX1", "COX2", "LOX", "NOS2", "HIF1A", "VEGFR2", "ANGPT1", "FGF2", "PDGFB", "TGFB1"
]

additional_genes = [f"GENE{i:03d}" for i in range(1, int(n_proteins * 0.4 - len(gene_symbols)) + 1)]

protein_ids = (uniprot_ids + gene_symbols + additional_genes)[:n_proteins]

# --- Generate Sample Metadata ---
metadata_rows = []
for trial, n in n_samples_per_study.items():
    for i in range(1, n + 1):
        row = {
            'sample_id': f"{trial}_S{i:03d}",
            'trial': trial,
            'treatment': np.random.choice(['Drug_A', 'Drug_B', 'Placebo'], p=[0.4, 0.4, 0.2]),
            'timepoint': np.random.choice(['Baseline', 'Week_4', 'Week_12', 'Week_24'], p=[0.3, 0.3, 0.25, 0.15]),
            'response': np.random.choice(['Responder', 'Non-Responder', 'Partial'], p=[0.35, 0.40, 0.25]),
            'age': int(np.random.normal(55, 12)),
            'sex': np.random.choice(['M', 'F'])
        }
        metadata_rows.append(row)

metadata = pd.DataFrame(metadata_rows)
sample_ids = metadata['sample_id'].tolist()

# --- Generate Expression Matrix ---
base_expression = np.random.normal(10, 2, n_proteins)
expression_matrix = np.zeros((n_proteins, n_total_samples))

for i in range(n_proteins):
    expression_matrix[i, :] = base_expression[i] + np.random.normal(0, 1.5, n_total_samples)

# --- Add Biological Effects ---
# Treatment effects
treatment_responsive = np.random.choice(n_proteins, 50, replace=False)
for p in treatment_responsive:
    drug_a_idx = metadata[metadata['treatment'] == 'Drug_A'].index.tolist()
    drug_b_idx = metadata[metadata['treatment'] == 'Drug_B'].index.tolist()
    
    effect_a = np.random.uniform(0.5, 2.0) * np.random.choice([-1, 1])
    effect_b = np.random.uniform(0.3, 1.5) * np.random.choice([-1, 1])
    
    expression_matrix[p, drug_a_idx] += effect_a
    expression_matrix[p, drug_b_idx] += effect_b

# Response-associated proteins
remaining = list(set(range(n_proteins)) - set(treatment_responsive))
response_proteins = np.random.choice(remaining, 40, replace=False)
for p in response_proteins:
    resp_idx = metadata[metadata['response'] == 'Responder'].index.tolist()
    non_resp_idx = metadata[metadata['response'] == 'Non-Responder'].index.tolist()
    
    effect = np.random.uniform(0.8, 2.5) * np.random.choice([-1, 1])
    
    expression_matrix[p, resp_idx] += effect
    expression_matrix[p, non_resp_idx] -= effect * 0.5

# Trial batch effects
trial_effects = {'Trial_A': 0, 'Trial_B': 0.3, 'Trial_C': -0.2}
for trial, effect in trial_effects.items():
    trial_idx = metadata[metadata['trial'] == trial].index.tolist()
    expression_matrix[:, trial_idx] += effect

# --- Introduce Missing Values ---
missingness_rate = 0.08
low_abundance = base_expression < np.percentile(base_expression, 30)

for i in range(n_proteins):
    rate = missingness_rate * 2 if low_abundance[i] else missingness_rate * 0.5
    n_missing = int(n_total_samples * rate)
    missing_idx = np.random.choice(n_total_samples, n_missing, replace=False)
    expression_matrix[i, missing_idx] = np.nan

# --- Create DataFrames ---
expr_df = pd.DataFrame(expression_matrix, index=protein_ids, columns=sample_ids)
expr_df.index.name = 'protein_id'

# --- Save Data ---
expr_df.to_csv('data/expression_matrix.csv')
metadata.to_csv('data/metadata.csv', index=False)

print(f"✓ Generated {n_proteins} proteins across {n_total_samples} samples")
print(f"✓ Trials: {', '.join(n_samples_per_study.keys())}")
print(f"✓ Saved to: data/expression_matrix.csv, data/metadata.csv")
