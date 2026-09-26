# Embedded Feature Selection for a Tiny Genomic Cohort (n = 28)

A step-by-step, heavily commented Jupyter tutorial for bioinformatics researchers. It covers
**penalized regression (Lasso / Elastic Net)**, **nested leave-one-out cross-validation**,
**permutation tests** and **stability selection**, applied to imputed SNP dosages and a continuous phenotype.

📓 **Notebook:** [`penalized_feature_selection_small_cohort.ipynb`](penalized_feature_selection_small_cohort.ipynb)
(committed with all outputs, figures and logs, so you can read it without running it).

## What makes this tutorial different

- **Ground truth you can check.** Genomes are simulated with the coalescent (`msprime`) under the published
  Gutenkunst *et al.* (2009) Out-of-Africa model. Imputation is simulated as posterior-mean dosages with Minimac-style Rsq,
  and the phenotype has three known causal variants. Every method is scored against the truth, not only against
  cross-validation.
- **Grounded in real-world numbers.** Mutation and recombination rates, LD decay, imputation-quality filters and
  effect sizes of large-effect traits (warfarin dose, Lp(a)) are taken from the literature. A table in Section 2 cites each one.
- **Pure Python.** QC, LD pruning (PLINK-style `--indep-pairwise`), in-fold LD clumping, reference-panel PCA and
  VCF/`.raw` loaders are written out in NumPy, so the logic is visible. Equivalent PLINK 2 commands are given in Section 21.
- **Honest validation at n = 28.** Every step that touches the phenotype (covariate adjustment, clumping, GWAS
  ranking, λ/ρ tuning) runs inside each outer LOO fold. A logged, transparent engine shows the choices made in every fold.

## Pipeline

```
QC (MAF ≥ 0.10, Rsq ≥ 0.8, HWE) → LD pruning r² < 0.1 (reference LD)  or  in-fold LD clumping
  → covariates (sex, age, projected PC1; unpenalized via Frisch–Waugh–Lovell)
  → nested LOOCV [outer: leave-one-out | inner: repeated 5-fold tuning of Elastic Net λ, ρ ∈ {0.7, 0.8, 0.9}]
  → Freedman–Lane permutation test (whole pipeline re-run) → complementary-pairs stability selection
```

## Headline results (from the committed run)

The simulated trait has 3 causal variants explaining 30% / 15% / 8% of the variance. "True R²" is the final model scored on 472 held-out
people, which is possible only in simulation.

| Strategy (all nested LOOCV, n = 28) | LOOCV R² | ΔR² vs covariates | Permutation p | True R² |
|---|---|---|---|---|
| Genome-wide, LD-pruned, Elastic Net (λ_1SE) | −0.37 | −0.26 | 0.51 | 0.03 |
| Genome-wide, LD-pruned, Elastic Net (≤5 SNPs) | −0.11 | +0.00 | 0.32 | −0.06 |
| Genome-wide, GWAS top-3 + Ridge | −0.26 | −0.15 | 0.41 | −0.03 |
| Genome-wide, in-fold clumping + EN (≤5 SNPs) | −0.03 | +0.08 | 0.15 | 0.00 |
| **Candidate genes, in-fold clumping + EN (≤5 SNPs)** | **+0.26** | **+0.36** | **0.01** | **0.28** |
| Candidate genes, GWAS top-3 + Ridge | +0.21 | +0.32 | 0.02 | 0.21 |

**Take-home messages.** At n = 28, genome-wide selection cannot beat chance correlations. LD pruning can discard the causal
signal, while clumping inside each fold keeps it. The data cannot choose the model size, so cap it in advance. A
pre-specified candidate-gene prior (which includes decoys) is the lever that works. Three of the 4 SNPs in the final model tag the three causal
variants, and at the gene level stability selection separates the true genes from the decoys.

## Contents (22 sections)

1. Setup, configuration and logging (one `CFG` dict and a log file)
2. Real-world grounding and a **power calculation for n = 28**
3. Coalescent simulation of human genomes
4. Imputation dosages, Rsq and regression calibration
5. Oligogenic phenotype with known truth, plus a hidden validation set
6. Variant QC at n = 28 (minor-allele *count*)
7. LD intuition: chance and structure-induced r² at n = 28
8. LD pruning, and why **in-fold clumping** keeps the causal signal
9. Reference-panel PCA and covariates
10. The math: OLS → Ridge → Lasso → Elastic Net (soft-thresholding, geometry, grouping, FWL)
11. Lasso vs Elastic Net inside a real LD block
12. Why LOOCV, why *nested*, and why the null LOOCV R² is −0.075
13. A transparent nested-LOOCV engine
14. Model comparison (EN, Lasso, 1-SE rule, ≤5-SNP cap, clumping, GWAS-top-k + Ridge, shallow RF/Extra Trees, oracle)
15. **Shrinking the search space with biological priors** (candidate genes with decoys)
16. Error bars: bootstrap, external truth, independent replicate cohorts
17. Freedman–Lane permutation test
18. Complementary-pairs stability selection (SNP-level and gene-level)
19. Negative control: a purely polygenic trait
20. Final model and a reporting checklist
21. Running it on your data (PLINK `.raw` / VCF `DS` loaders)
22. Take-home messages and references

## Run it

```bash
pip install -r requirements.txt
jupyter lab penalized_feature_selection_small_cohort.ipynb          # full run ≈ 20 min on 4 cores
TUTORIAL_FAST=1 jupyter nbconvert --to notebook --execute \
    penalized_feature_selection_small_cohort.ipynb --output quick_run.ipynb   # ≈ 7 min
```

The notebook writes its log (`tutorial_run.log`), result tables (CSV) and a reporting summary to `outputs/`.
