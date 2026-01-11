# Proteomics Shiny Explorer v2 - Research-Focused Roadmap

## Vision

Transform the proteomics demo app from a **data visualization tool** into a **publication-quality research platform** that enables biological discovery. Focus on statistical rigor, biological interpretation, and support for real clinical trial designs—not Big Data scalability.

**Target Users**: Clinical and translational researchers analyzing proteomics data from:
- Multi-arm clinical trials (Drug A vs Drug B vs Placebo)
- Longitudinal treatment studies (baseline → post-treatment)
- Biomarker discovery cohorts
- Multi-center studies with batch effects

**Design Philosophy**: Every feature should answer a research question or produce a publication-ready figure.

---

## Current State Assessment

### What Works Well ✅
- Clean interface with three logical tabs (Overview, Explorer, Differential)
- Solid basic statistics (t-test, Wilcoxon, FDR correction)
- Good quality control (missingness, PCA)
- Realistic synthetic data with biological effects
- Comprehensive documentation

### Critical Research Gaps ❌

| **Research Need** | **Current App** | **Impact** |
|------------------|----------------|------------|
| **Biological interpretation** | Stops at p-values | Can't publish without pathway analysis |
| **Batch effect correction** | Visible but not corrected | Invalid multi-center results |
| **Missing data handling** | Simple median imputation | Wrong for proteomics (MNAR) |
| **Paired sample analysis** | Treats all independent | Wastes statistical power |
| **Survival analysis** | None | Can't analyze clinical outcomes |
| **Multi-group comparisons** | Two groups only | Can't test Drug A vs B vs Placebo |
| **Clustered heatmaps** | None | Missing standard figure type |

---

## V2 Development Plan: Research-Focused Features

### Phase 1: Biological Interpretation (Weeks 1-3)

**Goal**: Transform lists of proteins into biological insights

#### 1.1 Pathway Enrichment Analysis (Week 1)
**Why Critical**: Reviewers ask "What biological processes are affected?" Lists of proteins aren't interpretable.

**Implementation**:
```r
# R/mod_pathway.R - New module

# Over-Representation Analysis (ORA)
library(clusterProfiler)
library(org.Hs.eg.db)

run_ora <- function(protein_ids, universe_ids, database = "GO_BP") {
  # Convert protein IDs to Entrez IDs
  gene_map <- bitr(protein_ids, fromType = "SYMBOL",
                   toType = "ENTREZID", OrgDb = org.Hs.eg.db)

  # Run enrichment
  enrichGO(gene = gene_map$ENTREZID,
           universe = universe_ids,
           OrgDb = org.Hs.eg.db,
           ont = "BP",  # Biological Process
           pAdjustMethod = "BH",
           pvalueCutoff = 0.05,
           qvalueCutoff = 0.2)
}
```

**UI Components**:
- New tab: "Pathway Enrichment"
- Inputs:
  - Database: GO (BP/MF/CC), KEGG, Reactome
  - p-value threshold (slider: 0.01-0.10, default 0.05)
  - Log2FC threshold (slider: 0.5-2.0, default 1.0)
- Outputs:
  - **Dot plot**: Top 20 enriched pathways (size = gene ratio, color = p-value)
  - **Bar plot**: Top pathways ranked by significance
  - **Gene-pathway heatmap**: Which proteins are in which pathways
  - **Downloadable table**: All enriched terms with gene lists

**Deliverables**:
- clusterProfiler integration
- 3 pathway databases (GO, KEGG, Reactome)
- 3 publication-ready visualizations
- CSV export of enrichment results
- **Estimated effort**: 24 hours

#### 1.2 Gene Set Enrichment Analysis (GSEA) (Week 2)
**Why Important**: Detects subtle coordinated changes that ORA misses.

**Implementation**:
```r
library(fgsea)

run_gsea <- function(ranked_proteins, gene_sets) {
  # ranked_proteins: named vector (protein_id = t-statistic or log2FC)
  # gene_sets: list of gene sets from MSigDB

  fgsea(pathways = gene_sets,
        stats = ranked_proteins,
        minSize = 15,
        maxSize = 500,
        nperm = 10000)
}
```

**UI Components**:
- In "Pathway Enrichment" tab
- Checkbox: "Run GSEA" (in addition to ORA)
- Gene set database dropdown: Hallmark, C2 (curated), C5 (GO)
- Outputs:
  - **GSEA table**: Pathway, ES, NES, p-value, leading edge genes
  - **Enrichment plot**: Running enrichment score for selected pathway
  - **Ridge plot**: Distribution of NES scores

**Deliverables**:
- fgsea integration
- MSigDB gene sets (download on first run)
- Enrichment score plots
- **Estimated effort**: 16 hours

#### 1.3 Protein-Protein Interaction Networks (Week 3)
**Why Important**: Identify functional modules and hub proteins.

**Implementation**:
```r
library(httr)
library(igraph)
library(visNetwork)

get_string_network <- function(protein_ids, species = 9606, score_threshold = 400) {
  # Query STRING API
  url <- "https://string-db.org/api/json/network"
  response <- GET(url, query = list(
    identifiers = paste(protein_ids, collapse = "%0d"),
    species = species,
    required_score = score_threshold
  ))

  # Parse and create igraph object
  data <- content(response)
  edges <- do.call(rbind, lapply(data, function(x) {
    data.frame(from = x$preferredName_A,
               to = x$preferredName_B,
               score = x$score)
  }))

  graph_from_data_frame(edges, directed = FALSE)
}
```

**UI Components**:
- New subtab under "Pathway Enrichment": "Networks"
- Inputs:
  - Confidence score (slider: 0.4-0.9, default 0.7)
  - Top N proteins (slider: 10-100, default 50)
- Outputs:
  - **Interactive network**: visNetwork with node colors = log2FC, size = -log10(p)
  - **Module detection**: Community detection (Louvain algorithm)
  - **Hub analysis**: Top 10 hub proteins by degree/betweenness
  - **Download**: Network as GraphML for Cytoscape

**Deliverables**:
- STRING API integration
- Interactive network visualization
- Community detection
- **Estimated effort**: 20 hours

**Phase 1 Total**: 60 hours, 3 weeks

---

### Phase 2: Statistical Rigor (Weeks 4-6)

**Goal**: Address proteomics-specific statistical challenges

#### 2.1 Batch Effect Correction (Week 4)
**Why Critical**: Multi-center trials (Trial_A/B/C) have technical batch effects that confound biology.

**Implementation**:
```r
library(sva)  # For ComBat
library(limma)  # For removeBatchEffect

# ComBat: Empirical Bayes batch correction
apply_combat <- function(expr_matrix, batch_var, preserve_vars = NULL) {
  # expr_matrix: proteins × samples
  # batch_var: vector of batch labels
  # preserve_vars: biological variables to preserve

  if (!is.null(preserve_vars)) {
    mod <- model.matrix(~ preserve_vars)
  } else {
    mod <- matrix(1, ncol = 1, nrow = ncol(expr_matrix))
  }

  ComBat(dat = expr_matrix,
         batch = batch_var,
         mod = mod,
         par.prior = TRUE,
         prior.plots = FALSE)
}
```

**UI Components**:
- In "Overview" tab, new section: "Batch Correction"
- Inputs:
  - Checkbox: "Apply batch correction"
  - Batch variable: Dropdown (trial, or custom if user uploads)
  - Method: ComBat vs limma removeBatchEffect
  - Variables to preserve: Checklist (treatment, response, timepoint)
- Outputs:
  - **Before/after PCA**: Side-by-side plots colored by batch
  - **RLE plots**: Relative log expression before/after
  - **Apply button**: Commits correction to downstream analyses

**Validation**:
- PCA variance explained by batch should decrease
- Biological signal (treatment, response) should remain

**Deliverables**:
- ComBat and limma methods
- Before/after diagnostic plots
- Applied to all downstream tabs
- **Estimated effort**: 20 hours

#### 2.2 Advanced Missing Data Imputation (Week 5)
**Why Critical**: Proteomics has MNAR (missing not at random) due to low abundance. Simple median imputation is wrong.

**Implementation**:
```r
library(imp4p)  # Proteomics-specific imputation
library(missForest)

# Hybrid imputation strategy
impute_proteomics <- function(expr_matrix, method = "hybrid") {
  if (method == "hybrid") {
    # imp4p: Detects MNAR vs MCAR and applies appropriate method
    impute.wrapper.MLE.MLE(expr_matrix)
  } else if (method == "qrilc") {
    # QRILC: For left-censored MNAR
    impute.QRILC(expr_matrix)
  } else if (method == "missforest") {
    # MissForest: For MCAR
    missForest(t(expr_matrix))$ximp %>% t()
  } else if (method == "minprob") {
    # MinProb: Sample from low end of distribution
    impute.MinProb(expr_matrix)
  }
}
```

**UI Components**:
- In "Overview" tab, section: "Missing Data Handling"
- **Diagnostic plot**: Missingness % vs mean intensity (scatter)
  - MNAR: Negative correlation
  - MCAR: No correlation
- Inputs:
  - Method dropdown: Hybrid (imp4p), QRILC, MissForest, MinProb, Median (current)
  - Apply button
- Outputs:
  - **Before/after distributions**: Density plots
  - **Sensitivity analysis**: Run differential with 2-3 methods, compare overlap

**Deliverables**:
- 5 imputation methods
- Diagnostic plot for missingness mechanism
- Sensitivity comparison tool
- **Estimated effort**: 24 hours

#### 2.3 Multi-Group Comparisons (Week 6)
**Why Important**: Current app only does pairwise (A vs B). Can't test Drug_A vs Drug_B vs Placebo simultaneously.

**Implementation**:
```r
library(limma)

# ANOVA for multi-group
run_anova_limma <- function(expr_matrix, group_var) {
  # Design matrix with group as factor
  design <- model.matrix(~ 0 + group_var)
  colnames(design) <- levels(group_var)

  # Fit linear model
  fit <- lmFit(expr_matrix, design)

  # F-test for overall group differences
  contrast_matrix <- makeContrasts(
    contrasts = combn(levels(group_var), 2,
                     FUN = function(x) paste(x, collapse = " - ")),
    levels = design
  )

  fit2 <- contrasts.fit(fit, contrast_matrix)
  fit2 <- eBayes(fit2)

  # Extract results
  topTable(fit2, number = Inf, sort.by = "F")
}
```

**UI Components**:
- In "Differential Analysis" tab
- Input: "Comparison type"
  - Pairwise (current)
  - Multi-group ANOVA (NEW)
- For ANOVA:
  - Select grouping variable (treatment, response, etc.)
  - Auto-detects if >2 groups
- Outputs:
  - **ANOVA results table**: F-statistic, p-value, adj.p-value
  - **Post-hoc pairwise**: All pairwise comparisons with Tukey correction
  - **Interaction plots**: Mean expression × group

**Deliverables**:
- Limma-based ANOVA
- Post-hoc pairwise tests
- **Estimated effort**: 16 hours

**Phase 2 Total**: 60 hours, 3 weeks

---

### Phase 3: Clinical Trial Support (Weeks 7-9)

**Goal**: Support real experimental designs from clinical studies

#### 3.1 Paired Sample Analysis (Week 7)
**Why Critical**: Longitudinal trials measure same patient before/after treatment. Current app treats as independent, losing statistical power.

**Implementation**:
```r
# Paired differential analysis
run_paired_analysis <- function(data, subject_id_var, group_var) {
  # Ensure paired structure
  paired_data <- data %>%
    group_by(!!sym(subject_id_var), protein_id) %>%
    filter(n_distinct(!!sym(group_var)) == 2) %>%  # Must have both timepoints
    ungroup()

  # Compute delta (change from baseline)
  delta_data <- paired_data %>%
    group_by(!!sym(subject_id_var), protein_id) %>%
    arrange(!!sym(group_var)) %>%
    summarise(delta = last(log2_intensity) - first(log2_intensity))

  # Paired t-test on delta
  delta_data %>%
    group_by(protein_id) %>%
    summarise(
      mean_delta = mean(delta, na.rm = TRUE),
      t_statistic = t.test(delta)$statistic,
      p_value = t.test(delta)$p.value
    ) %>%
    mutate(adj_p_value = p.adjust(p_value, method = "BH"))
}
```

**UI Components**:
- In "Differential Analysis" tab
- Checkbox: "Paired analysis"
  - If checked, requires "Subject ID" column in metadata
- Inputs:
  - Subject ID variable (dropdown)
  - Baseline vs Follow-up variable (dropdown)
- Outputs:
  - **Paired volcano plot**: Delta (change) vs p-value
  - **Slope graph**: Individual subject trajectories for top proteins
  - **Before-after boxplots**: Paired points connected with lines

**Metadata Update**:
- Add `subject_id` column to synthetic data generator
- Example: Subject 001 has Trial_A_S001_Baseline and Trial_A_S001_Week4

**Deliverables**:
- Paired t-test implementation
- Delta (change from baseline) calculation
- Paired-specific visualizations
- **Estimated effort**: 18 hours

#### 3.2 Survival Analysis (Week 8)
**Why Important**: Many clinical trials have time-to-event outcomes (recurrence, death). Need Cox regression and Kaplan-Meier curves.

**Implementation**:
```r
library(survival)
library(survminer)

# Cox regression for protein associations with survival
run_cox_analysis <- function(expr_matrix, time_var, event_var) {
  # expr_matrix: proteins × samples
  # time_var, event_var: vectors matching samples

  results <- apply(expr_matrix, 1, function(protein_expr) {
    cox_fit <- coxph(Surv(time_var, event_var) ~ protein_expr)
    summary_fit <- summary(cox_fit)

    data.frame(
      hazard_ratio = summary_fit$conf.int[1],
      HR_lower = summary_fit$conf.int[3],
      HR_upper = summary_fit$conf.int[4],
      p_value = summary_fit$coefficients[5],
      concordance = summary_fit$concordance[1]
    )
  })

  do.call(rbind, results) %>%
    rownames_to_column("protein_id") %>%
    mutate(adj_p_value = p.adjust(p_value, method = "BH"))
}
```

**UI Components**:
- New tab: "Survival Analysis"
- Metadata requirements:
  - Time-to-event column (days, months, years)
  - Event indicator (0/1 for censored/event)
- Inputs:
  - Time variable (dropdown)
  - Event variable (dropdown)
  - Protein to plot (selectize)
  - Stratification: Median split, tertiles, or quartiles
- Outputs:
  - **Cox results table**: Protein, HR, 95% CI, p-value
  - **Kaplan-Meier plot**: Selected protein, stratified by expression level
  - **Forest plot**: Top 20 proteins with HR and CI
  - **Time-dependent ROC**: AUC at 1-year, 3-year, 5-year

**Metadata Update**:
- Add `time_to_event` and `event_occurred` to synthetic data

**Deliverables**:
- Cox regression for all proteins
- Interactive KM plots
- Forest plot visualization
- **Estimated effort**: 24 hours

#### 3.3 Longitudinal Trajectory Analysis (Week 9)
**Why Important**: Current app has timepoints (Baseline, Week_4, Week_12, Week_24) but doesn't model temporal dynamics.

**Implementation**:
```r
library(lme4)
library(nlme)

# Mixed-effects model: protein ~ time + treatment + time×treatment + (1|subject)
run_longitudinal_model <- function(data, protein_id) {
  protein_data <- data %>% filter(protein_id == !!protein_id)

  # Linear mixed model
  lmer_fit <- lmer(log2_intensity ~ timepoint_numeric * treatment + (1 | subject_id),
                   data = protein_data)

  # Extract fixed effects
  fixed_eff <- fixef(lmer_fit)
  p_values <- summary(lmer_fit)$coefficients[, "Pr(>|t|)"]

  list(
    model = lmer_fit,
    interaction_p = p_values["timepoint_numeric:treatmentDrug_A"],  # Example
    coefficients = fixed_eff
  )
}
```

**UI Components**:
- In "Protein Explorer" tab
- If timepoint data detected:
  - **Profile plot**: Line plot with mean ± SE per treatment group
  - **Spaghetti plot**: Individual subject trajectories (if subject_id available)
- New section in "Differential Analysis":
  - "Longitudinal Analysis"
  - Test for time × treatment interactions
  - Results table: Proteins with significant trajectories

**Deliverables**:
- Mixed-effects models (lme4)
- Profile plots with confidence bands
- Trajectory clustering (k-means on time series)
- **Estimated effort**: 18 hours

**Phase 3 Total**: 60 hours, 3 weeks

---

### Phase 4: Enhanced Visualization (Weeks 10-11)

**Goal**: Add publication-quality figures

#### 4.1 Clustered Heatmaps (Week 10)
**Why Critical**: Standard figure in every proteomics paper. Current app has no heatmap.

**Implementation**:
```r
library(ComplexHeatmap)

create_proteomics_heatmap <- function(expr_matrix, metadata,
                                      top_n = 50, scale_rows = TRUE) {
  # Select top N differential proteins
  top_proteins <- expr_matrix[1:top_n, ]

  # Z-score scaling
  if (scale_rows) {
    top_proteins <- t(scale(t(top_proteins)))
  }

  # Sample annotations
  ha <- HeatmapAnnotation(
    Treatment = metadata$treatment,
    Response = metadata$response,
    Trial = metadata$trial,
    col = list(
      Treatment = c("Drug_A" = "#E41A1C", "Drug_B" = "#377EB8", "Placebo" = "#4DAF4A"),
      Response = c("Responder" = "#FF7F00", "Non-Responder" = "#984EA3", "Partial" = "#FFFF33"),
      Trial = c("Trial_A" = "#A65628", "Trial_B" = "#F781BF", "Trial_C" = "#999999")
    )
  )

  # Create heatmap
  Heatmap(top_proteins,
          name = "Z-score",
          top_annotation = ha,
          show_row_names = TRUE,
          show_column_names = FALSE,
          clustering_distance_rows = "pearson",
          clustering_method_rows = "ward.D2",
          clustering_distance_columns = "pearson",
          clustering_method_columns = "ward.D2",
          col = circlize::colorRamp2(c(-2, 0, 2), c("blue", "white", "red")))
}
```

**UI Components**:
- New tab: "Heatmap"
- Inputs:
  - Top N proteins (slider: 10-100, default 50)
  - Clustering method: Pearson, Euclidean, Spearman
  - Linkage: Complete, Average, Ward
  - Scale: Z-score (recommended), None
  - Annotations: Checkboxes for treatment, response, trial
- Outputs:
  - **Interactive heatmap**: ComplexHeatmap with dendrograms
  - **Row clusters**: Extract clusters, download protein lists
  - **Export**: PNG (300 DPI), PDF (vector)

**Deliverables**:
- ComplexHeatmap integration
- Customizable clustering parameters
- Publication-quality export
- **Estimated effort**: 16 hours

#### 4.2 Additional QC & Diagnostic Plots (Week 11)
**Why Important**: Address QC gaps identified in literature review.

**New Visualizations**:

1. **Sample Correlation Heatmap**
   - Pearson correlation matrix (samples × samples)
   - Hierarchical clustering
   - Identify outliers and batch structure
   - Location: Overview tab

2. **MA Plot**
   - Alternative to volcano plot
   - X-axis: Average expression
   - Y-axis: Log2 fold change
   - Detects intensity-dependent bias
   - Location: Differential tab

3. **Coefficient of Variation (CV) Plot**
   - CV distribution per protein
   - Separate technical vs biological replicates
   - Identify high-variance proteins
   - Location: Overview tab

4. **Relative Log Expression (RLE) Plot**
   - Box plots of sample-wise median-centered expression
   - Should center at zero after normalization
   - Detects normalization failures
   - Location: Overview tab

5. **Missingness Pattern Heatmap**
   - Binary heatmap: present (blue) vs missing (white)
   - Cluster to reveal systematic patterns
   - Location: Overview tab

**Deliverables**:
- 5 new diagnostic plots
- All exportable as PNG/PDF
- **Estimated effort**: 20 hours

**Phase 4 Total**: 36 hours, 2 weeks

---

### Phase 5: Reproducibility & Reporting (Weeks 12-13)

**Goal**: Enable reproducible research and manuscript preparation

#### 5.1 Analysis Report Generation (Week 12)
**Why Important**: Researchers need methods sections and supplementary materials.

**Implementation**:
```r
# R/report_template.Rmd

---
title: "Proteomics Analysis Report"
date: "`r Sys.Date()`"
params:
  data_summary: NULL
  diff_results: NULL
  pathway_results: NULL
  figures: NULL
---

## Dataset Summary
- Number of proteins: `r params$data_summary$n_proteins`
- Number of samples: `r params$data_summary$n_samples`
- Trials: `r paste(params$data_summary$trials, collapse = ", ")`

## Methods
### Data Processing
- Normalization: `r params$methods$normalization`
- Imputation: `r params$methods$imputation`
- Batch correction: `r params$methods$batch_correction`

### Differential Expression
- Statistical test: `r params$methods$test_type`
- Multiple testing correction: Benjamini-Hochberg FDR
- Significance thresholds: adj. p < 0.05, |log2FC| > 1

## Results
### Differential Proteins
`r nrow(params$diff_results %>% filter(adj_p_value < 0.05))` proteins were significantly differential.

### Pathway Enrichment
`r nrow(params$pathway_results %>% filter(p.adjust < 0.05))` pathways were enriched.

## Figures
```{r volcano, echo=FALSE}
params$figures$volcano_plot
```

## Supplementary Tables
```{r diff_table, echo=FALSE}
knitr::kable(params$diff_results %>% arrange(adj_p_value) %>% head(20))
```
```

**UI Components**:
- New tab: "Export & Reports"
- Inputs:
  - Report title
  - Author name
  - Include sections: Checkboxes (QC, Differential, Pathway, etc.)
  - Format: HTML, PDF, Word
- Outputs:
  - Download button: Generates RMarkdown report
  - Preview: Rendered HTML in app

**Deliverables**:
- Parameterized RMarkdown template
- Methods section auto-generation
- Figure + table embedding
- **Estimated effort**: 20 hours

#### 5.2 Session Management & Code Export (Week 13)
**Why Important**: Reproducibility requires saving analysis parameters and code.

**Features**:

1. **Save/Load Session**
   - Save current analysis state as .RData
   - Includes: data, parameters, results
   - Load previous session to continue

2. **Export R Script**
   - Generate standalone R script of analysis
   - All steps from data loading → results
   - Users can modify and re-run

3. **Parameter Logging**
   - JSON file with all analysis choices
   - Threshold values, methods, filters
   - Append to methods section

**UI Components**:
- In "Export & Reports" tab
- Buttons:
  - "Save Session" (.RData download)
  - "Load Session" (upload .RData)
  - "Export R Script" (.R file with commented code)
  - "Download Parameters" (JSON)

**Deliverables**:
- Session save/load functionality
- Code generation from UI actions
- Parameter JSON export
- **Estimated effort**: 16 hours

**Phase 5 Total**: 36 hours, 2 weeks

---

## Implementation Summary

### Total Effort Estimate

| Phase | Focus | Duration | Effort |
|-------|-------|----------|--------|
| 1 | Biological Interpretation | 3 weeks | 60 hours |
| 2 | Statistical Rigor | 3 weeks | 60 hours |
| 3 | Clinical Trial Support | 3 weeks | 60 hours |
| 4 | Enhanced Visualization | 2 weeks | 36 hours |
| 5 | Reproducibility | 2 weeks | 36 hours |
| **Total** | | **13 weeks** | **252 hours** |

### Technology Stack

**Core Framework** (no change from v1):
- R 4.3+
- Shiny 1.8+
- tidyverse

**New Packages Required**:

**Biological Interpretation**:
- clusterProfiler - Pathway enrichment
- fgsea - Gene set enrichment analysis
- org.Hs.eg.db - Human gene annotations
- enrichplot - Enrichment visualization
- igraph, visNetwork - Network analysis

**Statistical Methods**:
- sva - ComBat batch correction
- limma - Moderated statistics, batch correction
- imp4p - Proteomics imputation
- missForest - Random forest imputation

**Clinical Analysis**:
- survival - Cox regression
- survminer - Kaplan-Meier plots
- lme4, nlme - Mixed-effects models

**Visualization**:
- ComplexHeatmap - Advanced heatmaps
- ggpubr - Publication-ready plots
- patchwork - Plot composition

**Reporting**:
- rmarkdown - Report generation
- knitr - Dynamic documents

**Total new dependencies**: ~15 packages (all on CRAN/Bioconductor)

---

## File Structure Changes

```
proteomics-shiny-demo/
├── app.R                          # Main app (will grow to ~800 lines)
├── R/
│   ├── mod_pathway.R              # NEW: Pathway enrichment module
│   ├── mod_network.R              # NEW: PPI network module
│   ├── mod_survival.R             # NEW: Survival analysis module
│   ├── mod_heatmap.R              # NEW: Heatmap module
│   ├── utils_batch_correction.R   # NEW: Batch correction functions
│   ├── utils_imputation.R         # NEW: Imputation functions
│   ├── utils_longitudinal.R       # NEW: Mixed models, trajectories
│   └── generate_data.R            # UPDATED: Add subject_id, survival data
├── inst/
│   ├── rmarkdown/
│   │   └── report_template.Rmd    # NEW: Report template
│   └── extdata/
│       └── gene_sets/             # NEW: MSigDB gene sets (auto-downloaded)
├── data/                          # UPDATED: New metadata columns
│   ├── expression_matrix.csv
│   ├── metadata.csv               # Add: subject_id, time_to_event, event
│   └── proteomics_data.rds
├── docs/
│   ├── CLAUDE.md                  # UPDATED: Document new features
│   ├── V2_RESEARCH_ROADMAP.md     # This file
│   └── METHODS.md                 # NEW: Statistical methods documentation
├── README.md                      # UPDATED: New features, screenshots
└── CITATION.cff                   # NEW: For academic citation
```

---

## Synthetic Data Updates

To support new features, update data generators:

### New Metadata Columns

```r
# In R/generate_data.R and generate_data.py

# Add subject ID for paired analysis
metadata$subject_id <- paste0("Subject_",
  sprintf("%03d", rep(1:(n_total_samples/2), each = 2)))

# Add survival data
metadata$time_to_event <- round(rexp(n_total_samples, rate = 1/365))  # Days
metadata$event_occurred <- rbinom(n_total_samples, 1, prob = 0.3)  # 30% event rate

# Make timepoint numeric for longitudinal models
metadata$timepoint_numeric <- case_when(
  timepoint == "Baseline" ~ 0,
  timepoint == "Week_4" ~ 4,
  timepoint == "Week_12" ~ 12,
  timepoint == "Week_24" ~ 24
)

# Add gene symbols for pathway analysis
# Map UniProt IDs to gene symbols using biomaRt or manual annotation
```

---

## Prioritization: Quick Wins vs Long-Term

### Quick Wins (Weeks 1-2, ~30 hours)

These provide immediate research value with minimal effort:

1. **Pathway Enrichment (ORA)** - 16 hours
   - Most requested feature
   - clusterProfiler is mature and well-documented
   - Transforms protein lists into biology

2. **Sample Correlation Heatmap** - 6 hours
   - Essential QC tool
   - Simple to implement (corrplot or pheatmap)
   - Identifies outliers and batch effects

3. **MA Plot** - 4 hours
   - Alternative to volcano plot
   - Standard proteomics figure
   - Easy addition to differential tab

4. **Clustered Heatmap (basic)** - 8 hours
   - Use pheatmap (simpler than ComplexHeatmap initially)
   - Top N differential proteins
   - Publication figure

**Quick wins total**: 34 hours → Delivers 4 high-impact features in 2 weeks

### Must-Have for Publication (Add weeks 3-6, +60 hours)

5. **Batch Effect Correction** - 20 hours
6. **Advanced Imputation** - 24 hours
7. **Multi-Group ANOVA** - 16 hours

### Advanced Clinical Features (Weeks 7-9, +60 hours)

8. **Paired Analysis** - 18 hours
9. **Survival Analysis** - 24 hours
10. **Longitudinal Models** - 18 hours

---

## Success Metrics

### Research Impact
- [ ] Used in 3+ published proteomics studies within 1 year
- [ ] Cited in methods sections
- [ ] Positive feedback from 10+ researchers

### Feature Completeness
- [ ] Pathway enrichment with 3+ databases
- [ ] Batch effect correction with diagnostics
- [ ] Proteomics-appropriate imputation (4+ methods)
- [ ] Paired and longitudinal analysis support
- [ ] Survival analysis with KM curves
- [ ] 10+ publication-quality visualizations
- [ ] RMarkdown report generation

### Usability
- [ ] Analysis workflow takes <30 minutes for new users
- [ ] All major figures exportable as publication-ready PDFs
- [ ] Comprehensive methods documentation
- [ ] Tutorial with example workflows

### Code Quality
- [ ] 70%+ test coverage for statistical functions
- [ ] All dependencies on CRAN/Bioconductor
- [ ] Works on Windows, Mac, Linux
- [ ] Documented with roxygen2

---

## Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| **Bioconductor dependencies** (clusterProfiler) hard to install | Provide Docker image; document installation carefully |
| **Gene ID mapping** (UniProt → Entrez → Symbol) loses proteins | Include multiple ID types; use biomaRt for on-the-fly conversion |
| **Imputation methods** change results substantially | Provide sensitivity analysis tool; document method choice |
| **Pathway databases** require internet access | Cache MSigDB locally; provide offline mode |
| **Complex heatmaps** slow for >1000 proteins | Default to top 50; warn if selecting >200 |
| **Report generation** fails with missing data | Wrap in tryCatch; provide informative error messages |

---

## Validation Plan

### Statistical Validation
- [ ] Test on published proteomics datasets (ProteomeXchange)
- [ ] Compare results to original papers
- [ ] Verify batch correction reduces technical variance
- [ ] Confirm imputation methods perform as expected

### Usability Testing
- [ ] Beta test with 5 clinical proteomics researchers
- [ ] Collect feedback on workflow and pain points
- [ ] Measure time-to-first-result
- [ ] Iterate based on feedback

### Benchmarking
- [ ] Compare pathway enrichment to standalone tools (David, Enrichr)
- [ ] Validate batch correction against ComBat standalone
- [ ] Ensure statistical results match limma/DESeq2

---

## Documentation Plan

### User Documentation
- **README.md**: Quick start, installation, feature overview
- **Tutorial vignette**: Step-by-step analysis of example dataset
- **FAQ**: Common questions and troubleshooting
- **Video walkthrough**: 15-minute screencast

### Scientific Documentation
- **METHODS.md**: Detailed explanation of all statistical methods
- **REFERENCES.bib**: Citations for algorithms and methods
- **Validation report**: Benchmark against published datasets

### Developer Documentation
- **CLAUDE.md**: Updated for v2 features and architecture
- **API documentation**: Roxygen2 for all functions
- **CONTRIBUTING.md**: How to add new features

---

## Next Steps

### Week 1 Action Items

1. **Set up development environment**
   - Install Bioconductor packages
   - Test clusterProfiler on sample data
   - Download MSigDB gene sets

2. **Update synthetic data**
   - Add subject_id for paired analysis
   - Add survival data (time, event)
   - Add gene symbols for pathway mapping

3. **Implement Quick Win #1: Pathway Enrichment**
   - Create `R/mod_pathway.R`
   - Integrate clusterProfiler
   - Build ORA UI and outputs

4. **Document progress**
   - Update CLAUDE.md with new architecture
   - Create METHODS.md draft
   - Start changelog

### Sprint Planning

**Sprint 1 (Weeks 1-3)**: Biological interpretation
**Sprint 2 (Weeks 4-6)**: Statistical rigor
**Sprint 3 (Weeks 7-9)**: Clinical trial support
**Sprint 4 (Weeks 10-11)**: Visualization enhancements
**Sprint 5 (Weeks 12-13)**: Reproducibility features

Each sprint ends with:
- Demo to stakeholders
- User testing session
- Retrospective and plan adjustment

---

## Conclusion

This research-focused roadmap transforms the proteomics app from a **basic exploration tool** into a **publication-quality analysis platform**. Rather than pursuing Big Data scalability, it addresses the real needs of clinical proteomics researchers:

**Biological interpretation** through pathway enrichment and networks
**Statistical rigor** with batch correction and proteomics-specific imputation
**Clinical trial support** for paired, longitudinal, and survival analyses
**Publication figures** including clustered heatmaps and KM curves
**Reproducibility** through report generation and code export

**Estimated effort**: 252 hours (13 weeks) for full implementation
**Quick wins**: 34 hours (2 weeks) for pathway enrichment + essential QC plots

The modular approach allows incremental delivery, with each sprint adding publishable value. By week 6, the app would support real research workflows and produce methods sections for manuscripts.

---

*Document version: 1.0*
*Last updated: 2026-01-11*
*Author: Research-focused analysis by Claude Code (Sonnet 4.5)*
