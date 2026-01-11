# Proteomics Shiny Explorer v2 - Extension Roadmap

> **📌 Note**: This document focuses on **scalability and architecture** for Big Data proteomics (20,000+ proteins).
>
> **For research-focused features** (statistical methods, biological interpretation, clinical trial support), see **[V2_RESEARCH_ROADMAP.md](V2_RESEARCH_ROADMAP.md)** instead.

## Executive Summary

This document outlines a comprehensive v2 extension plan based on expert analysis of the v1 codebase, focusing on **scalability** and **production deployment** for large proteomics datasets.

**Current State**: 500 proteins × 140 samples, 630-line single-file app, basic QC/exploration/differential analysis
**Target State**: Production-ready platform supporting 20,000+ proteins, modular architecture, database backend, async computation

**⚠️ Most researchers should follow the [Research-Focused Roadmap](V2_RESEARCH_ROADMAP.md) instead**, which prioritizes:
- Pathway enrichment and biological interpretation
- Batch effect correction
- Advanced missing data imputation
- Paired/longitudinal/survival analysis
- Publication-quality visualizations

---

## Critical Findings from v1 Analysis

### Strengths to Preserve
- ✅ Clean reactive architecture with proper caching
- ✅ Excellent synthetic data generation with realistic biological effects
- ✅ Well-designed PCA visualization with proper preprocessing
- ✅ Statistical rigor (FDR correction, sample size validation)
- ✅ Comprehensive documentation (CLAUDE.md)

### Critical Gaps to Address
- ❌ **Scalability**: Current loop-based differential analysis would take 3-4 minutes for 20,000 proteins
- ❌ **Statistical depth**: Only two-group t-tests; no ANOVA, linear models, batch correction, or survival analysis
- ❌ **Workflow completeness**: Missing normalization, imputation, filtering (0% of preprocessing pipeline)
- ❌ **Pathway analysis**: No enrichment testing, network visualization, or functional annotation (0% coverage)
- ❌ **Data management**: No project saving, multi-dataset support, or reproducibility features
- ❌ **Interactivity**: Volcano plot brush defined but unused; no linked visualizations

---

## V2 Development Phases

### Phase 1: Foundation & Quick Wins (Month 1)

**Goal**: Modularize architecture and add high-impact visualizations

#### 1.1 Architecture Refactor (Week 1-2)
**Rationale**: Current 630-line monolithic app.R will become unmaintainable beyond 1,000 lines

**Tasks**:
- [ ] Split into Shiny modules:
  - `R/mod_overview.R` - QC and PCA (currently lines 252-353)
  - `R/mod_explorer.R` - Single-protein analysis (lines 357-464)
  - `R/mod_differential.R` - Volcano plots and tables (lines 473-623)
- [ ] Extract helper functions:
  - `R/utils_data.R` - Data loading, validation, transformations
  - `R/utils_stats.R` - Statistical tests, effect sizes, FDR correction
  - `R/utils_plot.R` - Common plot themes, color palettes
- [ ] Create `app.R` entry point (<50 lines that loads modules)
- [ ] Add `tests/testthat/` directory with unit tests for statistical functions

**Deliverables**:
- Modular codebase with clear separation of concerns
- 80%+ test coverage for statistical functions
- Documentation for each module

#### 1.2 Enhanced Visualizations (Week 2-3)
**Rationale**: Address immediate gaps in QC and exploration capabilities

**Tasks**:
- [ ] **MA Plot** (replaces volcano for certain use cases)
  - Plot: Average expression (x) vs Log2 FC (y)
  - Identifies intensity-dependent bias in differential analysis
  - Location: New output in Differential tab
  - Estimated effort: 3 hours

- [ ] **Sample Correlation Heatmap** (critical QC missing from v1)
  - Plot: Pearson correlation matrix of all samples
  - Hierarchical clustering with dendrograms
  - Color annotations by trial/treatment
  - Use `ComplexHeatmap` package
  - Location: New subtab under Overview
  - Estimated effort: 6 hours

- [ ] **Intensity Distribution QC**
  - Plot: Boxplot of log2 intensity per sample
  - Identifies normalization issues and outliers
  - Add violin overlay for distribution shape
  - Location: Overview tab, next to missingness plots
  - Estimated effort: 3 hours

- [ ] **Top Proteins Heatmap**
  - Plot: Clustered heatmap of top N differential proteins
  - Row/column annotations for metadata
  - Interactive selection of N (slider: 10-100)
  - Location: New tab "Heatmap"
  - Estimated effort: 8 hours

**Deliverables**:
- 4 new publication-ready visualizations
- Enhanced QC capabilities matching professional tools
- All plots exportable as PNG/PDF/SVG

#### 1.3 Interactivity Enhancements (Week 3-4)
**Rationale**: Volcano plot brush is declared (line 203) but unused; major missed opportunity

**Tasks**:
- [ ] **Volcano plot → Protein Explorer linking**
  - Implement brush selection handler
  - Update `input$selected_protein` on click
  - Switch to Protein Explorer tab automatically
  - Highlight selected protein in results table
  - Estimated effort: 4 hours

- [ ] **Plot export buttons**
  - Add download button above each plot
  - Options: PNG (300 DPI), PDF (vector), SVG
  - Custom sizing controls (width, height)
  - Estimated effort: 6 hours

- [ ] **Plotly conversion** (optional, for zoom/pan/tooltip)
  - Convert static ggplot2 to interactive plotly
  - Add hover tooltips with protein annotations
  - Zoom/pan for exploration
  - Trade-off: Slower rendering for large datasets
  - Estimated effort: 12 hours

**Deliverables**:
- Linked brushing between volcano and explorer
- Exportable plots in 3 formats
- (Optional) Interactive plotly versions

#### 1.4 Basic Filtering Controls (Week 4)
**Rationale**: Current app analyzes all data; users need subsetting

**Tasks**:
- [ ] **Sidebar filter panel**
  - Multi-select inputs for trial, treatment, timepoint
  - Apply to all downstream analyses
  - Display filtered sample count in real-time
  - Reset button to clear filters

- [ ] **Missingness threshold**
  - Slider: "Exclude proteins with >X% missing" (default 50%)
  - Applied before differential analysis
  - Update protein count dynamically

- [ ] **Detection frequency filter**
  - "Require ≥X% detected in at least one group"
  - Common in proteomics to remove rarely detected proteins

**Deliverables**:
- Flexible data subsetting without code
- Common proteomics filtering workflows

**Phase 1 Success Metrics**:
- Codebase modular and testable
- 7+ new visualizations
- Interactive volcano → explorer workflow
- Basic filtering capabilities
- Estimated total effort: **80-100 hours**

---

### Phase 2: Statistical Depth (Month 2)

**Goal**: Expand beyond two-group t-tests to professional-grade statistics

#### 2.1 Multi-Group ANOVA (Week 1)
**Rationale**: Current app only compares 2 groups; clinical trials need multi-arm comparisons

**Tasks**:
- [ ] Implement one-way ANOVA for N-group comparisons
  - Use `aov()` for F-test
  - Post-hoc pairwise tests with Tukey HSD
  - Effect size: eta-squared
- [ ] Two-way ANOVA for interaction effects
  - Treatment × Timepoint interactions
  - Visualization: Interaction plots
- [ ] Update UI to detect 3+ groups and switch to ANOVA
- [ ] Results table: F-statistic, p-value, pairwise comparisons

**Deliverables**:
- Multi-group comparison capability
- Interaction effect testing
- Post-hoc pairwise results with multiple testing correction

#### 2.2 Limma Integration (Week 1-2)
**Rationale**: Limma's moderated t-statistics borrow information across proteins for better power

**Tasks**:
- [ ] Install and integrate `limma` package
- [ ] Replace loop-based t-tests with `limma::lmFit` → `eBayes`
- [ ] Compute moderated F-statistics and log-odds
- [ ] Volcano plot: Option to toggle classic t-test vs. limma
- [ ] Performance comparison: Benchmark speed improvement
- [ ] Results table: Add B-statistic (log-odds of differential expression)

**Deliverables**:
- 10-100x faster differential analysis (vectorized)
- More powerful statistics for small sample sizes
- Direct comparison of methods

#### 2.3 Normalization Module (Week 2)
**Rationale**: Current app assumes pre-normalized data; raw data needs preprocessing

**Tasks**:
- [ ] **Normalization methods**:
  - Median normalization (divide by sample median)
  - Quantile normalization (make distributions identical)
  - VSN (variance-stabilizing normalization)
  - Loess normalization (intensity-dependent)
  - Option: None (skip normalization)

- [ ] **UI controls**:
  - Dropdown in Overview tab: Select method
  - Before/after comparison plots (side-by-side boxplots)
  - Apply button to commit normalization
  - Warning if data already normalized

- [ ] **Implementation**:
  - Use `preprocessCore` package for quantile
  - Use `vsn` package for VSN
  - Custom functions for median/loess
  - Cache normalized data in reactive

**Deliverables**:
- 4 normalization methods
- Visual validation of normalization effects
- Applied to all downstream analyses

#### 2.4 Batch Effect Correction (Week 2-3)
**Rationale**: Trial effects visible in PCA (lines 85-88 of generate_data.R); need correction

**Tasks**:
- [ ] **ComBat** (gold standard for batch correction)
  - Use `sva::ComBat` function
  - Specify trial as batch variable
  - Preserve treatment/response differences
  - Before/after PCA comparison

- [ ] **Limma removeBatchEffect** (alternative)
  - Use `limma::removeBatchEffect`
  - Faster but less rigorous than ComBat
  - Good for visualization, not for differential stats

- [ ] **UI integration**:
  - Checkbox: "Apply batch correction"
  - Dropdown: Select batch variable (trial, etc.)
  - Dropdown: Method (ComBat, limma, none)
  - Real-time PCA update

**Deliverables**:
- Batch correction with 2 methods
- Visual validation via PCA
- Option to apply or bypass

#### 2.5 Advanced Imputation (Week 3)
**Rationale**: Current median imputation is simplistic; proteomics needs MNAR-aware methods

**Tasks**:
- [ ] **Imputation methods**:
  - MinProb (draw from low end of distribution)
  - kNN (k-nearest neighbors in protein space)
  - QRILC (quantile regression for left-censored data)
  - MLE (maximum likelihood estimation)
  - Median (current method, keep as baseline)

- [ ] **UI**:
  - Sidebar in Overview: Select imputation method
  - Apply before analysis (warn if applied mid-session)
  - Display: % values imputed per method

- [ ] **Validation**:
  - Compare distributions of imputed vs. observed
  - Sensitivity analysis: Run differential with each method

**Deliverables**:
- 5 imputation methods
- Validation plots
- Method comparison tools

#### 2.6 Covariate Adjustment (Week 4)
**Rationale**: Age and sex are in metadata but unused; important confounders in clinical data

**Tasks**:
- [ ] Linear model with covariates
  - Formula: `expression ~ treatment + age + sex`
  - Use `lm()` or `limma::lmFit`
  - Extract treatment effect adjusted for covariates
- [ ] UI: Checklist to include/exclude covariates
- [ ] Results: Adjusted vs. unadjusted fold changes

**Deliverables**:
- Covariate-adjusted differential analysis
- Comparison of adjusted vs. unadjusted results

**Phase 2 Success Metrics**:
- Multi-group and interaction testing
- Limma integration with 10x+ speed improvement
- 4 normalization methods
- 2 batch correction methods
- 5 imputation methods
- Covariate adjustment capability
- Estimated total effort: **120-150 hours**

---

### Phase 3: Scalability & Performance (Month 3)

**Goal**: Support 20,000+ proteins and 500+ samples with sub-second interactions

#### 3.1 Database Backend (Week 1-2)
**Rationale**: Current in-memory approach won't scale to 20K proteins × 500 samples = 10M rows

**Tasks**:
- [ ] **SQLite integration** (simpler, single-file)
  - Create database schema on data load
  - Tables: `expression`, `metadata`, `proteins`
  - Indexes on sample_id, protein_id
  - Use `DBI` and `RSQLite` packages

- [ ] **Lazy loading with dbplyr**
  - Replace `expr_long` reactive with `tbl(con, "expression")`
  - Filter/join operations translated to SQL
  - Only fetch data when needed (collect() at last step)

- [ ] **Chunk processing for differential**
  - Process proteins in batches of 1000
  - Write results to database table
  - Stream results to UI as computed

- [ ] **(Optional) PostgreSQL support**
  - For multi-user deployments
  - Connection pooling with `pool` package
  - Environment-based config for credentials

**Deliverables**:
- SQLite backend for local use
- Lazy loading with dbplyr
- (Optional) PostgreSQL for production
- 10x+ scalability improvement

#### 3.2 Async Computation (Week 2-3)
**Rationale**: Long operations (differential, normalization) block Shiny; need non-blocking

**Tasks**:
- [ ] **Future/promises integration**
  - Install `future`, `promises`, `ipc` packages
  - Wrap differential analysis in `future_promise()`
  - Update progress bar asynchronously
  - Display spinner while computing

- [ ] **Background job for normalization**
  - Offload to separate R process
  - Poll for completion
  - Load results when ready

- [ ] **Parallel differential analysis**
  - Use `future.apply::future_lapply` to parallelize protein loop
  - Detect cores: `availableCores()`
  - Progress updates via IPC

**Deliverables**:
- Non-blocking UI during long computations
- Parallel processing across CPU cores
- Responsive progress indicators

#### 3.3 Code Optimization (Week 3)
**Rationale**: Vectorize remaining loops; use efficient data structures

**Tasks**:
- [ ] **Vectorized differential analysis**
  - Replace for-loop (lines 489-525) with `lapply` or `purrr::map`
  - Or use limma (already vectorized)
  - Benchmark: Compare old vs. new speed

- [ ] **data.table for filtering**
  - Convert `expr_long` to data.table
  - Use `[i, j, by]` syntax for fast filtering
  - Binary search on keys (sample_id, protein_id)

- [ ] **Memoization for expensive functions**
  - Cache PCA results across sessions with `memoise`
  - Store in `~/.cache/proteomics_app/`
  - Invalidate on data changes (hash-based)

**Deliverables**:
- 5-10x faster filtering and aggregation
- Persistent caching of PCA and correlations
- Benchmark report showing improvements

#### 3.4 UI Virtualization (Week 4)
**Rationale**: DT tables with 20,000 rows are slow; need virtual scrolling

**Tasks**:
- [ ] **DT server-side processing**
  - Use `server = TRUE` in `renderDataTable`
  - Render only visible rows
  - Backend filtering/sorting via SQL

- [ ] **Protein dropdown optimization**
  - Current: maxOptions=100 (line 359)
  - Alternative: Server-side search with `selectizeInput(server=TRUE)`
  - Only load matching proteins on search

- [ ] **Plot rendering optimization**
  - Downsample points for scatter plots >10,000 points
  - Use `scattermore` for fast rendering
  - WebGL acceleration for plotly

**Deliverables**:
- Smooth UI with 20,000+ proteins
- Instant table filtering and sorting
- Fast plot rendering for large datasets

**Phase 3 Success Metrics**:
- Support 20,000 proteins × 500 samples
- Sub-second filtering and exploration
- Non-blocking async computations
- Database-backed storage for scalability
- Estimated total effort: **100-120 hours**

---

### Phase 4: Advanced Analytics (Month 4)

**Goal**: Match professional platforms with pathway analysis, ML, and multi-omics

#### 4.1 Pathway Enrichment (Week 1-2)
**Rationale**: Differential proteins need functional interpretation; current app stops at p-values

**Tasks**:
- [ ] **Over-representation analysis (ORA)**
  - Use `clusterProfiler` package
  - Fisher's exact test for GO/KEGG enrichment
  - Upload custom gene sets (GMT format)
  - Input: Top N differential proteins (user-defined threshold)

- [ ] **Gene Set Enrichment Analysis (GSEA)**
  - Rank all proteins by t-statistic
  - Test for enrichment at top/bottom of ranked list
  - Use `fgsea` package (fast implementation)
  - Hallmark, GO, KEGG, Reactome gene sets

- [ ] **Visualization**:
  - Dot plot: Enriched pathways (size=gene ratio, color=p-value)
  - Bar plot: Top 20 pathways
  - Network plot: Pathway-pathway similarity
  - Gene-pathway heatmap

- [ ] **UI tab**: "Pathway Enrichment"
  - Input: Adjusted p-value threshold (0.05)
  - Input: Log2FC threshold (1.0)
  - Input: Gene set database (GO BP/MF/CC, KEGG, custom)
  - Output: Enrichment table (downloadable)
  - Output: Plots (dot, bar, network)

**Deliverables**:
- ORA and GSEA with 4+ gene set databases
- 4 enrichment visualization types
- Custom gene set upload
- Downloadable enrichment results

#### 4.2 Network Visualization (Week 2)
**Rationale**: Protein-protein interactions add biological context

**Tasks**:
- [ ] **STRING integration**
  - Query STRING API for PPI networks
  - Input: Top differential proteins
  - Output: Network graph (nodes=proteins, edges=interactions)

- [ ] **Network visualization**:
  - Use `visNetwork` or `igraph` + `ggraph`
  - Color nodes by log2FC (red/blue gradient)
  - Size nodes by -log10(p-value)
  - Edge width by STRING confidence score

- [ ] **Interactive features**:
  - Click node to view protein in Explorer
  - Highlight neighbors on hover
  - Filter edges by confidence threshold

**Deliverables**:
- STRING PPI networks
- Interactive network plots
- Protein-level drill-down from network

#### 4.3 Clustering & Heatmaps (Week 2-3)
**Rationale**: Current heatmap (Phase 1.2) is basic; need advanced clustering

**Tasks**:
- [ ] **Hierarchical clustering**
  - Distance metrics: Euclidean, correlation, Manhattan
  - Linkage methods: Complete, average, ward.D2
  - Optimal cluster number: Gap statistic, silhouette

- [ ] **K-means clustering**
  - Elbow plot to select k
  - Cluster membership table
  - Profile plots: Mean trajectory per cluster

- [ ] **Biclustering** (optional advanced)
  - Cluster proteins AND samples simultaneously
  - Use `biclust` package
  - Identify co-regulated protein modules

- [ ] **Enhanced heatmap features**:
  - Split rows/columns by clusters
  - Annotation tracks: Treatment, response, trial
  - Color scales: Viridis, RdBu, custom
  - Export: PNG, PDF, and cluster membership CSV

**Deliverables**:
- Hierarchical and k-means clustering
- Advanced heatmap with annotations
- Cluster export and profile plots

#### 4.4 Machine Learning Module (Week 3-4)
**Rationale**: Predict response from protein expression; identify biomarker panels

**Tasks**:
- [ ] **Feature selection**:
  - Recursive feature elimination (RFE)
  - LASSO regression (glmnet)
  - Random forest variable importance
  - Top N proteins by p-value (baseline)

- [ ] **Classification models**:
  - Random forest (`randomForest` or `ranger`)
  - SVM (`e1071` or `kernlab`)
  - Logistic regression with elastic net
  - Naive Bayes (baseline)

- [ ] **Cross-validation**:
  - K-fold CV (5 or 10 folds)
  - Leave-one-out CV for small N
  - Stratified splits to balance classes
  - Repeated CV for stability

- [ ] **Evaluation metrics**:
  - Accuracy, precision, recall, F1
  - ROC curve and AUC
  - Confusion matrix
  - Variable importance plot

- [ ] **UI tab**: "Machine Learning"
  - Sidebar: Select outcome (response, treatment)
  - Select features: All proteins, top N, custom set
  - Select model: RF, SVM, LASSO
  - Select CV: k-fold, LOOCV
  - Outputs: ROC curve, confusion matrix, top features

**Deliverables**:
- 4 classification algorithms
- Cross-validation framework
- ROC curves and performance metrics
- Biomarker panel identification

#### 4.5 Time-Series Analysis (Week 4)
**Rationale**: Timepoint data exists but unused; need longitudinal models

**Tasks**:
- [ ] **Linear mixed models**
  - Use `lme4::lmer` for repeated measures
  - Random intercepts per patient
  - Fixed effects: Treatment, time, treatment×time
  - Extract time-dependent treatment effects

- [ ] **Profile plots**
  - Line plots: Mean protein expression over time
  - Separate lines by treatment group
  - Confidence bands (±SE)
  - Facet by top N proteins

- [ ] **Trajectory clustering**
  - K-means on time-series profiles
  - Identify proteins with similar dynamics
  - Cluster plot: Spaghetti plot with cluster means

**Deliverables**:
- Mixed model analysis for longitudinal data
- Profile plots with confidence bands
- Trajectory clustering

**Phase 4 Success Metrics**:
- Pathway enrichment with ORA and GSEA
- PPI network visualization
- Advanced clustering (hierarchical, k-means, biclustering)
- Machine learning with 4 algorithms
- Time-series analysis with mixed models
- Estimated total effort: **120-150 hours**

---

### Phase 5: Production Features (Month 5+)

**Goal**: Project management, reproducibility, and collaboration

#### 5.1 Project Management (Week 1-2)
**Tasks**:
- [ ] Save/load sessions as RData or RDS
- [ ] Project comparison mode (load 2 projects side-by-side)
- [ ] Annotation system: Add notes to proteins
- [ ] Bookmark proteins of interest
- [ ] Audit trail: Log of all actions taken

#### 5.2 Reporting System (Week 2-3)
**Tasks**:
- [ ] RMarkdown template generation
- [ ] Automated PDF reports with all plots
- [ ] Customizable sections (include/exclude analyses)
- [ ] Parameter logging for reproducibility
- [ ] Export R script of analysis workflow

#### 5.3 Multi-Dataset Integration (Week 3-4)
**Tasks**:
- [ ] Upload multiple datasets
- [ ] Meta-analysis across studies
- [ ] Batch effect visualization (MDS, PCA)
- [ ] Integrated differential analysis

#### 5.4 Collaboration Features (Week 4+)
**Tasks**:
- [ ] User authentication (shinymanager)
- [ ] Share analysis links
- [ ] Comment threads on proteins
- [ ] Version control for analyses

---

## Technology Stack for V2

### Core Framework
- **R 4.3+**: Base language
- **Shiny 1.8+**: Web framework with modules
- **tidyverse**: Data manipulation (dplyr, ggplot2, tidyr)
- **bslib/thematic**: Modern UI theming

### Data Management
- **DBI/RSQLite**: Database backend
- **dbplyr**: Lazy evaluation
- **data.table**: Fast in-memory operations
- **arrow/fst**: Disk-based storage for large data

### Statistics
- **limma**: Moderated statistics, batch correction
- **sva**: ComBat batch correction
- **preprocessCore**: Quantile normalization
- **vsn**: Variance stabilization
- **lme4/nlme**: Mixed models
- **survival**: Cox regression, Kaplan-Meier

### Visualization
- **ComplexHeatmap**: Advanced heatmaps
- **pheatmap**: Simple heatmaps
- **ggraph/igraph**: Networks
- **plotly**: Interactive plots
- **patchwork**: Plot composition

### Pathway/Enrichment
- **clusterProfiler**: GO/KEGG enrichment
- **fgsea**: Fast GSEA
- **enrichR**: Multiple databases
- **org.Hs.eg.db**: Human annotations

### Machine Learning
- **caret**: Unified ML interface
- **randomForest/ranger**: RF models
- **glmnet**: LASSO/elastic net
- **e1071**: SVM
- **pROC**: ROC curves

### Performance
- **future/promises**: Async computation
- **memoise**: Caching
- **profvis**: Profiling
- **bench**: Benchmarking

### Testing & Quality
- **testthat**: Unit tests
- **shinytest2**: UI tests
- **lintr**: Code linting
- **styler**: Code formatting

### Deployment
- **renv**: Package management
- **Docker**: Containerization
- **shinyproxy/RStudio Connect**: Multi-user hosting

---

## File Structure for V2

```
proteomics-shiny-demo/
├── app.R                          # Entry point (~50 lines)
├── DESCRIPTION                    # Package metadata
├── NAMESPACE                      # Exports
├── renv.lock                      # Dependency versions
│
├── R/
│   ├── mod_overview.R             # QC and PCA module
│   ├── mod_explorer.R             # Single-protein module
│   ├── mod_differential.R         # Differential analysis module
│   ├── mod_heatmap.R              # Clustering and heatmaps
│   ├── mod_pathway.R              # Enrichment analysis
│   ├── mod_network.R              # PPI networks
│   ├── mod_ml.R                   # Machine learning
│   ├── mod_timeseries.R           # Longitudinal analysis
│   │
│   ├── utils_data.R               # Data loading, validation
│   ├── utils_stats.R              # Statistical tests
│   ├── utils_plot.R               # Plotting helpers
│   ├── utils_normalize.R          # Normalization methods
│   ├── utils_impute.R             # Imputation methods
│   ├── utils_db.R                 # Database operations
│   │
│   ├── global.R                   # Global variables, options
│   └── run_app.R                  # Launch function
│
├── inst/
│   ├── app/
│   │   └── www/                   # Static assets (CSS, JS, images)
│   ├── rmarkdown/
│   │   └── report_template.Rmd    # Report template
│   └── extdata/
│       └── gene_sets/             # Custom GMT files
│
├── data-raw/
│   ├── generate_data.R            # R data generator (v1)
│   └── generate_data.py           # Python data generator (v1)
│
├── data/                          # Generated data (gitignored)
│   ├── expression_matrix.csv
│   ├── metadata.csv
│   └── proteomics_data.db         # SQLite database (v2)
│
├── tests/
│   └── testthat/
│       ├── test-utils_stats.R     # Unit tests for stats
│       ├── test-utils_normalize.R
│       └── test-modules.R         # Integration tests
│
├── docs/
│   ├── CLAUDE.md                  # AI assistant guide (v1)
│   ├── V2_ROADMAP.md              # This file
│   ├── API.md                     # Function documentation
│   └── TUTORIAL.md                # User tutorial
│
├── docker/
│   ├── Dockerfile                 # Container definition
│   └── docker-compose.yml         # Multi-service setup
│
├── .github/
│   └── workflows/
│       └── R-CMD-check.yml        # CI/CD pipeline
│
├── README.md                      # User-facing overview
├── .gitignore
└── .Rbuildignore
```

---

## Performance Targets

| Metric | v1 (Current) | v2 (Target) | Improvement |
|--------|-------------|-------------|-------------|
| Max proteins | 500 | 20,000 | 40x |
| Max samples | 140 | 1,000 | 7x |
| Diff analysis time (500p) | 5s | 0.5s | 10x |
| Diff analysis time (20Kp) | ~180s (est.) | 5s | 36x |
| Initial load time | 2s | 2s | Same |
| Memory footprint | ~200 MB | ~500 MB | 2.5x |
| Filtering latency | Instant | Instant | Same |
| PCA computation | 1s | 1s | Same |

---

## Testing Strategy

### Unit Tests (testthat)
- Statistical functions (t-test, ANOVA, limma)
- Normalization methods (median, quantile, VSN)
- Imputation methods (MinProb, kNN, QRILC)
- Data validation (schema checking)
- Target: 80%+ code coverage

### Integration Tests (shinytest2)
- Module interactions (filter → differential)
- Reactive chains (data → stats → plot)
- UI state management
- File upload/download

### Performance Tests (bench)
- Differential analysis scaling (100, 1K, 10K, 20K proteins)
- Database query performance
- Plot rendering speed
- Memory profiling (profvis)

### User Acceptance Testing
- Biologist beta testers
- Usability study with 5-10 users
- Collect feedback on workflows

---

## Migration Path from v1 to v2

### Backwards Compatibility
- [ ] v1 CSV data format fully supported
- [ ] v1 synthetic data generation scripts unchanged
- [ ] Option to run in "v1 mode" (single-file app) for demos

### Migration Script
```r
# migrate_v1_to_v2.R
# Converts v1 RDS to v2 SQLite database

library(DBI)
library(RSQLite)

# Load v1 data
v1_data <- readRDS("data/proteomics_data.rds")

# Create v2 database
con <- dbConnect(SQLite(), "data/proteomics_data.db")

# Write tables
dbWriteTable(con, "expression", v1_data$expression_long)
dbWriteTable(con, "metadata", v1_data$metadata)
dbWriteTable(con, "proteins", data.frame(protein_id = v1_data$protein_ids))

# Create indexes
dbExecute(con, "CREATE INDEX idx_expression_sample ON expression(sample_id)")
dbExecute(con, "CREATE INDEX idx_expression_protein ON expression(protein_id)")

dbDisconnect(con)
message("Migration complete: data/proteomics_data.db")
```

---

## Documentation Plan

### User Documentation
- **README.md**: Quick start, installation, basic usage
- **TUTORIAL.md**: Step-by-step analysis workflows
- **FAQ.md**: Common questions and troubleshooting
- **VIDEO**: Screencast walkthrough (15 minutes)

### Developer Documentation
- **CLAUDE.md**: AI assistant context (updated for v2)
- **API.md**: Function reference with roxygen2
- **ARCHITECTURE.md**: Module structure, reactive flow
- **CONTRIBUTING.md**: How to add new features

### Scientific Documentation
- **METHODS.md**: Statistical methods explained
- **REFERENCES.md**: Citations for algorithms
- **VALIDATION.md**: Benchmark against published datasets

---

## Resource Requirements

### Development Team
- **Lead developer**: Full-stack R/Shiny (1 FTE for 5 months)
- **Bioinformatics consultant**: Method validation (0.2 FTE)
- **UX designer**: UI/workflow optimization (0.1 FTE)
- **Tester**: QA and beta testing (0.2 FTE)

### Infrastructure
- **Development**: Local machine + GitHub
- **Testing**: GitHub Actions (free tier)
- **Staging**: DigitalOcean droplet ($20/month)
- **Production**: RStudio Connect or ShinyProxy ($$$)

### Budget Estimate
- Development: $50K-80K (contractor rates)
- Infrastructure: $2K/year
- Tools/licenses: $1K (RStudio, DO, misc)
- Total Year 1: **$53K-83K**

---

## Success Criteria

### Functional Requirements
- [ ] Supports 20,000 proteins × 500 samples
- [ ] <5 second differential analysis for 20K proteins
- [ ] 15+ visualization types
- [ ] 10+ statistical methods
- [ ] Pathway enrichment with 3+ databases
- [ ] Machine learning with 4+ algorithms
- [ ] Project save/load
- [ ] Report generation

### Quality Requirements
- [ ] 80%+ test coverage
- [ ] Zero critical bugs in production
- [ ] <1% error rate in user sessions
- [ ] Accessible (WCAG 2.1 AA)
- [ ] Mobile-responsive (>768px width)

### User Satisfaction
- [ ] 90%+ positive feedback from beta testers
- [ ] <30 min time-to-first-result for new users
- [ ] <5 support requests per week
- [ ] Used in 3+ published studies

### Performance
- [ ] Sub-second UI interactions
- [ ] <2s initial load time
- [ ] Handles 10 concurrent users
- [ ] <500 MB memory per session

---

## Risks & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Scope creep beyond 5 months | High | High | Phase-gate reviews; MVP first |
| Performance targets not met | Medium | High | Early benchmarking; optimization sprints |
| Limma integration too complex | Low | Medium | Prototype in Phase 2.2; fallback to t-test |
| Database migration breaks existing workflows | Medium | Medium | Keep v1 CSV support; thorough testing |
| User adoption low | Medium | High | Beta testing; tutorial videos; published paper |
| Maintenance burden after launch | High | Medium | Comprehensive docs; modular code; CI/CD |

---

## Next Steps

### Immediate Actions (Week 1)
1. **Stakeholder approval**: Present roadmap to team/advisor
2. **Set up v2 branch**: `git checkout -b v2-development`
3. **Initialize package structure**: Use `usethis::create_package()`
4. **Set up renv**: Lock dependencies with `renv::init()`
5. **Create GitHub project board**: Track Phase 1 tasks
6. **Write Phase 1.1 architecture spike**: Prototype module structure

### First Sprint (Week 2-3)
- Complete Phase 1.1: Architecture refactor
- Add 2-3 new visualizations (MA plot, sample correlation)
- Set up CI/CD pipeline
- Write first batch of unit tests

### Review Cadence
- **Weekly**: Progress review, blocker discussion
- **End of Phase**: Demo, retrospective, plan next phase
- **Monthly**: Stakeholder update, budget check

---

## Conclusion

This v2 roadmap transforms the proteomics-shiny-demo from an educational proof-of-concept into a production-ready analysis platform. The phased approach balances quick wins (Phase 1) with deep statistical enhancements (Phase 2), scalability (Phase 3), and advanced analytics (Phase 4).

**Key takeaway**: The v1 architecture is solid for a demo but requires modularization, database backend, and async computation to scale. The statistical methods need expansion from two-group t-tests to ANOVA, limma, batch correction, and machine learning. Pathway enrichment and network visualization are critical for biological interpretation.

**Estimated total effort**: 420-520 hours over 5 months
**Expected outcome**: Platform matching 70-80% of commercial proteomics tools at zero software cost

**Questions or feedback?** Open an issue or contact the development team.

---

*Document version: 1.0*
*Last updated: 2026-01-11*
*Author: Expert analysis by Claude Code (Sonnet 4.5)*
