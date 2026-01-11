# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A cross-study proteomics visualization and analysis dashboard built with R/Shiny. The application enables quality control assessment, protein-level exploration, and differential expression analysis across multiple clinical trials.

## Common Commands

### Running the Application

```bash
# Generate synthetic data (required on first run or if data/ is missing)
Rscript R/generate_data.R

# Launch the Shiny app
R -e "shiny::runApp('.')"
```

Alternatively, open [app.R](app.R) in RStudio and click "Run App".

### Data Generation

Two implementations are available:

```bash
# R version (recommended - matches app dependencies)
Rscript R/generate_data.R

# Python version (alternative - requires uv)
uv run generate_data.py
```

Both generate:
- `data/expression_matrix.csv` - 500 proteins × 140 samples
- `data/metadata.csv` - Sample annotations
- `data/proteomics_data.rds` - Pre-processed R object (R version only)

## Prerequisites

**R packages:**
```r
install.packages(c("shiny", "tidyverse", "DT", "broom"))
```

**Python (for generate_data.py only):**
- Use `uv` package manager exclusively - never use `pip`
- Install dependencies: `uv pip install numpy pandas`
- Run scripts: `uv run generate_data.py`

## Architecture

### Single-File Shiny App Structure

The entire application is contained in [app.R](app.R) (~630 lines) organized as:

1. **Data Loading (lines 11-21)**: Auto-generates synthetic data if missing, loads pre-processed RDS object containing:
   - `expression_long` - Tidy format (protein_id, sample_id, log2_intensity, metadata columns)
   - `expression_wide` - Matrix format (protein_id as rows, sample_ids as columns)
   - `metadata` - Sample annotations
   - `protein_ids` - Vector of 500 protein identifiers

2. **UI Definition (lines 26-247)**: Three-tab interface using `navbarPage`:
   - **Overview Tab**: Dataset summary, missingness QC, PCA visualization
   - **Protein Explorer Tab**: Single-protein interactive plots with statistical testing
   - **Differential Analysis Tab**: Volcano plots, downloadable results tables

3. **Server Logic (lines 252-624)**: Reactive data flows and computations

### Key Server Architecture Patterns

#### Data Caching & Performance
- **PCA computation** ([app.R:300-325](app.R#L300-L325)): Computed once via `reactive()`, cached for re-rendering with different color/shape options
- **Differential analysis** ([app.R:473-539](app.R#L473-L539)): Uses `eventReactive(input$run_diff)` to compute only on button press
- **Protein selection** ([app.R:357-360](app.R#L357-L360)): Server-side `selectizeInput` to handle 500 proteins efficiently

#### Statistical Methods
- **Missingness**: Per-row/column NA counting ([app.R:272-297](app.R#L272-L297))
- **PCA**: `prcomp()` with row-median imputation for NAs, proteins with >50% missing excluded ([app.R:300-325](app.R#L300-L325))
- **Two-group comparison**: Welch's t-test or Wilcoxon ([app.R:410-437](app.R#L410-L437))
- **Multiple testing correction**: Benjamini-Hochberg FDR ([app.R:530](app.R#L530))

#### Dynamic UI Updates
- **Group selectors** ([app.R:362-367](app.R#L362-L367), [app.R:467-471](app.R#L467-L471)): `observe()` blocks update dropdown choices based on `input$group_by` selection
- Uses `.data[[input$variable]]` pattern for dynamic column reference in dplyr/ggplot

### Data Schema

Expression data is log2-transformed intensity values with ~8% missingness (higher in low-abundance proteins).

**Metadata columns** ([metadata.csv](data/metadata.csv)):
- `sample_id`: `{Trial}_{Sxxx}` format (e.g., `Trial_A_S001`)
- `trial`: `Trial_A`, `Trial_B`, `Trial_C`
- `treatment`: `Drug_A`, `Drug_B`, `Placebo`
- `timepoint`: `Baseline`, `Week_4`, `Week_12`, `Week_24`
- `response`: `Responder`, `Non-Responder`, `Partial`
- `age`, `sex`: Patient demographics

### Synthetic Data Generation

Both [R/generate_data.R](R/generate_data.R) and [generate_data.py](generate_data.py) create identical datasets with:

1. **Base expression**: Normal distribution (mean=10, sd=2) in log2 space
2. **Biological effects**:
   - 50 treatment-responsive proteins (effect size: 0.5-2.0 log2 FC)
   - 40 response-associated proteins (effect size: 0.8-2.5 log2 FC)
   - Trial batch effects (Trial_A=0, Trial_B=+0.3, Trial_C=-0.2)
3. **Missingness**: 8% overall, 16% for low-abundance proteins

## Extending the App

### Adding New Metadata Columns

1. Add column to data generation scripts
2. Update UI dropdown choices (e.g., [app.R:82-86](app.R#L82-L86))
3. Use `.data[[input$variable]]` in server logic for dynamic reference

### Adding Plot Types

Example for MA plot in Differential tab:

```r
output$ma_plot <- renderPlot({
  df <- diff_results()
  ggplot(df, aes(x = (mean_group_a + mean_group_b)/2, y = log2FC)) +
    geom_point(aes(color = significance), alpha = 0.5) +
    labs(x = "Average Expression", y = "Log2 FC")
})
```

### Performance Characteristics

- **Initial load**: <2 seconds for 500×140 dataset
- **Differential analysis**: ~5 seconds (loops through proteins with progress bar [app.R:489-492](app.R#L489-L492))
- **PCA**: Computed once and cached via reactive
