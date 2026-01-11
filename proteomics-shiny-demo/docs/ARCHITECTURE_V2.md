# Proteomics Shiny Explorer v2 - Technical Architecture

## Overview

This document provides detailed technical architecture for the v2 extension, including module design, data flow, reactive architecture, and implementation patterns.

---

## Module Structure

### Design Philosophy
- **Separation of concerns**: UI, server logic, and business logic separated
- **Reusability**: Modules can be composed and reused
- **Testability**: Pure functions extractable for unit testing
- **Namespacing**: Module IDs prevent input/output collisions

### Module Pattern

```r
# R/mod_example.R

#' Example Module UI
#'
#' @param id Module namespace ID
#' @export
mod_example_ui <- function(id) {
  ns <- NS(id)

  tagList(
    selectInput(ns("parameter"), "Select Parameter", choices = c("A", "B", "C")),
    plotOutput(ns("plot")),
    verbatimTextOutput(ns("summary"))
  )
}

#' Example Module Server
#'
#' @param id Module namespace ID
#' @param data Reactive expression returning data frame
#' @return Reactive expression with module outputs
#' @export
mod_example_server <- function(id, data) {
  moduleServer(id, function(input, output, session) {

    # Internal reactive logic
    filtered_data <- reactive({
      req(data())
      data() %>%
        filter(parameter == input$parameter)
    })

    # Outputs
    output$plot <- renderPlot({
      ggplot(filtered_data(), aes(x = value)) +
        geom_histogram()
    })

    output$summary <- renderPrint({
      summary(filtered_data()$value)
    })

    # Return reactive for downstream modules
    reactive({
      list(
        filtered = filtered_data(),
        selected_param = input$parameter
      )
    })
  })
}
```

### Module Communication

**Pattern 1: Parent passes data to child**
```r
# In app server
data <- reactive({ load_data() })
mod_child_server("child", data = data)
```

**Pattern 2: Child returns reactive to parent**
```r
# In app server
child_output <- mod_child_server("child", data = data)

# Use child output in another module
mod_sibling_server("sibling", child_data = child_output)
```

**Pattern 3: Shared reactive values**
```r
# In app server
shared <- reactiveValues(
  current_protein = NULL,
  filters = list()
)

mod_explorer_server("explorer", shared = shared)
mod_differential_server("diff", shared = shared)
```

---

## Proposed Module Hierarchy

```
app.R (root server)
├── mod_data_upload (handles file input, validation)
│   └── Returns: reactive(data_list)
│
├── mod_preprocessing (normalization, imputation, batch correction)
│   ├── Input: data from upload module
│   └── Returns: reactive(processed_data)
│
├── mod_overview (QC and exploration)
│   ├── Input: processed_data
│   ├── Sub-modules:
│   │   ├── mod_summary_stats
│   │   ├── mod_missingness_qc
│   │   ├── mod_sample_correlation
│   │   └── mod_pca
│   └── Returns: reactive(qc_flags)
│
├── mod_explorer (single-protein analysis)
│   ├── Input: processed_data, shared$current_protein
│   ├── Sub-modules:
│   │   ├── mod_protein_selector
│   │   ├── mod_protein_plot
│   │   └── mod_protein_stats
│   └── Returns: reactive(current_protein_data)
│
├── mod_differential (multi-protein comparison)
│   ├── Input: processed_data
│   ├── Sub-modules:
│   │   ├── mod_diff_params (group selection, thresholds)
│   │   ├── mod_volcano_plot (interactive, sets shared$current_protein)
│   │   ├── mod_ma_plot
│   │   └── mod_diff_table
│   └── Returns: reactive(diff_results)
│
├── mod_heatmap (clustering and heatmaps)
│   ├── Input: diff_results or processed_data
│   ├── Sub-modules:
│   │   ├── mod_clustering (hierarchical, k-means)
│   │   └── mod_heatmap_plot
│   └── Returns: reactive(cluster_assignments)
│
├── mod_pathway (enrichment analysis)
│   ├── Input: diff_results
│   ├── Sub-modules:
│   │   ├── mod_ora (over-representation)
│   │   ├── mod_gsea (gene set enrichment)
│   │   └── mod_pathway_plots
│   └── Returns: reactive(enrichment_results)
│
├── mod_network (PPI networks)
│   ├── Input: diff_results
│   └── Returns: reactive(network_graph)
│
├── mod_ml (machine learning)
│   ├── Input: processed_data
│   ├── Sub-modules:
│   │   ├── mod_feature_selection
│   │   ├── mod_model_training
│   │   └── mod_model_evaluation
│   └── Returns: reactive(ml_model)
│
└── mod_reports (export and reproducibility)
    ├── Input: All module outputs
    └── Generates: RMarkdown report
```

---

## Data Flow Architecture

### Reactive Dependency Graph

```
[User Upload] → [Data Validation] → [Data Loading]
                                          ↓
                                    [data_raw] reactive
                                          ↓
                           ┌──────────────┼──────────────┐
                           ↓              ↓              ↓
                    [Normalization] [Imputation] [Batch Correction]
                           └──────────────┬──────────────┘
                                          ↓
                                [processed_data] reactive
                                          ↓
              ┌───────────────────────────┼───────────────────────────┐
              ↓                           ↓                           ↓
        [Overview QC]              [Differential]                [Pathway]
              ↓                           ↓                           ↓
        [PCA Plot]                 [Volcano Plot] ────────> [Protein Explorer]
        [Correlation]              [Results Table]          [Detail Plots]
              │                           │                           │
              └───────────────────────────┴───────────────────────────┘
                                          ↓
                                   [Report Generator]
```

### Key Reactive Principles

**1. Single source of truth**
- `processed_data()` is the canonical reactive
- All analyses derive from this
- Changes propagate automatically

**2. Lazy evaluation**
- Use `reactive()` for automatic re-computation
- Use `eventReactive()` for manual triggers (e.g., "Run Analysis" button)
- Use `isolate()` to break reactive dependencies

**3. Caching expensive operations**
```r
# Bad: Recomputes PCA on every color change
output$pca_plot <- renderPlot({
  pca_result <- prcomp(data(), scale = TRUE)
  plot_pca(pca_result, color = input$color_by)
})

# Good: Computes PCA once, re-renders on color change
pca_result <- reactive({
  prcomp(data(), scale = TRUE)
})

output$pca_plot <- renderPlot({
  plot_pca(pca_result(), color = input$color_by)
})
```

**4. Async for long operations**
```r
# Differential analysis with async
diff_results <- eventReactive(input$run_diff, {
  future_promise({
    run_differential_analysis(
      data = data(),
      group_var = input$group_by,
      group_a = input$group_a,
      group_b = input$group_b
    )
  }) %...>% {
    # This runs when promise resolves
    showNotification("Analysis complete!", type = "message")
    .
  }
})
```

---

## Database Schema

### SQLite Schema for Scalability

```sql
-- Metadata table (N rows, where N = number of samples)
CREATE TABLE metadata (
  sample_id TEXT PRIMARY KEY,
  trial TEXT NOT NULL,
  treatment TEXT,
  timepoint TEXT,
  response TEXT,
  age INTEGER,
  sex TEXT,
  batch TEXT,
  -- Add custom metadata columns as needed
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_metadata_trial ON metadata(trial);
CREATE INDEX idx_metadata_treatment ON metadata(treatment);

-- Proteins table (M rows, where M = number of proteins)
CREATE TABLE proteins (
  protein_id TEXT PRIMARY KEY,
  gene_symbol TEXT,
  uniprot_id TEXT,
  description TEXT,
  organism TEXT DEFAULT 'Homo sapiens',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_proteins_gene ON proteins(gene_symbol);

-- Expression table (N × M rows, long format)
CREATE TABLE expression (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  protein_id TEXT NOT NULL,
  sample_id TEXT NOT NULL,
  log2_intensity REAL,  -- NULL for missing values
  raw_intensity REAL,
  imputed BOOLEAN DEFAULT 0,
  FOREIGN KEY (protein_id) REFERENCES proteins(protein_id),
  FOREIGN KEY (sample_id) REFERENCES metadata(sample_id)
);

CREATE INDEX idx_expression_protein ON expression(protein_id);
CREATE INDEX idx_expression_sample ON expression(sample_id);
CREATE INDEX idx_expression_composite ON expression(protein_id, sample_id);

-- Results cache table (for pre-computed differential analyses)
CREATE TABLE diff_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  protein_id TEXT NOT NULL,
  comparison TEXT NOT NULL,  -- e.g., "Drug_A_vs_Placebo"
  log2fc REAL,
  mean_group_a REAL,
  mean_group_b REAL,
  statistic REAL,
  p_value REAL,
  adj_p_value REAL,
  method TEXT,  -- "t-test", "limma", etc.
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (protein_id) REFERENCES proteins(protein_id)
);

CREATE INDEX idx_diff_comparison ON diff_results(comparison);
CREATE INDEX idx_diff_adjp ON diff_results(adj_p_value);

-- Analysis sessions (for project management)
CREATE TABLE sessions (
  session_id TEXT PRIMARY KEY,
  name TEXT,
  description TEXT,
  parameters TEXT,  -- JSON blob of analysis parameters
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Database Access Pattern

```r
# R/utils_db.R

#' Connect to proteomics database
#' @param db_path Path to SQLite file
#' @return DBI connection
connect_db <- function(db_path = "data/proteomics_data.db") {
  con <- dbConnect(RSQLite::SQLite(), db_path)
  # Enable foreign keys
  dbExecute(con, "PRAGMA foreign_keys = ON")
  con
}

#' Load expression data for specific proteins
#' @param con Database connection
#' @param protein_ids Vector of protein IDs (NULL = all)
#' @param sample_ids Vector of sample IDs (NULL = all)
#' @return Tibble with expression data
load_expression <- function(con, protein_ids = NULL, sample_ids = NULL) {
  query <- "SELECT e.*, m.trial, m.treatment, m.response
            FROM expression e
            JOIN metadata m ON e.sample_id = m.sample_id"

  conditions <- c()
  if (!is.null(protein_ids)) {
    conditions <- c(conditions, sprintf("e.protein_id IN ('%s')",
                                       paste(protein_ids, collapse = "','")))
  }
  if (!is.null(sample_ids)) {
    conditions <- c(conditions, sprintf("e.sample_id IN ('%s')",
                                       paste(sample_ids, collapse = "','")))
  }

  if (length(conditions) > 0) {
    query <- paste(query, "WHERE", paste(conditions, collapse = " AND "))
  }

  dbGetQuery(con, query) %>% as_tibble()
}

#' Cache differential results
#' @param con Database connection
#' @param results Data frame with diff results
#' @param comparison Comparison name
cache_diff_results <- function(con, results, comparison) {
  results_db <- results %>%
    mutate(comparison = comparison,
           method = "limma",
           created_at = Sys.time())

  dbWriteTable(con, "diff_results", results_db, append = TRUE)
}

#' Retrieve cached results if available
#' @param con Database connection
#' @param comparison Comparison name
#' @return Tibble or NULL if not cached
get_cached_diff_results <- function(con, comparison) {
  query <- sprintf("SELECT * FROM diff_results WHERE comparison = '%s'", comparison)
  result <- dbGetQuery(con, query)

  if (nrow(result) == 0) return(NULL)
  as_tibble(result)
}
```

---

## Statistical Functions Layer

### Separation of Statistics from UI

```r
# R/utils_stats.R

#' Two-group differential analysis
#'
#' @param data Data frame with columns: protein_id, sample_id, log2_intensity, group
#' @param group_a Name of group A
#' @param group_b Name of group B
#' @param method "t-test", "wilcox", or "limma"
#' @return Data frame with columns: protein_id, log2fc, p_value, etc.
#' @export
run_two_group_analysis <- function(data, group_a, group_b,
                                    method = c("t-test", "wilcox", "limma")) {
  method <- match.arg(method)

  # Validate inputs
  stopifnot(
    "data must be a data frame" = is.data.frame(data),
    "Required columns missing" = all(c("protein_id", "log2_intensity", "group") %in% names(data)),
    "Groups not found" = all(c(group_a, group_b) %in% data$group)
  )

  # Filter to groups of interest
  data_filtered <- data %>%
    filter(group %in% c(group_a, group_b))

  # Dispatch to method
  switch(method,
    "t-test" = run_t_test(data_filtered, group_a, group_b),
    "wilcox" = run_wilcox(data_filtered, group_a, group_b),
    "limma" = run_limma(data_filtered, group_a, group_b)
  )
}

#' Welch's t-test for all proteins
#' @keywords internal
run_t_test <- function(data, group_a, group_b) {
  data %>%
    group_by(protein_id) %>%
    summarise(
      n_a = sum(group == group_a, na.rm = TRUE),
      n_b = sum(group == group_b, na.rm = TRUE),
      mean_a = mean(log2_intensity[group == group_a], na.rm = TRUE),
      mean_b = mean(log2_intensity[group == group_b], na.rm = TRUE),
      log2fc = mean_a - mean_b,

      # Only test if >=2 samples per group
      test_result = if_else(n_a >= 2 & n_b >= 2, {
        test <- tryCatch(
          t.test(log2_intensity[group == group_a],
                 log2_intensity[group == group_b]),
          error = function(e) NULL
        )
        if (!is.null(test)) list(statistic = test$statistic, p_value = test$p.value)
        else list(statistic = NA_real_, p_value = NA_real_)
      }, list(statistic = NA_real_, p_value = NA_real_))
    ) %>%
    unnest_wider(test_result) %>%
    mutate(adj_p_value = p.adjust(p_value, method = "BH"))
}

#' Limma moderated t-test (vectorized, faster)
#' @keywords internal
run_limma <- function(data, group_a, group_b) {
  require(limma)

  # Reshape to matrix (proteins × samples)
  expr_matrix <- data %>%
    select(protein_id, sample_id, log2_intensity) %>%
    pivot_wider(names_from = sample_id, values_from = log2_intensity) %>%
    column_to_rownames("protein_id") %>%
    as.matrix()

  # Design matrix
  group_vec <- data %>%
    distinct(sample_id, group) %>%
    arrange(match(sample_id, colnames(expr_matrix))) %>%
    mutate(group = factor(group, levels = c(group_b, group_a))) %>%
    pull(group)

  design <- model.matrix(~ group_vec)

  # Fit linear model
  fit <- lmFit(expr_matrix, design)
  fit <- eBayes(fit)

  # Extract results
  topTable(fit, coef = 2, number = Inf, sort.by = "none") %>%
    rownames_to_column("protein_id") %>%
    as_tibble() %>%
    select(
      protein_id,
      log2fc = logFC,
      avg_expr = AveExpr,
      t_statistic = t,
      p_value = P.Value,
      adj_p_value = adj.P.Val,
      b_statistic = B  # Log-odds of differential expression
    )
}

#' ANOVA for multi-group comparison
#' @param data Data frame
#' @param groups Vector of group names to compare
#' @return Data frame with F-statistics and p-values
#' @export
run_anova <- function(data, groups) {
  data %>%
    filter(group %in% groups) %>%
    group_by(protein_id) %>%
    summarise(
      n_groups = n_distinct(group),
      aov_result = if_else(n_groups >= 2, {
        fit <- tryCatch(
          aov(log2_intensity ~ group),
          error = function(e) NULL
        )
        if (!is.null(fit)) {
          summary_fit <- summary(fit)[[1]]
          list(
            f_statistic = summary_fit$"F value"[1],
            p_value = summary_fit$"Pr(>F)"[1],
            df_between = summary_fit$Df[1],
            df_within = summary_fit$Df[2]
          )
        } else {
          list(f_statistic = NA, p_value = NA, df_between = NA, df_within = NA)
        }
      }, list(f_statistic = NA, p_value = NA, df_between = NA, df_within = NA))
    ) %>%
    unnest_wider(aov_result) %>%
    mutate(adj_p_value = p.adjust(p_value, method = "BH"))
}
```

---

## Normalization & Preprocessing

### Preprocessing Pipeline

```r
# R/utils_normalize.R

#' Apply normalization to expression data
#'
#' @param data Expression data (long or wide format)
#' @param method "median", "quantile", "vsn", "loess", or "none"
#' @param format "long" or "wide"
#' @return Normalized data in same format as input
#' @export
normalize_expression <- function(data,
                                  method = c("median", "quantile", "vsn", "loess", "none"),
                                  format = c("long", "wide")) {
  method <- match.arg(method)
  format <- match.arg(format)

  if (method == "none") return(data)

  # Convert to matrix if long format
  if (format == "long") {
    data_wide <- data %>%
      select(protein_id, sample_id, log2_intensity) %>%
      pivot_wider(names_from = sample_id, values_from = log2_intensity)

    expr_matrix <- data_wide %>%
      select(-protein_id) %>%
      as.matrix()
    rownames(expr_matrix) <- data_wide$protein_id
  } else {
    expr_matrix <- data %>%
      select(-protein_id) %>%
      as.matrix()
    rownames(expr_matrix) <- data$protein_id
  }

  # Apply normalization
  expr_norm <- switch(method,
    "median" = normalize_median(expr_matrix),
    "quantile" = normalize_quantile(expr_matrix),
    "vsn" = normalize_vsn(expr_matrix),
    "loess" = normalize_loess(expr_matrix)
  )

  # Convert back to original format
  if (format == "long") {
    expr_norm %>%
      as.data.frame() %>%
      rownames_to_column("protein_id") %>%
      pivot_longer(-protein_id, names_to = "sample_id", values_to = "log2_intensity") %>%
      left_join(data %>% select(-log2_intensity), by = c("protein_id", "sample_id"))
  } else {
    expr_norm %>%
      as.data.frame() %>%
      rownames_to_column("protein_id")
  }
}

#' Median normalization
#' @keywords internal
normalize_median <- function(mat) {
  medians <- apply(mat, 2, median, na.rm = TRUE)
  global_median <- median(medians, na.rm = TRUE)
  sweep(mat, 2, medians - global_median, "-")
}

#' Quantile normalization
#' @keywords internal
normalize_quantile <- function(mat) {
  require(preprocessCore)
  normalize.quantiles(mat, copy = TRUE)
}

#' VSN normalization
#' @keywords internal
normalize_vsn <- function(mat) {
  require(vsn)
  fit <- vsn2(mat)
  predict(fit, mat)
}
```

---

## Performance Optimization Strategies

### 1. Vectorization
```r
# SLOW: Loop through proteins
results <- list()
for (i in seq_along(proteins)) {
  results[[i]] <- compute_stat(data[protein == proteins[i]])
}
results_df <- bind_rows(results)

# FAST: Use group_by + summarise (vectorized)
results_df <- data %>%
  group_by(protein_id) %>%
  summarise(stat = compute_stat(log2_intensity))
```

### 2. Parallel Processing
```r
library(future)
plan(multisession, workers = 4)  # Use 4 cores

# Parallel differential analysis
diff_results <- future_lapply(unique(data$protein_id), function(prot) {
  data_prot <- data %>% filter(protein_id == prot)
  compute_differential(data_prot, group_a, group_b)
}, future.seed = TRUE) %>%
  bind_rows()
```

### 3. Database Lazy Loading
```r
# SLOW: Load all data into memory
data <- dbGetQuery(con, "SELECT * FROM expression")

# FAST: Use dbplyr for lazy evaluation
expr_tbl <- tbl(con, "expression")

# Queries are translated to SQL and executed on database
results <- expr_tbl %>%
  filter(sample_id %in% selected_samples) %>%
  left_join(tbl(con, "metadata"), by = "sample_id") %>%
  collect()  # Only fetch results at the end
```

### 4. Memoization
```r
library(memoise)

# Cache expensive PCA computation
pca_cached <- memoise(function(data_matrix) {
  prcomp(data_matrix, scale = TRUE, center = TRUE)
})

# First call: Computes PCA
result1 <- pca_cached(expr_matrix)

# Second call with same input: Returns cached result
result2 <- pca_cached(expr_matrix)  # Instant!
```

---

## Testing Strategy

### Unit Tests Example

```r
# tests/testthat/test-utils_stats.R

test_that("run_two_group_analysis returns correct structure", {
  # Create test data
  test_data <- tibble(
    protein_id = rep(c("P1", "P2"), each = 10),
    sample_id = paste0("S", 1:20),
    log2_intensity = rnorm(20, mean = 10, sd = 2),
    group = rep(c("A", "B"), 10)
  )

  # Run analysis
  result <- run_two_group_analysis(test_data, "A", "B", method = "t-test")

  # Check structure
  expect_s3_class(result, "data.frame")
  expect_named(result, c("protein_id", "n_a", "n_b", "mean_a", "mean_b",
                         "log2fc", "statistic", "p_value", "adj_p_value"))
  expect_equal(nrow(result), 2)  # 2 proteins

  # Check values
  expect_true(all(!is.na(result$log2fc)))
  expect_true(all(result$p_value >= 0 & result$p_value <= 1))
})

test_that("run_two_group_analysis handles missing data", {
  test_data <- tibble(
    protein_id = rep("P1", 10),
    sample_id = paste0("S", 1:10),
    log2_intensity = c(rep(NA, 5), rnorm(5)),
    group = rep(c("A", "B"), each = 5)
  )

  result <- run_two_group_analysis(test_data, "A", "B", method = "t-test")

  # Should return NA for p-value since group A is all NA
  expect_true(is.na(result$p_value))
})

test_that("normalize_expression median method works", {
  test_matrix <- matrix(rnorm(100, mean = 10), nrow = 10, ncol = 10)
  rownames(test_matrix) <- paste0("P", 1:10)
  colnames(test_matrix) <- paste0("S", 1:10)

  # Add systematic bias to some columns
  test_matrix[, 1:5] <- test_matrix[, 1:5] + 2

  # Normalize
  test_df <- as.data.frame(test_matrix) %>% rownames_to_column("protein_id")
  normalized <- normalize_expression(test_df, method = "median", format = "wide")

  # Check that column medians are now equal
  norm_matrix <- normalized %>% select(-protein_id) %>% as.matrix()
  medians <- apply(norm_matrix, 2, median, na.rm = TRUE)

  expect_true(sd(medians) < 0.01)  # Should be nearly identical
})
```

### Integration Tests Example

```r
# tests/testthat/test-modules.R

library(shinytest2)

test_that("Differential module updates on button click", {
  app <- AppDriver$new(app_dir = "../..")

  # Select groups
  app$set_inputs(group_by = "treatment")
  app$set_inputs(group_a = "Drug_A")
  app$set_inputs(group_b = "Placebo")

  # Click run button
  app$click("run_diff")

  # Wait for results
  app$wait_for_idle(timeout = 10000)

  # Check that results table is populated
  table_data <- app$get_value(output = "diff_table")
  expect_true(nrow(table_data) > 0)

  # Check that volcano plot was rendered
  expect_true(app$get_screenshot("volcano_plot") is not NULL)
})
```

---

## Deployment Considerations

### Dockerization

```dockerfile
# docker/Dockerfile
FROM rocker/shiny:4.3.2

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install R packages
RUN R -e "install.packages(c('shiny', 'tidyverse', 'DT', 'broom', \
    'limma', 'ComplexHeatmap', 'clusterProfiler', 'dbplyr', \
    'RSQLite', 'future', 'promises', 'shinyWidgets'))"

# Copy app files
COPY . /srv/shiny-server/proteomics-app

# Set working directory
WORKDIR /srv/shiny-server/proteomics-app

# Expose port
EXPOSE 3838

# Run app
CMD ["R", "-e", "shiny::runApp(port=3838, host='0.0.0.0')"]
```

### Environment Configuration

```r
# R/config.R

#' Load configuration from environment variables or defaults
#' @return List of configuration parameters
load_config <- function() {
  list(
    # Database
    db_path = Sys.getenv("PROTEOMICS_DB_PATH", "data/proteomics_data.db"),
    db_type = Sys.getenv("PROTEOMICS_DB_TYPE", "sqlite"),  # sqlite or postgres

    # Performance
    max_workers = as.integer(Sys.getenv("PROTEOMICS_MAX_WORKERS", "4")),
    enable_caching = as.logical(Sys.getenv("PROTEOMICS_ENABLE_CACHE", "TRUE")),
    cache_dir = Sys.getenv("PROTEOMICS_CACHE_DIR", "~/.cache/proteomics_app"),

    # UI
    max_protein_display = as.integer(Sys.getenv("PROTEOMICS_MAX_DISPLAY", "100")),
    default_theme = Sys.getenv("PROTEOMICS_THEME", "flatly"),

    # Analysis defaults
    default_fdr = as.numeric(Sys.getenv("PROTEOMICS_FDR", "0.05")),
    default_fc = as.numeric(Sys.getenv("PROTEOMICS_FC", "1.0")),
    min_samples_per_group = as.integer(Sys.getenv("PROTEOMICS_MIN_SAMPLES", "3"))
  )
}
```

---

## Summary

This architecture provides:

1. **Modularity**: Shiny modules with clear interfaces and namespacing
2. **Scalability**: Database backend with lazy loading and async computation
3. **Testability**: Pure functions extracted from modules for unit testing
4. **Maintainability**: Separation of UI, server, and business logic
5. **Performance**: Vectorization, parallelization, caching, and database optimization
6. **Flexibility**: Configuration via environment variables for deployment

**Next steps**:
1. Implement Phase 1.1 (architecture refactor) using these patterns
2. Write comprehensive tests for statistical functions
3. Benchmark performance improvements
4. Document all modules with roxygen2

