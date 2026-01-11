# Quick Start Guide: Implementing V2 Research Features

This guide helps you start implementing the [V2 Research Roadmap](../V2_RESEARCH_ROADMAP.md) with the highest-impact features first.

---

## Week 1-2: Quick Wins (34 hours)

These 4 features provide immediate research value and are easy to implement.

### 1. Pathway Enrichment Analysis (16 hours)

**What it does**: Converts differential protein lists into biological pathways (GO, KEGG, Reactome).

**Installation**:
```r
# In R console
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install(c("clusterProfiler", "org.Hs.eg.db", "enrichplot"))
```

**Implementation skeleton**:
```r
# Create R/mod_pathway.R

library(clusterProfiler)
library(org.Hs.eg.db)
library(enrichplot)

#' Pathway Enrichment Module UI
mod_pathway_ui <- function(id) {
  ns <- NS(id)

  tagList(
    h3("Pathway Enrichment Analysis"),

    fluidRow(
      column(4,
        wellPanel(
          selectInput(ns("database"), "Database:",
                     choices = c("GO Biological Process" = "BP",
                               "GO Molecular Function" = "MF",
                               "GO Cellular Component" = "CC",
                               "KEGG Pathways" = "KEGG")),
          numericInput(ns("p_cutoff"), "P-value cutoff:", 0.05, 0.001, 0.1, 0.01),
          numericInput(ns("fc_cutoff"), "Log2FC cutoff:", 1.0, 0.5, 3.0, 0.1),
          actionButton(ns("run_enrich"), "Run Enrichment", class = "btn-primary")
        )
      ),

      column(8,
        tabsetPanel(
          tabPanel("Dot Plot", plotOutput(ns("dotplot"), height = "600px")),
          tabPanel("Bar Plot", plotOutput(ns("barplot"), height = "600px")),
          tabPanel("Enrichment Table",
                  downloadButton(ns("download_table"), "Download CSV"),
                  DTOutput(ns("enrich_table")))
        )
      )
    )
  )
}

#' Pathway Enrichment Module Server
mod_pathway_server <- function(id, diff_results) {
  moduleServer(id, function(input, output, session) {

    # Run enrichment when button clicked
    enrich_result <- eventReactive(input$run_enrich, {
      req(diff_results())

      # Filter significant proteins
      sig_proteins <- diff_results() %>%
        filter(abs(log2fc) > input$fc_cutoff,
               adj_p_value < input$p_cutoff) %>%
        pull(protein_id)

      if (length(sig_proteins) < 3) {
        showNotification("Too few significant proteins for enrichment", type = "warning")
        return(NULL)
      }

      withProgress(message = "Running enrichment analysis...", {
        # Convert gene symbols to Entrez IDs
        gene_map <- bitr(sig_proteins,
                        fromType = "SYMBOL",
                        toType = "ENTREZID",
                        OrgDb = org.Hs.eg.db)

        # Run enrichment
        if (input$database %in% c("BP", "MF", "CC")) {
          enrichGO(gene = gene_map$ENTREZID,
                  OrgDb = org.Hs.eg.db,
                  ont = input$database,
                  pAdjustMethod = "BH",
                  pvalueCutoff = 0.05,
                  qvalueCutoff = 0.2,
                  readable = TRUE)
        } else {
          enrichKEGG(gene = gene_map$ENTREZID,
                    organism = "hsa",
                    pAdjustMethod = "BH",
                    pvalueCutoff = 0.05,
                    qvalueCutoff = 0.2)
        }
      })
    })

    # Dot plot
    output$dotplot <- renderPlot({
      req(enrich_result())
      dotplot(enrich_result(), showCategory = 20)
    })

    # Bar plot
    output$barplot <- renderPlot({
      req(enrich_result())
      barplot(enrich_result(), showCategory = 20)
    })

    # Enrichment table
    output$enrich_table <- renderDT({
      req(enrich_result())
      as.data.frame(enrich_result())
    })

    # Download handler
    output$download_table <- downloadHandler(
      filename = function() {
        paste0("pathway_enrichment_", Sys.Date(), ".csv")
      },
      content = function(file) {
        write.csv(as.data.frame(enrich_result()), file, row.names = FALSE)
      }
    )
  })
}
```

**Add to app.R**:
```r
# In UI section, add new tab:
tabPanel("Pathway Enrichment",
         icon = icon("project-diagram"),
         mod_pathway_ui("pathway"))

# In server section:
mod_pathway_server("pathway", diff_results = diff_results)
```

---

### 2. Sample Correlation Heatmap (6 hours)

**What it does**: QC tool to identify outliers, batch effects, and sample mix-ups.

**Implementation**:
```r
# Add to Overview tab in app.R

# In UI:
h4("Sample Correlation Heatmap"),
plotOutput("sample_correlation", height = "500px"),

# In server:
output$sample_correlation <- renderPlot({
  # Get expression matrix (samples as columns)
  expr_matrix <- proteomics_data$expression_wide %>%
    select(-protein_id) %>%
    as.matrix()

  # Compute correlation
  cor_matrix <- cor(expr_matrix, use = "pairwise.complete.obs", method = "pearson")

  # Create heatmap with pheatmap
  library(pheatmap)

  # Annotation
  annotation_col <- proteomics_data$metadata %>%
    select(sample_id, trial, treatment, response) %>%
    column_to_rownames("sample_id")

  pheatmap(cor_matrix,
           color = colorRampPalette(c("blue", "white", "red"))(100),
           breaks = seq(0.5, 1, length.out = 101),
           clustering_distance_rows = "euclidean",
           clustering_distance_cols = "euclidean",
           annotation_col = annotation_col,
           annotation_row = annotation_col,
           show_rownames = FALSE,
           show_colnames = FALSE,
           main = "Sample-Sample Correlation")
})
```

---

### 3. MA Plot (4 hours)

**What it does**: Alternative to volcano plot, detects intensity-dependent bias.

**Implementation**:
```r
# Add to Differential Analysis tab

# In UI, add to tabsetPanel:
tabPanel("MA Plot",
         plotOutput("ma_plot", height = "500px", brush = "ma_brush"))

# In server:
output$ma_plot <- renderPlot({
  req(diff_results())

  df <- diff_results() %>%
    mutate(
      avg_expr = (mean_group_a + mean_group_b) / 2,
      significance = case_when(
        adj_p_value < 0.05 & abs(log2fc) > 1 ~ "Significant",
        TRUE ~ "Not Significant"
      )
    )

  ggplot(df, aes(x = avg_expr, y = log2fc, color = significance)) +
    geom_point(alpha = 0.5, size = 1.5) +
    geom_hline(yintercept = c(-1, 1), linetype = "dashed", color = "gray50") +
    geom_hline(yintercept = 0, color = "black") +
    scale_color_manual(values = c("Significant" = "red", "Not Significant" = "gray70")) +
    labs(
      x = "Average Log2 Expression",
      y = "Log2 Fold Change",
      title = paste(input$group_a, "vs", input$group_b),
      color = NULL
    ) +
    theme_minimal() +
    theme(legend.position = "top")
})
```

---

### 4. Basic Clustered Heatmap (8 hours)

**What it does**: Standard figure in proteomics papers, shows top differential proteins clustered.

**Implementation**:
```r
# Add new tab to app.R

# In UI:
tabPanel("Heatmap",
         icon = icon("th"),
         fluidRow(
           column(3,
             wellPanel(
               sliderInput("heatmap_n", "Number of proteins:",
                          min = 10, max = 100, value = 50, step = 10),
               selectInput("heatmap_scale", "Scaling:",
                          choices = c("Z-score (row)" = "row",
                                    "None" = "none")),
               actionButton("create_heatmap", "Create Heatmap", class = "btn-primary")
             )
           ),
           column(9,
             plotOutput("clustered_heatmap", height = "700px")
           )
         ))

# In server:
output$clustered_heatmap <- renderPlot({
  req(input$create_heatmap)

  isolate({
    # Get top N differential proteins
    top_proteins <- diff_results() %>%
      arrange(adj_p_value) %>%
      head(input$heatmap_n) %>%
      pull(protein_id)

    # Get expression matrix for these proteins
    expr_matrix <- proteomics_data$expression_wide %>%
      filter(protein_id %in% top_proteins) %>%
      column_to_rownames("protein_id") %>%
      as.matrix()

    # Metadata for annotation
    annotation_col <- proteomics_data$metadata %>%
      select(sample_id, trial, treatment, response) %>%
      column_to_rownames("sample_id")

    # Color scheme for annotations
    ann_colors <- list(
      trial = c("Trial_A" = "#E41A1C", "Trial_B" = "#377EB8", "Trial_C" = "#4DAF4A"),
      treatment = c("Drug_A" = "#984EA3", "Drug_B" = "#FF7F00", "Placebo" = "#A65628"),
      response = c("Responder" = "#F781BF", "Non-Responder" = "#999999", "Partial" = "#66C2A5")
    )

    # Create heatmap
    pheatmap(expr_matrix,
             scale = input$heatmap_scale,
             clustering_distance_rows = "correlation",
             clustering_distance_cols = "correlation",
             clustering_method = "ward.D2",
             annotation_col = annotation_col,
             annotation_colors = ann_colors,
             color = colorRampPalette(c("blue", "white", "red"))(100),
             show_colnames = FALSE,
             fontsize_row = 8,
             main = paste("Top", input$heatmap_n, "Differential Proteins"))
  })
})
```

---

## Testing Quick Wins

After implementing these 4 features, test with the synthetic data:

```bash
# Generate data
Rscript R/generate_data.R

# Run app
R -e "shiny::runApp('.')"
```

**Expected results**:
1. **Pathway tab**: Should show enriched GO terms for differential proteins
2. **Sample correlation**: Should see Trial A/B/C clustering separately (batch effect)
3. **MA plot**: Should see symmetric distribution around y=0
4. **Heatmap**: Should see clustering of responders vs non-responders

---

## Week 3-4: Critical Research Features (40 hours)

### 5. Batch Effect Correction (20 hours)

**Install**:
```r
BiocManager::install("sva")  # For ComBat
```

**Key function**:
```r
library(sva)

apply_combat_correction <- function(expr_matrix, batch_var, covariates = NULL) {
  # expr_matrix: proteins × samples
  # batch_var: vector of batch labels (Trial_A, Trial_B, Trial_C)
  # covariates: data frame of variables to preserve (treatment, response)

  if (!is.null(covariates)) {
    mod <- model.matrix(~ ., data = covariates)
  } else {
    mod <- model.matrix(~ 1, data = data.frame(rep = 1:ncol(expr_matrix)))
  }

  ComBat(dat = expr_matrix,
         batch = batch_var,
         mod = mod,
         par.prior = TRUE)
}
```

**UI**: Add checkbox in Overview tab to toggle batch correction, show before/after PCA.

---

### 6. Advanced Missing Data Imputation (24 hours)

**Install**:
```r
install.packages("imp4p")
install.packages("missForest")
```

**Key functions**:
```r
library(imp4p)

# Hybrid imputation (detects MNAR vs MCAR automatically)
impute_hybrid <- function(expr_matrix) {
  imp4p::impute.wrapper.MLE.MLE(expr_matrix)
}

# QRILC for left-censored MNAR
impute_qrilc <- function(expr_matrix) {
  imp4p::impute.QRILC(expr_matrix)
}

# MissForest for MCAR
library(missForest)
impute_missforest <- function(expr_matrix) {
  result <- missForest(t(expr_matrix))
  t(result$ximp)
}
```

**UI**: Dropdown for imputation method in Overview tab, apply before analysis.

---

## Week 5-6: Multi-Group & Paired Analysis (36 hours)

### 7. Multi-Group ANOVA (16 hours)

**Use limma for vectorized ANOVA**:
```r
library(limma)

run_anova_limma <- function(expr_matrix, groups) {
  # Create design matrix
  design <- model.matrix(~ 0 + groups)
  colnames(design) <- levels(groups)

  # Fit model
  fit <- lmFit(expr_matrix, design)

  # Create contrasts for all pairwise comparisons
  contrasts <- combn(levels(groups), 2, function(x) {
    paste(x[1], "-", x[2])
  })

  contrast_matrix <- makeContrasts(contrasts = contrasts, levels = design)
  fit2 <- contrasts.fit(fit, contrast_matrix)
  fit2 <- eBayes(fit2)

  # Extract F-statistics
  topTable(fit2, number = Inf, sort.by = "F")
}
```

---

### 8. Paired Sample Analysis (18 hours)

**Update data generator to add subject IDs**:
```r
# In R/generate_data.R
metadata$subject_id <- paste0("Subject_", sprintf("%03d",
  rep(1:(n_total_samples %/% 2), each = 2)))
```

**Paired t-test**:
```r
run_paired_analysis <- function(data, subject_id, group_a, group_b) {
  paired_data <- data %>%
    filter(group %in% c(group_a, group_b)) %>%
    group_by(subject_id, protein_id) %>%
    filter(n() == 2) %>%  # Must have both timepoints
    ungroup()

  paired_data %>%
    group_by(subject_id, protein_id) %>%
    summarise(delta = log2_intensity[group == group_b] -
                      log2_intensity[group == group_a]) %>%
    group_by(protein_id) %>%
    summarise(
      mean_delta = mean(delta, na.rm = TRUE),
      p_value = t.test(delta)$p.value
    ) %>%
    mutate(adj_p_value = p.adjust(p_value, method = "BH"))
}
```

---

## Installation Checklist

Before starting, ensure all required packages are installed:

```r
# CRAN packages
install.packages(c(
  "shiny",
  "tidyverse",
  "DT",
  "broom",
  "pheatmap",
  "missForest"
))

# Bioconductor packages
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install(c(
  "clusterProfiler",
  "org.Hs.eg.db",
  "enrichplot",
  "limma",
  "sva"
))

# imp4p (proteomics-specific imputation)
install.packages("imp4p")
```

**System requirements**:
- R ≥ 4.3
- 4GB RAM minimum
- Internet connection (for first-time gene set downloads)

---

## Development Workflow

### 1. Create feature branch
```bash
git checkout -b feature/pathway-enrichment
```

### 2. Implement feature
- Write module in `R/mod_*.R`
- Add UI to `app.R`
- Add server logic to `app.R`

### 3. Test
- Generate fresh synthetic data: `Rscript R/generate_data.R`
- Run app: `R -e "shiny::runApp('.')"`
- Test all inputs and outputs
- Take screenshot for documentation

### 4. Document
- Update CLAUDE.md with new feature
- Add roxygen2 comments to functions
- Update README.md

### 5. Commit and merge
```bash
git add .
git commit -m "Add pathway enrichment module"
git push origin feature/pathway-enrichment
# Create PR, merge to main
```

---

## Troubleshooting

### clusterProfiler: "No gene can be mapped"
**Cause**: Gene symbols don't match database

**Fix**: Ensure protein IDs are official gene symbols (TP53, not tp53)

### ComBat: "Only two batches detected"
**Cause**: Batch variable has only 2 levels

**Fix**: ComBat works with 2+ batches, but check that batch_var is correct

### Heatmap: "NA/NaN/Inf in foreign function call"
**Cause**: Missing values in expression matrix

**Fix**: Apply imputation before creating heatmap

### Pathway enrichment: Very few enriched terms
**Cause**: Too few significant proteins, or wrong background

**Fix**: Relax p-value cutoff, or use GSEA instead of ORA

---

## Resources

### Documentation
- **clusterProfiler**: https://yulab-smu.top/biomedical-knowledge-mining-book/
- **limma User's Guide**: https://bioconductor.org/packages/release/bioc/vignettes/limma/inst/doc/usersguide.pdf
- **imp4p vignette**: https://cran.r-project.org/web/packages/imp4p/vignettes/imp4p.html

### Example Datasets
- **ProteomeXchange**: http://proteomecentral.proteomexchange.org/
- **PRIDE Archive**: https://www.ebi.ac.uk/pride/

### Tutorials
- **Proteomics data analysis in R**: https://lgatto.github.io/ProteomicsAnnotationHubData/
- **Batch effect correction**: https://pmc.ncbi.nlm.nih.gov/articles/PMC8447595/

---

## Success Checklist

After completing Quick Wins + Critical Features, you should have:

- [ ] Pathway enrichment working with GO/KEGG
- [ ] Sample correlation heatmap identifying batch effects
- [ ] MA plot as alternative to volcano
- [ ] Clustered heatmap of top proteins
- [ ] ComBat batch correction with before/after PCA
- [ ] 3-4 imputation methods available
- [ ] Multi-group ANOVA for 3+ groups
- [ ] Paired analysis for longitudinal studies

**Total time**: ~6 weeks (114 hours)
**Result**: Publication-quality research tool

---

*Quick Start Guide Version 1.0*
*Last updated: 2026-01-11*
