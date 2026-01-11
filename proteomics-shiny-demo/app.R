# ==============================================================================
# Shiny Proteomics Explorer
# Cross-study proteomics visualization, QC, and differential analysis
# ==============================================================================

library(shiny)
library(tidyverse)
library(DT)
library(broom)
library(pheatmap)  # Feature 1: Sample Correlation Heatmap
library(matrixStats)  # Feature 12: RLE plot (rowMedians function)

# --- Load Data ---
# Check if RDS exists, otherwise generate it
if (!file.exists("data/proteomics_data.rds")) {
  source("R/generate_data.R")
}

data <- readRDS("data/proteomics_data.rds")
expr_long <- data$expression_long
expr_wide <- data$expression_wide
metadata <- data$metadata
protein_list <- data$protein_ids

# ==============================================================================
# UI
# ==============================================================================
ui <- navbarPage(

  title = "Proteomics Explorer",
  theme = NULL,
  
  # --- Tab 1: Overview ---
  tabPanel(
    "Overview",
    icon = icon("chart-bar"),

    # Dataset Summary (always visible)
    fluidRow(
      column(12,
        wellPanel(
          h3(icon("database"), " Dataset Summary"),
          fluidRow(
            column(3,
              div(class = "text-center",
                h2(textOutput("n_samples"), class = "text-primary"),
                p("Samples")
              )
            ),
            column(3,
              div(class = "text-center",
                h2(textOutput("n_proteins"), class = "text-primary"),
                p("Proteins")
              )
            ),
            column(3,
              div(class = "text-center",
                h2(textOutput("n_trials"), class = "text-primary"),
                p("Studies")
              )
            ),
            column(3,
              h5("Breakdown by Trial:"),
              tableOutput("trial_summary")
            )
          )
        )
      )
    ),

    # Organized QC sections with tabsetPanel
    tabsetPanel(
      type = "tabs",

      # Tab: Missingness & Quality
      tabPanel(
        "Missingness & Quality",
        icon = icon("chart-area"),
        br(),
        fluidRow(
          column(6,
            wellPanel(
              h4(icon("chart-area"), " Missingness Distribution"),
              selectInput("miss_view", "View by:",
                          choices = c("Per Sample" = "sample", "Per Protein" = "protein")),
              plotOutput("missingness_plot", height = "300px")
            )
          ),
          column(6,
            wellPanel(
              h4(icon("th"), " Missingness Pattern Heatmap"),
              p("Systematic missingness patterns (MNAR vs MCAR)", class = "text-muted", style = "font-size: 12px;"),
              plotOutput("missingness_heatmap", height = "300px"),
              downloadButton("download_miss_heatmap", "Download PNG", class = "btn-sm")
            )
          )
        ),
        fluidRow(
          column(6,
            wellPanel(
              h4(icon("chart-line"), " Relative Log Expression (RLE)"),
              p("Boxplots should center at zero for normalized data", class = "text-muted", style = "font-size: 12px;"),
              plotOutput("rle_plot", height = "300px"),
              downloadButton("download_rle", "Download PNG", class = "btn-sm")
            )
          ),
          column(6,
            wellPanel(
              h4(icon("percent"), " Coefficient of Variation (CV)"),
              p("Distribution of protein variability across samples", class = "text-muted", style = "font-size: 12px;"),
              plotOutput("cv_plot", height = "300px"),
              downloadButton("download_cv", "Download PNG", class = "btn-sm")
            )
          )
        )
      ),

      # Tab: PCA & Clustering
      tabPanel(
        "PCA & Clustering",
        icon = icon("project-diagram"),
        br(),
        fluidRow(
          column(3,
            wellPanel(
              h4(icon("cog"), " PCA Settings"),
              selectInput("pca_color", "Color by:",
                          choices = c("Trial" = "trial",
                                      "Treatment" = "treatment",
                                      "Response" = "response",
                                      "Timepoint" = "timepoint")),
              selectInput("pca_shape", "Shape by:",
                          choices = c("None" = "none",
                                      "Trial" = "trial",
                                      "Treatment" = "treatment",
                                      "Response" = "response"))
            )
          ),
          column(9,
            wellPanel(
              h4(icon("compress-arrows-alt"), " Principal Component Analysis"),
              plotOutput("pca_plot", height = "400px")
            )
          )
        ),

        # Sample Correlation Heatmap (Feature 1)
        fluidRow(
          column(12,
            wellPanel(
              h4(icon("th"), " Sample Correlation Heatmap"),
              p("Pearson correlation matrix showing similarity between samples. Hierarchical clustering reveals batch effects and outliers."),
              fluidRow(
                column(9,
                  plotOutput("correlation_heatmap", height = "500px")
                ),
                column(3,
                  h5(icon("cog"), " Display Options"),
                  selectInput("corr_annotation", "Color annotations:",
                              choices = c("Trial" = "trial",
                                          "Treatment" = "treatment",
                                          "Response" = "response",
                                          "Timepoint" = "timepoint",
                                          "None" = "none"),
                              selected = "trial"),
                  hr(),
                  downloadButton("download_corr_plot", "Download PNG", class = "btn-primary btn-sm"),
                  br(), br(),
                  downloadButton("download_corr_pdf", "Download PDF", class = "btn-secondary btn-sm")
                )
              )
            )
          )
        )
      )
    )
  ),

  # --- Tab 2: Protein Explorer ---
  tabPanel(
    "Protein Explorer",
    icon = icon("search"),
    
    sidebarLayout(
      sidebarPanel(
        width = 3,
        h4(icon("dna"), " Select Protein"),
        selectizeInput("protein_select", "Protein ID:",
                       choices = NULL,  # Server-side update
                       options = list(maxOptions = 100)),
        hr(),
        h4(icon("layer-group"), " Grouping"),
        selectInput("group_by", "Compare by:",
                    choices = c("Treatment" = "treatment",
                                "Response" = "response",
                                "Trial" = "trial",
                                "Timepoint" = "timepoint")),
        hr(),
        h4(icon("calculator"), " Statistics"),
        selectInput("stat_test", "Test:",
                    choices = c("t-test" = "t.test",
                                "Wilcoxon" = "wilcox")),
        conditionalPanel(
          condition = "input.group_by != 'timepoint'",
          selectInput("group_a", "Group A:", choices = NULL),
          selectInput("group_b", "Group B:", choices = NULL)
        )
      ),
      
      mainPanel(
        width = 9,
        fluidRow(
          column(8,
            wellPanel(
              h4(textOutput("protein_title")),
              plotOutput("protein_boxplot", height = "400px")
            )
          ),
          column(4,
            wellPanel(
              h4(icon("chart-line"), " Statistics Summary"),
              verbatimTextOutput("stat_results"),
              hr(),
              h5("Sample Sizes:"),
              tableOutput("sample_sizes")
            )
          )
        ),
        
        fluidRow(
          column(12,
            wellPanel(
              h4(icon("chart-bar"), " Expression Distribution"),
              plotOutput("protein_density", height = "250px")
            )
          )
        )
      )
    )
  ),
  
  # --- Tab 3: Differential Analysis ---
  tabPanel(
    "Differential",
    icon = icon("balance-scale"),
    
    sidebarLayout(
      sidebarPanel(
        width = 3,
        h4(icon("exchange-alt"), " Group Comparison"),
        selectInput("diff_variable", "Compare by:",
                    choices = c("Treatment" = "treatment",
                                "Response" = "response",
                                "Trial" = "trial")),
        selectInput("diff_group_a", "Group A:", choices = NULL),
        selectInput("diff_group_b", "Group B:", choices = NULL),
        hr(),
        h4(icon("sliders-h"), " Thresholds"),
        sliderInput("fc_thresh", "Log2 FC threshold:",
                    min = 0, max = 3, value = 1, step = 0.25),
        sliderInput("pval_thresh", "P-value threshold:",
                    min = 0, max = 0.1, value = 0.05, step = 0.01),
        hr(),
        actionButton("run_diff", "Run Analysis", 
                     class = "btn-primary btn-block", icon = icon("play")),
        hr(),
        downloadButton("download_results", "Download CSV", class = "btn-success btn-block")
      ),
      
      mainPanel(
        width = 9,
        fluidRow(
          column(7,
            wellPanel(
              h4(icon("mountain"), " Volcano Plot"),
              plotOutput("volcano_plot", height = "450px", brush = "volcano_brush")
            )
          ),
          column(5,
            wellPanel(
              h4(icon("chart-pie"), " Results Summary"),
              fluidRow(
                column(4, 
                  div(class = "text-center",
                    h3(textOutput("n_up"), style = "color: #e74c3c;"),
                    p("Up-regulated")
                  )
                ),
                column(4,
                  div(class = "text-center",
                    h3(textOutput("n_down"), style = "color: #3498db;"),
                    p("Down-regulated")
                  )
                ),
                column(4,
                  div(class = "text-center",
                    h3(textOutput("n_ns")),
                    p("Not Significant")
                  )
                )
              ),
              hr(),
              h5("Top Differential Proteins:"),
              tableOutput("top_proteins")
            )
          )
        ),

        # --- Feature 2: MA Plot ---
        fluidRow(
          column(7,
            wellPanel(
              h4(icon("chart-line"), " MA Plot"),
              p("M-A plot showing relationship between fold change and average expression",
                class = "text-muted", style = "font-size: 12px;"),
              plotOutput("ma_plot", height = "450px")
            )
          ),
          column(5,
            wellPanel(
              h4(icon("download"), " Export Plots"),
              downloadButton("download_volcano", "Download Volcano (PNG)",
                             class = "btn-primary btn-sm btn-block"),
              br(), br(),
              downloadButton("download_ma", "Download MA Plot (PNG)",
                             class = "btn-primary btn-sm btn-block"),
              hr(),
              p("High-resolution 300 DPI PNG files suitable for publication",
                class = "text-muted", style = "font-size: 11px;")
            )
          )
        ),

        fluidRow(
          column(12,
            wellPanel(
              h4(icon("table"), " Full Results Table"),
              DTOutput("diff_table")
            )
          )
        )
      )
    )
  ),

  # ========================================
  # TAB 4: Clustered Heatmap
  # ========================================
  tabPanel(
    title = "Heatmap",
    icon = icon("th"),

    fluidRow(
      column(12,
        wellPanel(
          h3(icon("fire"), " Clustered Heatmap of Top Differential Proteins"),
          p("Visualize expression patterns of the most significant proteins across all samples.",
            "Z-scores show how many standard deviations each value is from the protein's mean."),

          fluidRow(
            column(3,
              h4(icon("sliders"), " Selection Criteria"),
              helpText("Must run Differential Analysis first (previous tab)"),

              numericInput("heatmap_top_n",
                           "Number of top proteins:",
                           value = 50, min = 10, max = 200, step = 10),

              selectInput("heatmap_order_by",
                          "Rank proteins by:",
                          choices = c("P-value" = "pvalue",
                                      "Adjusted P-value" = "padj",
                                      "Absolute fold change" = "abs_fc")),

              checkboxInput("heatmap_sig_only",
                            "Only significant proteins",
                            value = TRUE),

              hr(),

              selectInput("heatmap_annotation",
                          "Sample annotations:",
                          choices = c("Treatment" = "treatment",
                                      "Response" = "response",
                                      "Timepoint" = "timepoint",
                                      "Trial" = "trial",
                                      "None" = "none"),
                          selected = "treatment"),

              checkboxInput("heatmap_cluster_rows",
                            "Cluster proteins (rows)",
                            value = TRUE),

              checkboxInput("heatmap_cluster_cols",
                            "Cluster samples (columns)",
                            value = TRUE),

              hr(),

              actionButton("render_heatmap",
                           "Generate Heatmap",
                           icon = icon("play"),
                           class = "btn-primary btn-block"),

              br(),

              downloadButton("download_heatmap_png",
                             "Download PNG",
                             class = "btn-sm btn-block"),
              downloadButton("download_heatmap_pdf",
                             "Download PDF",
                             class = "btn-sm btn-block")
            ),

            column(9,
              wellPanel(
                plotOutput("clustered_heatmap", height = "700px")
              ),

              wellPanel(
                h5(icon("info-circle"), " Interpretation Guide"),
                HTML("<ul>
                  <li><b>Red</b>: Expression above protein mean (Z-score > 0)</li>
                  <li><b>Blue</b>: Expression below protein mean (Z-score < 0)</li>
                  <li><b>Dendrograms</b>: Show similarity relationships (shorter branches = more similar)</li>
                  <li><b>Sample clusters</b>: Samples with similar expression profiles group together</li>
                  <li><b>Protein clusters</b>: Co-regulated proteins group together</li>
                </ul>")
              )
            )
          )
        )
      )
    )
  )
)

# ==============================================================================
# Server
# ==============================================================================
server <- function(input, output, session) {
  
  # --- Overview Tab ---
  output$n_samples <- renderText({ nrow(metadata) })
  output$n_proteins <- renderText({ length(protein_list) })
  output$n_trials <- renderText({ n_distinct(metadata$trial) })
  
  output$trial_summary <- renderTable({
    metadata %>%
      group_by(Trial = trial) %>%
      summarise(
        Samples = n(),
        `Drug A` = sum(treatment == "Drug_A"),
        `Drug B` = sum(treatment == "Drug_B"),
        Placebo = sum(treatment == "Placebo"),
        .groups = "drop"
      )
  })
  
  # Missingness Plot
  output$missingness_plot <- renderPlot({
    if (input$miss_view == "sample") {
      miss_data <- expr_long %>%
        group_by(sample_id) %>%
        summarise(missing_pct = mean(is.na(log2_intensity)) * 100, .groups = "drop")
      
      ggplot(miss_data, aes(x = missing_pct)) +
        geom_histogram(bins = 30, fill = "#3498db", color = "white", alpha = 0.8) +
        labs(x = "% Missing Values", y = "Number of Samples",
             title = "Missingness Distribution (per Sample)") +
        theme_minimal(base_size = 12) +
        theme(plot.title = element_text(face = "bold"))
      
    } else {
      miss_data <- expr_long %>%
        group_by(protein_id) %>%
        summarise(missing_pct = mean(is.na(log2_intensity)) * 100, .groups = "drop")
      
      ggplot(miss_data, aes(x = missing_pct)) +
        geom_histogram(bins = 30, fill = "#9b59b6", color = "white", alpha = 0.8) +
        labs(x = "% Missing Values", y = "Number of Proteins",
             title = "Missingness Distribution (per Protein)") +
        theme_minimal(base_size = 12) +
        theme(plot.title = element_text(face = "bold"))
    }
  })
  
  # PCA Plot
  pca_data <- reactive({
    # Wide matrix for PCA
    mat <- expr_wide %>%
      column_to_rownames("protein_id") %>%
      as.matrix()

    # Remove proteins with >50% missing
    keep_proteins <- rowMeans(is.na(mat)) < 0.5
    mat <- mat[keep_proteins, ]

    # Impute remaining NAs with row median
    for (i in seq_len(nrow(mat))) {
      row_median <- median(mat[i, ], na.rm = TRUE)
      # Only impute if median is not NA (i.e., row has some non-NA values)
      if (!is.na(row_median) && any(is.na(mat[i, ]))) {
        mat[i, is.na(mat[i, ])] <- row_median
      }
    }

    # Remove any remaining rows that are all NA
    mat <- mat[rowSums(!is.na(mat)) > 0, ]

    # PCA
    pca <- prcomp(t(mat), scale. = TRUE)

    pca_df <- as.data.frame(pca$x[, 1:5]) %>%
      rownames_to_column("sample_id") %>%
      left_join(metadata, by = "sample_id")

    var_explained <- round(summary(pca)$importance[2, 1:5] * 100, 1)

    list(pca_df = pca_df, var_explained = var_explained)
  })
  
  output$pca_plot <- renderPlot({
    pca <- pca_data()
    df <- pca$pca_df
    var_exp <- pca$var_explained
    
    p <- ggplot(df, aes(x = PC1, y = PC2, color = .data[[input$pca_color]]))
    
    if (input$pca_shape != "none") {
      p <- p + geom_point(aes(shape = .data[[input$pca_shape]]), size = 3, alpha = 0.8)
    } else {
      p <- p + geom_point(size = 3, alpha = 0.8)
    }
    
    # Build plot with proper legend handling
    p <- p +
      stat_ellipse(level = 0.68, linetype = "dashed") +
      labs(x = paste0("PC1 (", var_exp[1], "%)"),
           y = paste0("PC2 (", var_exp[2], "%)"),
           title = "PCA of Protein Expression",
           color = str_to_title(input$pca_color)) +
      theme_minimal(base_size = 14) +
      theme(
        plot.title = element_text(face = "bold", size = 16),
        legend.position = "right"
      ) +
      scale_color_brewer(palette = "Set2")

    # Only add shape label if shape is used
    if (input$pca_shape != "none") {
      p <- p + labs(shape = str_to_title(input$pca_shape))
    }

    p
  })

  # --- Sample Correlation Heatmap (Feature 1) ---

  # Reactive: Compute correlation matrix and prepare data
  corr_data <- reactive({
    # Use the same processed matrix as PCA (proteins × samples)
    mat <- expr_wide %>%
      column_to_rownames("protein_id") %>%
      as.matrix()

    # Remove proteins with >50% missing
    keep_proteins <- rowMeans(is.na(mat)) < 0.5
    mat <- mat[keep_proteins, ]

    # Impute remaining NAs with row median
    for (i in seq_len(nrow(mat))) {
      row_median <- median(mat[i, ], na.rm = TRUE)
      # Only impute if median is not NA (i.e., row has some non-NA values)
      if (!is.na(row_median) && any(is.na(mat[i, ]))) {
        mat[i, is.na(mat[i, ])] <- row_median
      }
    }

    # Remove any remaining rows that are all NA
    mat <- mat[rowSums(!is.na(mat)) > 0, ]

    # Compute Pearson correlation between samples (columns)
    cor_mat <- cor(mat, use = "pairwise.complete.obs")

    list(cor_mat = cor_mat)
  })

  # Render correlation heatmap
  output$correlation_heatmap <- renderPlot({
    cor_mat <- corr_data()$cor_mat

    # Prepare annotation data frame
    if (input$corr_annotation != "none") {
      annotation_col <- metadata %>%
        select(sample_id, annotation = !!sym(input$corr_annotation)) %>%
        column_to_rownames("sample_id")

      # Define color palette for annotation
      unique_vals <- unique(annotation_col$annotation)
      if (length(unique_vals) <= 9) {
        ann_colors <- list(
          annotation = setNames(RColorBrewer::brewer.pal(max(3, length(unique_vals)), "Set2")[1:length(unique_vals)],
                                unique_vals)
        )
      } else {
        # For >9 categories, use a larger palette
        ann_colors <- list(
          annotation = setNames(rainbow(length(unique_vals)), unique_vals)
        )
      }

      pheatmap(
        cor_mat,
        annotation_col = annotation_col,
        annotation_colors = ann_colors,
        color = colorRampPalette(c("blue", "white", "red"))(100),
        breaks = seq(0.6, 1.0, length.out = 101),
        clustering_distance_rows = "correlation",
        clustering_distance_cols = "correlation",
        clustering_method = "ward.D2",
        show_rownames = FALSE,
        show_colnames = FALSE,
        fontsize = 10,
        main = "Sample Correlation Heatmap (Pearson)",
        legend = TRUE,
        border_color = NA
      )
    } else {
      # No annotation
      pheatmap(
        cor_mat,
        color = colorRampPalette(c("blue", "white", "red"))(100),
        breaks = seq(0.6, 1.0, length.out = 101),
        clustering_distance_rows = "correlation",
        clustering_distance_cols = "correlation",
        clustering_method = "ward.D2",
        show_rownames = FALSE,
        show_colnames = FALSE,
        fontsize = 10,
        main = "Sample Correlation Heatmap (Pearson)",
        legend = TRUE,
        border_color = NA
      )
    }
  })

  # Download handlers for correlation heatmap
  output$download_corr_plot <- downloadHandler(
    filename = function() {
      paste0("sample_correlation_", Sys.Date(), ".png")
    },
    content = function(file) {
      png(file, width = 10, height = 8, units = "in", res = 300)

      cor_mat <- corr_data()$cor_mat

      if (input$corr_annotation != "none") {
        annotation_col <- metadata %>%
          select(sample_id, annotation = !!sym(input$corr_annotation)) %>%
          column_to_rownames("sample_id")

        unique_vals <- unique(annotation_col$annotation)
        if (length(unique_vals) <= 9) {
          ann_colors <- list(
            annotation = setNames(RColorBrewer::brewer.pal(max(3, length(unique_vals)), "Set2")[1:length(unique_vals)],
                                  unique_vals)
          )
        } else {
          ann_colors <- list(
            annotation = setNames(rainbow(length(unique_vals)), unique_vals)
          )
        }

        pheatmap(cor_mat, annotation_col = annotation_col, annotation_colors = ann_colors,
                 color = colorRampPalette(c("blue", "white", "red"))(100),
                 breaks = seq(0.6, 1.0, length.out = 101),
                 clustering_distance_rows = "correlation",
                 clustering_distance_cols = "correlation",
                 clustering_method = "ward.D2",
                 show_rownames = FALSE, show_colnames = FALSE,
                 main = "Sample Correlation Heatmap (Pearson)")
      } else {
        pheatmap(cor_mat,
                 color = colorRampPalette(c("blue", "white", "red"))(100),
                 breaks = seq(0.6, 1.0, length.out = 101),
                 clustering_distance_rows = "correlation",
                 clustering_distance_cols = "correlation",
                 clustering_method = "ward.D2",
                 show_rownames = FALSE, show_colnames = FALSE,
                 main = "Sample Correlation Heatmap (Pearson)")
      }

      dev.off()
    }
  )

  output$download_corr_pdf <- downloadHandler(
    filename = function() {
      paste0("sample_correlation_", Sys.Date(), ".pdf")
    },
    content = function(file) {
      pdf(file, width = 10, height = 8)

      cor_mat <- corr_data()$cor_mat

      if (input$corr_annotation != "none") {
        annotation_col <- metadata %>%
          select(sample_id, annotation = !!sym(input$corr_annotation)) %>%
          column_to_rownames("sample_id")

        unique_vals <- unique(annotation_col$annotation)
        if (length(unique_vals) <= 9) {
          ann_colors <- list(
            annotation = setNames(RColorBrewer::brewer.pal(max(3, length(unique_vals)), "Set2")[1:length(unique_vals)],
                                  unique_vals)
          )
        } else {
          ann_colors <- list(
            annotation = setNames(rainbow(length(unique_vals)), unique_vals)
          )
        }

        pheatmap(cor_mat, annotation_col = annotation_col, annotation_colors = ann_colors,
                 color = colorRampPalette(c("blue", "white", "red"))(100),
                 breaks = seq(0.6, 1.0, length.out = 101),
                 clustering_distance_rows = "correlation",
                 clustering_distance_cols = "correlation",
                 clustering_method = "ward.D2",
                 show_rownames = FALSE, show_colnames = FALSE,
                 main = "Sample Correlation Heatmap (Pearson)")
      } else {
        pheatmap(cor_mat,
                 color = colorRampPalette(c("blue", "white", "red"))(100),
                 breaks = seq(0.6, 1.0, length.out = 101),
                 clustering_distance_rows = "correlation",
                 clustering_distance_cols = "correlation",
                 clustering_method = "ward.D2",
                 show_rownames = FALSE, show_colnames = FALSE,
                 main = "Sample Correlation Heatmap (Pearson)")
      }

      dev.off()
    }
  )

  # ========================================
  # Feature 12: Additional QC Plots
  # ========================================

  # 1. RLE (Relative Log Expression) Plot
  output$rle_plot <- renderPlot({
    # Get expression matrix
    mat <- expr_wide %>%
      column_to_rownames("protein_id") %>%
      as.matrix()

    # Calculate median protein expression across all samples
    protein_medians <- rowMedians(mat, na.rm = TRUE)

    # Subtract median from each protein (center at zero)
    rle_mat <- mat - protein_medians

    # Reshape to long format for ggplot
    rle_data <- rle_mat %>%
      as.data.frame() %>%
      rownames_to_column("protein_id") %>%
      pivot_longer(-protein_id, names_to = "sample_id", values_to = "rle") %>%
      filter(!is.na(rle))

    # Create boxplot
    ggplot(rle_data, aes(x = sample_id, y = rle)) +
      geom_boxplot(fill = "#3498db", alpha = 0.7, outlier.size = 0.5) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth = 1) +
      labs(x = "Sample", y = "Relative Log Expression",
           title = "RLE Plot: Data Quality Assessment",
           subtitle = "Boxplots should center at zero for well-normalized data") +
      theme_minimal(base_size = 12) +
      theme(axis.text.x = element_blank(),
            axis.ticks.x = element_blank(),
            plot.title = element_text(face = "bold"))
  })

  # Download RLE plot
  output$download_rle <- downloadHandler(
    filename = function() { paste0("rle_plot_", Sys.Date(), ".png") },
    content = function(file) {
      png(file, width = 10, height = 6, units = "in", res = 300)

      mat <- expr_wide %>%
        column_to_rownames("protein_id") %>%
        as.matrix()
      protein_medians <- rowMedians(mat, na.rm = TRUE)
      rle_mat <- mat - protein_medians
      rle_data <- rle_mat %>%
        as.data.frame() %>%
        rownames_to_column("protein_id") %>%
        pivot_longer(-protein_id, names_to = "sample_id", values_to = "rle") %>%
        filter(!is.na(rle))

      print(ggplot(rle_data, aes(x = sample_id, y = rle)) +
        geom_boxplot(fill = "#3498db", alpha = 0.7, outlier.size = 0.5) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth = 1) +
        labs(x = "Sample", y = "Relative Log Expression",
             title = "RLE Plot: Data Quality Assessment") +
        theme_minimal(base_size = 14) +
        theme(axis.text.x = element_blank(),
              axis.ticks.x = element_blank(),
              plot.title = element_text(face = "bold")))

      dev.off()
    }
  )

  # 2. CV (Coefficient of Variation) Plot
  output$cv_plot <- renderPlot({
    # Calculate CV per protein (sd/mean * 100)
    mat <- expr_wide %>%
      column_to_rownames("protein_id") %>%
      as.matrix()

    cv_values <- apply(mat, 1, function(x) {
      x_clean <- x[!is.na(x)]
      if (length(x_clean) > 1) {
        (sd(x_clean) / mean(x_clean)) * 100
      } else {
        NA
      }
    })

    cv_data <- data.frame(
      protein_id = names(cv_values),
      cv = cv_values
    ) %>%
      filter(!is.na(cv))

    # Create histogram
    ggplot(cv_data, aes(x = cv)) +
      geom_histogram(bins = 50, fill = "#2ecc71", color = "white", alpha = 0.8) +
      geom_vline(xintercept = median(cv_data$cv), linetype = "dashed",
                 color = "red", linewidth = 1) +
      geom_vline(xintercept = c(10, 30), linetype = "dotted",
                 color = "gray50", linewidth = 0.7) +
      annotate("text", x = median(cv_data$cv), y = Inf,
               label = paste("Median:", round(median(cv_data$cv), 1), "%"),
               vjust = 1.5, hjust = -0.1, color = "red") +
      labs(x = "Coefficient of Variation (%)", y = "Number of Proteins",
           title = "CV Distribution Across Proteins",
           subtitle = "Typical range: 10-30% for biological data") +
      theme_minimal(base_size = 12) +
      theme(plot.title = element_text(face = "bold"))
  })

  # Download CV plot
  output$download_cv <- downloadHandler(
    filename = function() { paste0("cv_plot_", Sys.Date(), ".png") },
    content = function(file) {
      png(file, width = 8, height = 6, units = "in", res = 300)

      mat <- expr_wide %>%
        column_to_rownames("protein_id") %>%
        as.matrix()
      cv_values <- apply(mat, 1, function(x) {
        x_clean <- x[!is.na(x)]
        if (length(x_clean) > 1) (sd(x_clean) / mean(x_clean)) * 100 else NA
      })
      cv_data <- data.frame(protein_id = names(cv_values), cv = cv_values) %>%
        filter(!is.na(cv))

      print(ggplot(cv_data, aes(x = cv)) +
        geom_histogram(bins = 50, fill = "#2ecc71", color = "white", alpha = 0.8) +
        geom_vline(xintercept = median(cv_data$cv), linetype = "dashed",
                   color = "red", linewidth = 1) +
        geom_vline(xintercept = c(10, 30), linetype = "dotted", color = "gray50") +
        annotate("text", x = median(cv_data$cv), y = Inf,
                 label = paste("Median:", round(median(cv_data$cv), 1), "%"),
                 vjust = 1.5, hjust = -0.1, color = "red") +
        labs(x = "Coefficient of Variation (%)", y = "Number of Proteins",
             title = "CV Distribution Across Proteins") +
        theme_minimal(base_size = 14) +
        theme(plot.title = element_text(face = "bold")))

      dev.off()
    }
  )

  # 3. Missingness Pattern Heatmap
  output$missingness_heatmap <- renderPlot({
    # Create binary missingness matrix
    mat <- expr_wide %>%
      column_to_rownames("protein_id") %>%
      as.matrix()

    # Convert to binary: 1 = missing, 0 = present
    miss_mat <- is.na(mat) * 1

    # Filter to proteins with >5% missingness for better visualization
    protein_miss_pct <- rowMeans(miss_mat)
    miss_mat_filtered <- miss_mat[protein_miss_pct > 0.05, ]

    # If too many proteins, sample to 100
    if (nrow(miss_mat_filtered) > 100) {
      set.seed(42)
      sample_idx <- sample(1:nrow(miss_mat_filtered), 100)
      miss_mat_filtered <- miss_mat_filtered[sample_idx, ]
    }

    # Create heatmap
    pheatmap(
      miss_mat_filtered,
      color = c("gray90", "#e74c3c"),
      breaks = c(-0.1, 0.5, 1.1),
      cluster_rows = TRUE,
      cluster_cols = TRUE,
      clustering_distance_rows = "binary",
      clustering_distance_cols = "binary",
      clustering_method = "ward.D2",
      show_rownames = FALSE,
      show_colnames = FALSE,
      main = "Missingness Pattern (Gray=Present, Red=Missing)",
      legend = FALSE,
      fontsize = 10
    )
  })

  # Download missingness heatmap
  output$download_miss_heatmap <- downloadHandler(
    filename = function() { paste0("missingness_heatmap_", Sys.Date(), ".png") },
    content = function(file) {
      png(file, width = 10, height = 8, units = "in", res = 300)

      mat <- expr_wide %>%
        column_to_rownames("protein_id") %>%
        as.matrix()
      miss_mat <- is.na(mat) * 1
      protein_miss_pct <- rowMeans(miss_mat)
      miss_mat_filtered <- miss_mat[protein_miss_pct > 0.05, ]

      if (nrow(miss_mat_filtered) > 100) {
        set.seed(42)
        sample_idx <- sample(1:nrow(miss_mat_filtered), 100)
        miss_mat_filtered <- miss_mat_filtered[sample_idx, ]
      }

      pheatmap(
        miss_mat_filtered,
        color = c("gray90", "#e74c3c"),
        breaks = c(-0.1, 0.5, 1.1),
        cluster_rows = TRUE,
        cluster_cols = TRUE,
        clustering_distance_rows = "binary",
        clustering_distance_cols = "binary",
        clustering_method = "ward.D2",
        show_rownames = FALSE,
        show_colnames = FALSE,
        main = "Missingness Pattern (Gray=Present, Red=Missing)",
        legend = FALSE
      )

      dev.off()
    }
  )

  # --- Protein Explorer Tab ---
  # Update protein choices
  updateSelectizeInput(session, "protein_select", 
                       choices = protein_list, 
                       selected = protein_list[1],
                       server = TRUE)
  
  # Update group choices based on group_by selection
  observe({
    groups <- unique(metadata[[input$group_by]])
    updateSelectInput(session, "group_a", choices = groups, selected = groups[1])
    updateSelectInput(session, "group_b", choices = groups, selected = groups[2])
  })
  
  protein_data <- reactive({
    req(input$protein_select)
    expr_long %>%
      filter(protein_id == input$protein_select) %>%
      filter(!is.na(log2_intensity))
  })
  
  output$protein_title <- renderText({
    paste("Expression of", input$protein_select)
  })
  
  output$protein_boxplot <- renderPlot({
    req(nrow(protein_data()) > 0)
    
    ggplot(protein_data(), aes(x = .data[[input$group_by]], y = log2_intensity, 
                               fill = .data[[input$group_by]])) +
      geom_violin(alpha = 0.6, trim = FALSE) +
      geom_boxplot(width = 0.2, outlier.shape = NA, alpha = 0.8) +
      geom_jitter(width = 0.1, alpha = 0.5, size = 1.5) +
      labs(x = str_to_title(input$group_by), 
           y = "Log2 Intensity",
           title = paste(input$protein_select, "Expression by", str_to_title(input$group_by))) +
      theme_minimal(base_size = 14) +
      theme(
        plot.title = element_text(face = "bold"),
        legend.position = "none"
      ) +
      scale_fill_brewer(palette = "Set2")
  })
  
  output$protein_density <- renderPlot({
    req(nrow(protein_data()) > 0)
    
    ggplot(protein_data(), aes(x = log2_intensity, fill = .data[[input$group_by]])) +
      geom_density(alpha = 0.5) +
      labs(x = "Log2 Intensity", y = "Density",
           fill = str_to_title(input$group_by)) +
      theme_minimal(base_size = 12) +
      scale_fill_brewer(palette = "Set2")
  })
  
  stat_result <- reactive({
    req(input$protein_select, input$group_a, input$group_b)
    
    df <- protein_data() %>%
      filter(.data[[input$group_by]] %in% c(input$group_a, input$group_b))
    
    if (nrow(df) < 4) return(NULL)
    
    group_a_vals <- df %>% filter(.data[[input$group_by]] == input$group_a) %>% pull(log2_intensity)
    group_b_vals <- df %>% filter(.data[[input$group_by]] == input$group_b) %>% pull(log2_intensity)
    
    if (length(group_a_vals) < 2 | length(group_b_vals) < 2) return(NULL)
    
    if (input$stat_test == "t.test") {
      test <- t.test(group_a_vals, group_b_vals)
    } else {
      test <- wilcox.test(group_a_vals, group_b_vals)
    }
    
    list(
      test = test,
      mean_a = mean(group_a_vals, na.rm = TRUE),
      mean_b = mean(group_b_vals, na.rm = TRUE),
      log2fc = mean(group_a_vals, na.rm = TRUE) - mean(group_b_vals, na.rm = TRUE),
      n_a = length(group_a_vals),
      n_b = length(group_b_vals)
    )
  })
  
  output$stat_results <- renderPrint({
    res <- stat_result()
    if (is.null(res)) {
      cat("Insufficient data for statistical test")
      return()
    }
    
    cat("Test:", ifelse(input$stat_test == "t.test", "Welch's t-test", "Wilcoxon rank-sum"), "\n")
    cat("─────────────────────────\n")
    cat(sprintf("Mean %s: %.3f\n", input$group_a, res$mean_a))
    cat(sprintf("Mean %s: %.3f\n", input$group_b, res$mean_b))
    cat(sprintf("Log2 FC: %.3f\n", res$log2fc))
    cat("─────────────────────────\n")
    cat(sprintf("P-value: %.2e\n", res$test$p.value))
    cat(sprintf("Significant: %s\n", ifelse(res$test$p.value < 0.05, "Yes*", "No")))
  })
  
  output$sample_sizes <- renderTable({
    res <- stat_result()
    if (is.null(res)) return(NULL)
    
    tibble(
      Group = c(input$group_a, input$group_b),
      N = c(res$n_a, res$n_b)
    )
  })
  
  # --- Differential Analysis Tab ---
  observe({
    groups <- unique(metadata[[input$diff_variable]])
    updateSelectInput(session, "diff_group_a", choices = groups, selected = groups[1])
    updateSelectInput(session, "diff_group_b", choices = groups, selected = groups[2])
  })
  
  diff_results <- eventReactive(input$run_diff, {
    req(input$diff_group_a, input$diff_group_b)
    
    withProgress(message = 'Computing differential expression...', value = 0, {
      
      # Get samples for each group
      samples_a <- metadata %>%
        filter(.data[[input$diff_variable]] == input$diff_group_a) %>%
        pull(sample_id)
      
      samples_b <- metadata %>%
        filter(.data[[input$diff_variable]] == input$diff_group_b) %>%
        pull(sample_id)
      
      results <- tibble()
      
      for (i in seq_along(protein_list)) {
        if (i %% 50 == 0) {
          incProgress(50/length(protein_list))
        }
        
        protein <- protein_list[i]
        
        vals_a <- expr_long %>%
          filter(protein_id == protein, sample_id %in% samples_a) %>%
          pull(log2_intensity) %>%
          na.omit()
        
        vals_b <- expr_long %>%
          filter(protein_id == protein, sample_id %in% samples_b) %>%
          pull(log2_intensity) %>%
          na.omit()
        
        if (length(vals_a) < 3 | length(vals_b) < 3) next
        
        mean_a <- mean(vals_a)
        mean_b <- mean(vals_b)
        log2fc <- mean_a - mean_b
        
        pval <- tryCatch({
          t.test(vals_a, vals_b)$p.value
        }, error = function(e) NA_real_)
        
        results <- bind_rows(results, tibble(
          protein_id = protein,
          mean_group_a = mean_a,
          mean_group_b = mean_b,
          log2FC = log2fc,
          pvalue = pval,
          n_group_a = length(vals_a),
          n_group_b = length(vals_b)
        ))
      }
      
      results %>%
        mutate(
          avg_expression = (mean_group_a + mean_group_b) / 2,  # For MA plot
          neg_log10_pval = -log10(pvalue),
          padj = p.adjust(pvalue, method = "BH"),
          significance = case_when(
            pvalue < input$pval_thresh & log2FC > input$fc_thresh ~ "Up",
            pvalue < input$pval_thresh & log2FC < -input$fc_thresh ~ "Down",
            TRUE ~ "NS"
          )
        ) %>%
        arrange(pvalue)
    })
  })
  
  output$volcano_plot <- renderPlot({
    req(diff_results())
    
    df <- diff_results()
    
    ggplot(df, aes(x = log2FC, y = neg_log10_pval, color = significance)) +
      geom_point(alpha = 0.6, size = 2) +
      geom_hline(yintercept = -log10(input$pval_thresh), linetype = "dashed", color = "gray50") +
      geom_vline(xintercept = c(-input$fc_thresh, input$fc_thresh), linetype = "dashed", color = "gray50") +
      scale_color_manual(values = c("Up" = "#e74c3c", "Down" = "#3498db", "NS" = "gray70")) +
      labs(x = "Log2 Fold Change", 
           y = "-Log10 P-value",
           title = paste(input$diff_group_a, "vs", input$diff_group_b),
           subtitle = paste("FC threshold:", input$fc_thresh, "| P-value threshold:", input$pval_thresh),
           color = "Direction") +
      theme_minimal(base_size = 14) +
      theme(
        plot.title = element_text(face = "bold", size = 16),
        legend.position = "bottom"
      )
  })

  # --- Feature 2: MA Plot ---
  output$ma_plot <- renderPlot({
    req(diff_results())

    df <- diff_results()

    ggplot(df, aes(x = avg_expression, y = log2FC, color = significance)) +
      geom_point(alpha = 0.6, size = 2) +
      geom_hline(yintercept = 0, linetype = "solid", color = "black", linewidth = 0.5) +
      geom_hline(yintercept = c(-input$fc_thresh, input$fc_thresh),
                 linetype = "dashed", color = "gray50") +
      scale_color_manual(values = c("Up" = "#e74c3c", "Down" = "#3498db", "NS" = "gray70")) +
      labs(x = "Average Expression (log2)",
           y = "Log2 Fold Change (M)",
           title = paste("MA Plot:", input$diff_group_a, "vs", input$diff_group_b),
           subtitle = "Identifying intensity-dependent bias",
           color = "Direction") +
      theme_minimal(base_size = 14) +
      theme(
        plot.title = element_text(face = "bold", size = 16),
        legend.position = "bottom"
      )
  })

  # Download handlers for plots
  output$download_volcano <- downloadHandler(
    filename = function() {
      paste0("volcano_plot_", Sys.Date(), ".png")
    },
    content = function(file) {
      png(file, width = 10, height = 8, units = "in", res = 300)

      df <- diff_results()

      p <- ggplot(df, aes(x = log2FC, y = neg_log10_pval, color = significance)) +
        geom_point(alpha = 0.6, size = 2) +
        geom_hline(yintercept = -log10(input$pval_thresh), linetype = "dashed", color = "gray50") +
        geom_vline(xintercept = c(-input$fc_thresh, input$fc_thresh), linetype = "dashed", color = "gray50") +
        scale_color_manual(values = c("Up" = "#e74c3c", "Down" = "#3498db", "NS" = "gray70")) +
        labs(x = "Log2 Fold Change",
             y = "-Log10 P-value",
             title = paste(input$diff_group_a, "vs", input$diff_group_b),
             subtitle = paste("FC threshold:", input$fc_thresh, "| P-value threshold:", input$pval_thresh),
             color = "Direction") +
        theme_minimal(base_size = 14) +
        theme(
          plot.title = element_text(face = "bold", size = 16),
          legend.position = "bottom"
        )

      print(p)
      dev.off()
    }
  )

  output$download_ma <- downloadHandler(
    filename = function() {
      paste0("ma_plot_", Sys.Date(), ".png")
    },
    content = function(file) {
      png(file, width = 10, height = 8, units = "in", res = 300)

      df <- diff_results()

      p <- ggplot(df, aes(x = avg_expression, y = log2FC, color = significance)) +
        geom_point(alpha = 0.6, size = 2) +
        geom_hline(yintercept = 0, linetype = "solid", color = "black", linewidth = 0.5) +
        geom_hline(yintercept = c(-input$fc_thresh, input$fc_thresh),
                   linetype = "dashed", color = "gray50") +
        scale_color_manual(values = c("Up" = "#e74c3c", "Down" = "#3498db", "NS" = "gray70")) +
        labs(x = "Average Expression (log2)",
             y = "Log2 Fold Change",
             title = paste("MA Plot:", input$diff_group_a, "vs", input$diff_group_b),
             subtitle = "Identifying intensity-dependent bias",
             color = "Direction") +
        theme_minimal(base_size = 14) +
        theme(
          plot.title = element_text(face = "bold", size = 16),
          legend.position = "bottom"
        )

      print(p)
      dev.off()
    }
  )

  output$n_up <- renderText({
    req(diff_results())
    sum(diff_results()$significance == "Up")
  })
  
  output$n_down <- renderText({
    req(diff_results())
    sum(diff_results()$significance == "Down")
  })
  
  output$n_ns <- renderText({
    req(diff_results())
    sum(diff_results()$significance == "NS")
  })
  
  output$top_proteins <- renderTable({
    req(diff_results())
    
    diff_results() %>%
      filter(significance != "NS") %>%
      head(8) %>%
      select(Protein = protein_id, `Log2 FC` = log2FC, `P-value` = pvalue) %>%
      mutate(
        `Log2 FC` = round(`Log2 FC`, 3),
        `P-value` = formatC(`P-value`, format = "e", digits = 2)
      )
  })
  
  output$diff_table <- renderDT({
    req(diff_results())
    
    diff_results() %>%
      select(
        Protein = protein_id,
        `Mean A` = mean_group_a,
        `Mean B` = mean_group_b,
        `Log2 FC` = log2FC,
        `P-value` = pvalue,
        `Adj. P-value` = padj,
        Direction = significance
      ) %>%
      mutate(
        `Mean A` = round(`Mean A`, 3),
        `Mean B` = round(`Mean B`, 3),
        `Log2 FC` = round(`Log2 FC`, 3),
        `P-value` = formatC(`P-value`, format = "e", digits = 2),
        `Adj. P-value` = formatC(`Adj. P-value`, format = "e", digits = 2)
      )
  }, 
  filter = "top",
  options = list(pageLength = 15, scrollX = TRUE))
  
  output$download_results <- downloadHandler(
    filename = function() {
      paste0("diff_analysis_", input$diff_group_a, "_vs_", input$diff_group_b, "_",
             Sys.Date(), ".csv")
    },
    content = function(file) {
      write_csv(diff_results(), file)
    }
  )

  # ========================================
  # Heatmap Tab - Feature 3
  # ========================================

  # Reactive: Get top proteins for heatmap based on diff results
  heatmap_proteins <- eventReactive(input$render_heatmap, {
    req(diff_results())

    df <- diff_results()

    # Filter for significant only if checkbox selected
    if (input$heatmap_sig_only) {
      df <- df %>% filter(significance != "NS")
    }

    # Check if we have any proteins left
    if (nrow(df) == 0) {
      return(NULL)
    }

    # Order by selected metric
    df <- df %>%
      mutate(abs_fc = abs(log2FC)) %>%
      arrange(
        if (input$heatmap_order_by == "pvalue") pvalue
        else if (input$heatmap_order_by == "padj") padj
        else desc(abs_fc)
      )

    # Take top N
    top_proteins <- head(df$protein_id, input$heatmap_top_n)

    return(top_proteins)
  })

  # Reactive: Prepare expression matrix and metadata for heatmap
  heatmap_data <- reactive({
    req(heatmap_proteins())

    top_prots <- heatmap_proteins()

    if (is.null(top_prots) || length(top_prots) == 0) {
      return(NULL)
    }

    # Extract expression matrix for top proteins
    mat <- expr_wide %>%
      filter(protein_id %in% top_prots) %>%
      column_to_rownames("protein_id") %>%
      as.matrix()

    # Reorder to match top_prots order
    mat <- mat[top_prots[top_prots %in% rownames(mat)], ]

    # Impute NAs with row median (same as PCA logic)
    for (i in seq_len(nrow(mat))) {
      row_median <- median(mat[i, ], na.rm = TRUE)
      if (!is.na(row_median) && any(is.na(mat[i, ]))) {
        mat[i, is.na(mat[i, ])] <- row_median
      }
    }

    # Z-score normalization per protein (row)
    mat_scaled <- t(scale(t(mat)))

    # Prepare annotation data frame
    if (input$heatmap_annotation != "none") {
      annot_df <- metadata %>%
        select(sample_id, annotation = !!sym(input$heatmap_annotation)) %>%
        column_to_rownames("sample_id")
    } else {
      annot_df <- NULL
    }

    list(
      matrix = mat_scaled,
      annotation = annot_df
    )
  })

  # Output: Clustered heatmap
  output$clustered_heatmap <- renderPlot({
    req(heatmap_data())

    hmap_data <- heatmap_data()

    if (is.null(hmap_data)) {
      plot.new()
      text(0.5, 0.5, "No proteins available for heatmap.\nTry adjusting filters or run differential analysis first.",
           cex = 1.2)
      return()
    }

    # Create heatmap using pheatmap
    pheatmap(
      hmap_data$matrix,
      annotation_col = hmap_data$annotation,
      cluster_rows = input$heatmap_cluster_rows,
      cluster_cols = input$heatmap_cluster_cols,
      clustering_distance_rows = "euclidean",
      clustering_distance_cols = "correlation",
      clustering_method = "ward.D2",
      color = colorRampPalette(c("#3498db", "white", "#e74c3c"))(100),
      breaks = seq(-3, 3, length.out = 101),  # Z-scores typically -3 to +3
      show_rownames = ifelse(nrow(hmap_data$matrix) <= 50, TRUE, FALSE),  # Hide row names if >50 proteins
      show_colnames = FALSE,  # Too many samples to show names
      fontsize_row = 8,
      main = paste("Top", nrow(hmap_data$matrix), "Differential Proteins"),
      border_color = NA
    )
  }, height = 700)

  # Download heatmap as PNG
  output$download_heatmap_png <- downloadHandler(
    filename = function() {
      paste0("heatmap_top_", input$heatmap_top_n, "_", Sys.Date(), ".png")
    },
    content = function(file) {
      req(heatmap_data())
      hmap_data <- heatmap_data()

      png(file, width = 12, height = 10, units = "in", res = 300)

      pheatmap(
        hmap_data$matrix,
        annotation_col = hmap_data$annotation,
        cluster_rows = input$heatmap_cluster_rows,
        cluster_cols = input$heatmap_cluster_cols,
        clustering_distance_rows = "euclidean",
        clustering_distance_cols = "correlation",
        clustering_method = "ward.D2",
        color = colorRampPalette(c("#3498db", "white", "#e74c3c"))(100),
        breaks = seq(-3, 3, length.out = 101),
        show_rownames = ifelse(nrow(hmap_data$matrix) <= 50, TRUE, FALSE),
        show_colnames = FALSE,
        fontsize_row = 8,
        main = paste("Top", nrow(hmap_data$matrix), "Differential Proteins"),
        border_color = NA
      )

      dev.off()
    }
  )

  # Download heatmap as PDF
  output$download_heatmap_pdf <- downloadHandler(
    filename = function() {
      paste0("heatmap_top_", input$heatmap_top_n, "_", Sys.Date(), ".pdf")
    },
    content = function(file) {
      req(heatmap_data())
      hmap_data <- heatmap_data()

      pdf(file, width = 12, height = 10)

      pheatmap(
        hmap_data$matrix,
        annotation_col = hmap_data$annotation,
        cluster_rows = input$heatmap_cluster_rows,
        cluster_cols = input$heatmap_cluster_cols,
        clustering_distance_rows = "euclidean",
        clustering_distance_cols = "correlation",
        clustering_method = "ward.D2",
        color = colorRampPalette(c("#3498db", "white", "#e74c3c"))(100),
        breaks = seq(-3, 3, length.out = 101),
        show_rownames = ifelse(nrow(hmap_data$matrix) <= 50, TRUE, FALSE),
        show_colnames = FALSE,
        fontsize_row = 8,
        main = paste("Top", nrow(hmap_data$matrix), "Differential Proteins"),
        border_color = NA
      )

      dev.off()
    }
  )
}

# ==============================================================================
# Run App
# ==============================================================================
shinyApp(ui = ui, server = server)
