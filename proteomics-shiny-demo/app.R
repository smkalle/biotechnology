# ==============================================================================
# Shiny Proteomics Explorer
# Cross-study proteomics visualization, QC, and differential analysis
# ==============================================================================

library(shiny)
library(tidyverse)
library(DT)
library(broom)

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
    
    fluidRow(
      # Summary Stats
      column(4,
        wellPanel(
          h4(icon("database"), " Dataset Summary"),
          hr(),
          fluidRow(
            column(4, 
              div(class = "text-center",
                h2(textOutput("n_samples"), class = "text-primary"),
                p("Samples")
              )
            ),
            column(4,
              div(class = "text-center",
                h2(textOutput("n_proteins"), class = "text-primary"),
                p("Proteins")
              )
            ),
            column(4,
              div(class = "text-center",
                h2(textOutput("n_trials"), class = "text-primary"),
                p("Studies")
              )
            )
          ),
          hr(),
          h5("Breakdown by Trial:"),
          tableOutput("trial_summary")
        )
      ),
      
      # Missingness Histogram
      column(4,
        wellPanel(
          h4(icon("chart-area"), " Data Quality: Missingness"),
          selectInput("miss_view", "View by:", 
                      choices = c("Per Sample" = "sample", "Per Protein" = "protein")),
          plotOutput("missingness_plot", height = "280px")
        )
      ),
      
      # PCA Controls
      column(4,
        wellPanel(
          h4(icon("project-diagram"), " PCA Settings"),
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
      )
    ),
    
    fluidRow(
      column(12,
        wellPanel(
          h4(icon("compress-arrows-alt"), " Principal Component Analysis"),
          plotOutput("pca_plot", height = "400px")
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
      mat[i, is.na(mat[i, ])] <- median(mat[i, ], na.rm = TRUE)
    }
    
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
    
    p + 
      stat_ellipse(level = 0.68, linetype = "dashed") +
      labs(x = paste0("PC1 (", var_exp[1], "%)"),
           y = paste0("PC2 (", var_exp[2], "%)"),
           title = "PCA of Protein Expression",
           color = str_to_title(input$pca_color),
           shape = ifelse(input$pca_shape == "none", NULL, str_to_title(input$pca_shape))) +
      theme_minimal(base_size = 14) +
      theme(
        plot.title = element_text(face = "bold", size = 16),
        legend.position = "right"
      ) +
      scale_color_brewer(palette = "Set2")
  })
  
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
}

# ==============================================================================
# Run App
# ==============================================================================
shinyApp(ui = ui, server = server)
