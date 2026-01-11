# ==============================================================================
# Synthetic Proteomics Data Generator
# Generates realistic cross-study proteomics data for demo purposes
# ==============================================================================

library(tidyverse)

set.seed(42)

# --- Configuration ---
n_proteins <- 500
n_samples_per_study <- c(Trial_A = 60, Trial_B = 45, Trial_C = 35)
n_total_samples <- sum(n_samples_per_study)

# --- Generate Protein IDs ---
# Mix of UniProt IDs and gene symbols
uniprot_ids <- paste0("P", sprintf("%05d", sample(10000:99999, n_proteins * 0.6)))
gene_symbols <- c(
  "ALB", "TNF", "IL6", "IL10", "VEGFA", "EGFR", "HER2", "TP53", "BRCA1", "MYC",
  "KRAS", "BRAF", "AKT1", "PTEN", "CDK4", "MTOR", "PIK3CA", "MAPK1", "JAK2", "STAT3",
  "BCL2", "BAX", "CASP3", "FAS", "APAF1", "CYCS", "BID", "MCL1", "XIAP", "SURVIVIN",
  "CD4", "CD8A", "FOXP3", "CD25", "PD1", "PDL1", "CTLA4", "LAG3", "TIM3", "TIGIT",
  "GAPDH", "ACTB", "TUBB", "HSP90", "HSP70", "GRP78", "CALR", "CANX", "PDI", "ERO1",
  "COX1", "COX2", "LOX", "NOS2", "HIF1A", "VEGFR2", "ANGPT1", "FGF2", "PDGFB", "TGFb1"
)
additional_genes <- paste0("GENE", sprintf("%03d", 1:(n_proteins * 0.4 - length(gene_symbols))))

protein_ids <- c(uniprot_ids, gene_symbols, additional_genes) %>% 
  head(n_proteins)

# --- Generate Sample Metadata ---
metadata <- tibble()

for (trial in names(n_samples_per_study)) {
  n <- n_samples_per_study[trial]
  
  trial_meta <- tibble(
    sample_id = paste0(trial, "_S", sprintf("%03d", 1:n)),
    trial = trial,
    treatment = sample(c("Drug_A", "Drug_B", "Placebo"), n, replace = TRUE, 
                       prob = c(0.4, 0.4, 0.2)),
    timepoint = sample(c("Baseline", "Week_4", "Week_12", "Week_24"), n, replace = TRUE,
                       prob = c(0.3, 0.3, 0.25, 0.15)),
    response = sample(c("Responder", "Non-Responder", "Partial"), n, replace = TRUE,
                      prob = c(0.35, 0.40, 0.25)),
    age = round(rnorm(n, mean = 55, sd = 12)),
    sex = sample(c("M", "F"), n, replace = TRUE)
  )
  
  metadata <- bind_rows(metadata, trial_meta)
}

# --- Generate Expression Matrix ---
# Base expression levels (log2 scale, ~6-14 typical range)
base_expression <- rnorm(n_proteins, mean = 10, sd = 2)

# Create expression matrix
expression_matrix <- matrix(
  nrow = n_proteins,
  ncol = n_total_samples,
  dimnames = list(protein_ids, metadata$sample_id)
)

# Populate with realistic variation
for (i in seq_len(n_proteins)) {
  # Sample-to-sample variation
  expression_matrix[i, ] <- base_expression[i] + rnorm(n_total_samples, sd = 1.5)
}

# --- Add Biological Effects ---
# Treatment effects for specific proteins (simulate drug targets)
treatment_responsive_proteins <- sample(1:n_proteins, 50)
for (p in treatment_responsive_proteins) {
  drug_a_samples <- which(metadata$treatment == "Drug_A")
  drug_b_samples <- which(metadata$treatment == "Drug_B")
  
  effect_a <- runif(1, min = 0.5, max = 2.0) * sample(c(-1, 1), 1)
  effect_b <- runif(1, min = 0.3, max = 1.5) * sample(c(-1, 1), 1)
  
  expression_matrix[p, drug_a_samples] <- expression_matrix[p, drug_a_samples] + effect_a
  expression_matrix[p, drug_b_samples] <- expression_matrix[p, drug_b_samples] + effect_b
}

# Response-associated proteins
response_proteins <- sample(setdiff(1:n_proteins, treatment_responsive_proteins), 40)
for (p in response_proteins) {
  responder_samples <- which(metadata$response == "Responder")
  non_responder_samples <- which(metadata$response == "Non-Responder")
  
  effect <- runif(1, min = 0.8, max = 2.5) * sample(c(-1, 1), 1)
  
  expression_matrix[p, responder_samples] <- expression_matrix[p, responder_samples] + effect
  expression_matrix[p, non_responder_samples] <- expression_matrix[p, non_responder_samples] - effect * 0.5
}

# Trial batch effects
trial_effects <- c(Trial_A = 0, Trial_B = 0.3, Trial_C = -0.2)
for (trial in names(trial_effects)) {
  trial_samples <- which(metadata$trial == trial)
  expression_matrix[, trial_samples] <- expression_matrix[, trial_samples] + trial_effects[trial]
}

# --- Introduce Missing Values (realistic pattern) ---
# ~5-15% missingness, more common in low-abundance proteins
missingness_rate <- 0.08
low_abundance <- base_expression < quantile(base_expression, 0.3)

for (i in seq_len(n_proteins)) {
  rate <- ifelse(low_abundance[i], missingness_rate * 2, missingness_rate * 0.5)
  n_missing <- round(n_total_samples * rate)
  missing_idx <- sample(1:n_total_samples, n_missing)
  expression_matrix[i, missing_idx] <- NA
}

# --- Convert to Tidy Format and Save ---
expression_df <- as.data.frame(expression_matrix) %>%
  rownames_to_column("protein_id") %>%
  as_tibble()

# Save data
write_csv(expression_df, "data/expression_matrix.csv")
write_csv(metadata, "data/metadata.csv")

# Also save as RDS for faster loading
expression_long <- expression_df %>%
  pivot_longer(-protein_id, names_to = "sample_id", values_to = "log2_intensity") %>%
  left_join(metadata, by = "sample_id")

saveRDS(list(
  expression_wide = expression_df,
  expression_long = expression_long,
  metadata = metadata,
  protein_ids = protein_ids
), "data/proteomics_data.rds")

cat("✓ Generated", n_proteins, "proteins across", n_total_samples, "samples\n")
cat("✓ Trials:", paste(names(n_samples_per_study), collapse = ", "), "\n")
cat("✓ Saved to: data/expression_matrix.csv, data/metadata.csv, data/proteomics_data.rds\n")
