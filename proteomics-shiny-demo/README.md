# 🧬 Proteomics Shiny Explorer

A cross-study proteomics visualization and analysis dashboard built with R/Shiny. Enables QC assessment, protein-level exploration, and differential expression analysis across multiple clinical trials.

![R](https://img.shields.io/badge/R-4.0+-blue.svg)
![Shiny](https://img.shields.io/badge/Shiny-1.7+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Features

- **Cross-Study Overview**: Sample/protein counts, missingness QC, and PCA visualization colored by study/treatment/response
- **Protein Explorer**: Interactive boxplot/violin plots with t-test or Wilcoxon statistics for any protein
- **Differential Analysis**: Compute log2 fold-change and p-values across groups with interactive volcano plot and downloadable results

## 📊 Data Schema

### Expression Matrix (`expression_matrix.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `protein_id` | string | UniProt accession (e.g., P01234) or gene symbol (e.g., EGFR) |
| `[sample_id]` | numeric | Log2-transformed intensity values. One column per sample. NA = missing |

**Example:**
```
protein_id,Trial_A_S001,Trial_A_S002,Trial_B_S001,...
P01234,10.23,10.45,9.87,...
EGFR,12.34,NA,12.01,...
```

### Sample Metadata (`metadata.csv`)

| Column | Type | Values | Description |
|--------|------|--------|-------------|
| `sample_id` | string | `{Trial}_{Sxxx}` | Unique sample identifier |
| `trial` | string | `Trial_A`, `Trial_B`, `Trial_C` | Clinical study identifier |
| `treatment` | string | `Drug_A`, `Drug_B`, `Placebo` | Treatment arm |
| `timepoint` | string | `Baseline`, `Week_4`, `Week_12`, `Week_24` | Collection timepoint |
| `response` | string | `Responder`, `Non-Responder`, `Partial` | Clinical response classification |
| `age` | integer | 18-90 | Patient age at enrollment |
| `sex` | string | `M`, `F` | Biological sex |

**Example:**
```csv
sample_id,trial,treatment,timepoint,response,age,sex
Trial_A_S001,Trial_A,Drug_A,Baseline,Responder,54,F
Trial_A_S002,Trial_A,Placebo,Week_4,Non-Responder,62,M
```

## 🚀 Quick Start

### Prerequisites

```r
install.packages(c("shiny", "tidyverse", "DT", "broom"))
```

### Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/proteomics-shiny-demo.git
cd proteomics-shiny-demo

# Generate synthetic data (first run only)
Rscript R/generate_data.R

# Launch the app
R -e "shiny::runApp('.')"
```

Or in RStudio: Open `app.R` and click "Run App"

### Using Your Own Data

1. Replace `data/expression_matrix.csv` with your log2-transformed proteomics data
2. Replace `data/metadata.csv` with your sample annotations
3. Ensure column names match the schema above
4. Restart the app

## 📁 Project Structure

```
proteomics-shiny-demo/
├── app.R                    # Main Shiny application
├── R/
│   └── generate_data.R      # Synthetic data generator
├── data/
│   ├── expression_matrix.csv
│   ├── metadata.csv
│   └── proteomics_data.rds  # Pre-processed R object
├── www/                     # Static assets (CSS, images)
└── README.md
```

## 🖼️ Screenshots

### Overview Tab
- Dataset summary statistics
- Missingness histogram (per-sample or per-protein view)
- Interactive PCA plot with grouping options

### Protein Explorer Tab
- Searchable protein selector (500 proteins)
- Violin + boxplot visualization
- Real-time statistical testing (t-test / Wilcoxon)
- Distribution density plots

### Differential Analysis Tab
- Group comparison selector
- Adjustable FC and p-value thresholds
- Interactive volcano plot
- Searchable results table with CSV download

## 🔧 Technical Details

### Statistical Methods

| Analysis | Method | Description |
|----------|--------|-------------|
| Missingness | Per-row/column counting | % NA values |
| PCA | `prcomp()` with scaling | Row-median imputation for NAs |
| Two-group comparison | Welch's t-test / Wilcoxon | User-selectable |
| Multiple testing | Benjamini-Hochberg | `p.adjust(method="BH")` |

### Performance

- **500 proteins × 140 samples** loads in <2 seconds
- Differential analysis completes in ~5 seconds
- PCA computed once and cached

## 🏗️ Extending the App

### Add New Metadata Columns

1. Add column to `metadata.csv`
2. Update dropdown choices in `ui` 
3. Reference new column in `server` with `.data[[input$variable]]`

### Add New Plot Types

```r
# Example: Add MA plot to Differential tab
output$ma_plot <- renderPlot({
  df <- diff_results()
  ggplot(df, aes(x = (mean_group_a + mean_group_b)/2, y = log2FC)) +
    geom_point(aes(color = significance), alpha = 0.5) +
    labs(x = "Average Expression", y = "Log2 FC")
})
```

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 References

- [Shiny Documentation](https://shiny.rstudio.com/)
- [ggplot2 Reference](https://ggplot2.tidyverse.org/)
- [DT Package](https://rstudio.github.io/DT/)

---

**Built for demonstrating cross-study proteomics analysis workflows** | Inspired by pharma industry best practices
