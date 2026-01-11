# Proteomics Shiny Explorer V2 - Incremental Implementation Plan

## Philosophy: One Feature at a Time

Rather than a "big bang" refactor, this plan implements V2 features **incrementally** - each feature is fully functional, tested, and merged before starting the next. This approach:

- ✅ Maintains a working app at all times
- ✅ Allows early user feedback on each feature
- ✅ Reduces risk of large merge conflicts
- ✅ Provides immediate value after each feature
- ✅ Easier to test and validate

---

## Feature Implementation Order

Features are prioritized by:
1. **Research impact** - What researchers need most urgently
2. **Implementation risk** - Start with well-understood packages
3. **Dependencies** - Foundation features before dependent features
4. **User feedback** - Quick wins that demonstrate value

---

## 🎯 Quick Wins (Weeks 1-2)

These deliver immediate research value with minimal complexity:

### Feature 1: Sample Correlation Heatmap (Week 1, Day 1-2)
**Estimated effort: 6 hours**

**Why first?**
- Essential QC tool for detecting outliers and batch effects
- Simple to implement (one plot, no complex UI)
- High visual impact
- No external dependencies beyond base R

**Implementation**:
- Location: Overview tab, new subtab or section
- Uses existing `expr_wide` data
- Single plot with `pheatmap` or `corrplot`

**Tasks**:
- [ ] Add correlation heatmap plot to Overview tab
- [ ] Compute Pearson correlation matrix across samples
- [ ] Add hierarchical clustering with dendrograms
- [ ] Color annotations by trial/treatment
- [ ] Export button (PNG/PDF)

**Files to modify**:
- [app.R](app.R) (~30 lines added to Overview tab)

**Success criteria**:
- Heatmap renders correctly
- Outlier samples are visually identifiable
- Export works in PNG/PDF formats

---

### Feature 2: MA Plot in Differential Tab (Week 1, Day 3)
**Estimated effort: 4 hours**

**Why second?**
- Complements existing volcano plot
- Standard proteomics visualization
- Uses existing differential results
- Simple ggplot2 implementation

**Implementation**:
- Location: Differential Analysis tab, below volcano plot
- X-axis: Average expression (mean of both groups)
- Y-axis: Log2 fold change
- Color by significance

**Tasks**:
- [ ] Add MA plot output to Differential tab UI
- [ ] Compute average expression in server logic
- [ ] Create ggplot with colored points
- [ ] Add threshold lines (log2FC = ±1)
- [ ] Export button

**Files to modify**:
- [app.R](app.R) (~25 lines in Differential tab)

**Success criteria**:
- MA plot displays alongside volcano plot
- Points colored by significance
- Export works

---

### Feature 3: Basic Clustered Heatmap (Week 1, Day 4-5)
**Estimated effort: 8 hours**

**Why third?**
- Most requested figure for publications
- High visual impact
- Foundation for later clustering features
- Use simple `pheatmap` before `ComplexHeatmap`

**Implementation**:
- New tab: "Heatmap"
- Top N differential proteins (default 50)
- Z-score scaling
- Hierarchical clustering

**Tasks**:
- [ ] Create new "Heatmap" tab in UI
- [ ] Add slider for top N proteins (10-100)
- [ ] Extract top differential proteins
- [ ] Z-score scaling across samples
- [ ] Create pheatmap with dendrograms
- [ ] Add treatment/response annotations
- [ ] Export button (PNG/PDF)

**Files to modify**:
- [app.R](app.R) (~60 lines for new tab)

**Dependencies**:
- `pheatmap` package (simple installation)

**Success criteria**:
- Heatmap displays top differential proteins
- Clustering reveals biological patterns
- Annotations show sample groups
- Export produces publication-quality figure

---

### Feature 4: Pathway Enrichment (ORA) (Week 2, Days 1-5)
**Estimated effort: 16 hours**

**Why fourth?**
- Transforms protein lists into biological insights
- Most impactful research feature
- Well-established package (clusterProfiler)
- Foundation for GSEA later

**Implementation**:
- New tab: "Pathway Enrichment"
- Over-representation analysis using clusterProfiler
- GO Biological Process, KEGG, Reactome databases

**Tasks**:
- [ ] Install clusterProfiler and dependencies
- [ ] Create new "Pathway Enrichment" tab
- [ ] Add UI controls (database, thresholds)
- [ ] Extract significant proteins from differential results
- [ ] Run enrichGO() or enrichKEGG()
- [ ] Create dot plot visualization
- [ ] Create bar plot visualization
- [ ] Create downloadable results table
- [ ] Handle gene ID mapping (protein → gene symbol → Entrez)
- [ ] Error handling for mapping failures

**Files to modify**:
- [app.R](app.R) (~100 lines for new tab)

**Dependencies**:
- `clusterProfiler` (Bioconductor)
- `org.Hs.eg.db` (Bioconductor)
- `enrichplot` (Bioconductor)

**Success criteria**:
- Enrichment runs successfully on differential results
- Dot plot shows top 20 pathways
- Results downloadable as CSV
- Gene ID mapping >80% success rate

---

## 🔬 Statistical Rigor (Weeks 3-4)

### Feature 5: Batch Effect Correction (Week 3, Days 1-4)
**Estimated effort: 20 hours**

**Why next?**
- Critical for multi-center data quality
- Visible improvement in PCA plots
- Foundation for valid differential analysis
- Well-tested ComBat method

**Implementation**:
- Add "Batch Correction" section to Overview tab
- ComBat from `sva` package
- Before/after PCA comparison
- Option to apply globally

**Tasks**:
- [ ] Install `sva` package
- [ ] Add batch correction UI controls
- [ ] Implement ComBat wrapper function
- [ ] Create before/after PCA plots
- [ ] Add "Apply" button to commit correction
- [ ] Update reactive data flow
- [ ] Ensure downstream analyses use corrected data
- [ ] Add warning if already applied

**Files to modify**:
- [app.R](app.R) (~80 lines in Overview, reactive updates)

**Dependencies**:
- `sva` package (Bioconductor)

**Success criteria**:
- PCA shows reduced batch effect after correction
- Biological signal preserved (treatment groups still separate)
- Differential analysis uses corrected data when applied

---

### Feature 6: Advanced Imputation Methods (Week 3, Day 5 - Week 4, Day 2)
**Estimated effort: 24 hours**

**Why next?**
- Addresses proteomics-specific missing data (MNAR)
- Improves statistical power
- Multiple methods for comparison
- Educational value (sensitivity analysis)

**Implementation**:
- Add "Missing Data Handling" section to Overview tab
- 5 imputation methods: Hybrid, QRILC, MinProb, MissForest, Median
- Diagnostic plots for missingness pattern

**Tasks**:
- [ ] Install imputation packages (imp4p, missForest)
- [ ] Create missingness diagnostic plot (intensity vs % missing)
- [ ] Implement wrapper functions for each method
- [ ] Add method selection dropdown
- [ ] Add "Apply Imputation" button
- [ ] Create before/after density plots
- [ ] Add sensitivity analysis: compare methods
- [ ] Update reactive data flow

**Files to modify**:
- [app.R](app.R) (~100 lines in Overview, new utility functions)

**Dependencies**:
- `imp4p` (if available on CRAN/Bioconductor)
- `missForest` (CRAN)
- Alternative: implement MinProb and QRILC from scratch

**Success criteria**:
- All 5 methods successfully impute missing values
- Diagnostic plot identifies MNAR vs MCAR patterns
- Sensitivity analysis shows method impact on differential results

---

### Feature 7: Multi-Group ANOVA (Week 4, Days 3-5)
**Estimated effort: 16 hours**

**Why next?**
- Extends beyond pairwise comparisons
- Enables Drug A vs B vs Placebo studies
- Foundation for interaction testing
- Uses limma for efficiency

**Implementation**:
- Extend Differential Analysis tab
- Add "Multi-group ANOVA" option
- Post-hoc pairwise comparisons

**Tasks**:
- [ ] Install `limma` package
- [ ] Add "Analysis Type" selector (Pairwise vs ANOVA)
- [ ] Implement limma ANOVA wrapper
- [ ] Create ANOVA results table (F-statistic, p-value)
- [ ] Implement post-hoc pairwise tests
- [ ] Create interaction plots
- [ ] Update volcano plot for ANOVA context
- [ ] Add downloadable ANOVA results

**Files to modify**:
- [app.R](app.R) (~80 lines in Differential tab)

**Dependencies**:
- `limma` (Bioconductor)

**Success criteria**:
- ANOVA correctly identifies multi-group differences
- Post-hoc tests control for multiple comparisons
- Results match manual limma analysis

---

## 🏥 Clinical Trial Features (Weeks 5-7)

### Feature 8: Paired Sample Analysis (Week 5)
**Estimated effort: 18 hours**

**Why next?**
- Critical for longitudinal studies
- Increases statistical power
- Requires synthetic data update
- Foundation for trajectory analysis

**Implementation**:
- Add "Paired Analysis" option to Differential tab
- Requires subject_id in metadata
- Paired t-test on delta (change from baseline)

**Tasks**:
- [ ] Update synthetic data generator: add `subject_id`
- [ ] Regenerate data with paired samples
- [ ] Add "Paired Analysis" checkbox to UI
- [ ] Add "Subject ID" variable selector
- [ ] Implement paired t-test calculation
- [ ] Create paired volcano plot (delta vs p-value)
- [ ] Create slope graph visualization
- [ ] Create before-after boxplots with paired lines

**Files to modify**:
- [R/generate_data.R](R/generate_data.R) (~30 lines)
- [app.R](app.R) (~70 lines in Differential tab)

**Success criteria**:
- Paired analysis correctly computes within-subject changes
- Slope graph shows individual trajectories
- Statistical power increased vs unpaired (verify with known effect)

---

### Feature 9: Longitudinal Trajectory Analysis (Week 6)
**Estimated effort: 18 hours**

**Why next?**
- Timepoint data already exists (Baseline, Week_4, etc.)
- Natural extension of paired analysis
- Mixed-effects models are gold standard
- Profile plots are intuitive

**Implementation**:
- Add time-series plots to Protein Explorer
- Mixed-effects models for trajectory testing
- New section in Differential for time × treatment interactions

**Tasks**:
- [ ] Install `lme4` and `nlme` packages
- [ ] Update metadata with numeric timepoint
- [ ] Add profile plots to Protein Explorer tab
- [ ] Implement mixed-effects model wrapper
- [ ] Create trajectory testing in Differential tab
- [ ] Add confidence bands to profile plots
- [ ] Create spaghetti plots for individual subjects
- [ ] Results table: time × treatment interactions

**Files to modify**:
- [R/generate_data.R](R/generate_data.R) (~10 lines)
- [app.R](app.R) (~90 lines across tabs)

**Dependencies**:
- `lme4` (CRAN)
- `nlme` (CRAN)

**Success criteria**:
- Profile plots show mean trajectories by group
- Mixed models correctly identify time × treatment interactions
- Individual subject trajectories visible

---

### Feature 10: Survival Analysis (Week 7)
**Estimated effort: 24 hours**

**Why next?**
- Common clinical trial endpoint
- High research impact
- Mature packages (survival)
- Striking visualizations (KM curves)

**Implementation**:
- New tab: "Survival Analysis"
- Cox regression for protein associations
- Kaplan-Meier curves for selected proteins
- Forest plots for hazard ratios

**Tasks**:
- [ ] Install `survival` and `survminer` packages
- [ ] Update synthetic data: add `time_to_event`, `event_occurred`
- [ ] Create new "Survival Analysis" tab
- [ ] Implement Cox regression for all proteins
- [ ] Create Cox results table (HR, 95% CI, p-value)
- [ ] Add protein selector for KM plots
- [ ] Create Kaplan-Meier plot with risk tables
- [ ] Implement stratification (median split, tertiles)
- [ ] Create forest plot for top proteins
- [ ] Add downloadable survival results

**Files to modify**:
- [R/generate_data.R](R/generate_data.R) (~20 lines)
- [app.R](app.R) (~120 lines for new tab)

**Dependencies**:
- `survival` (CRAN)
- `survminer` (CRAN)

**Success criteria**:
- Cox regression runs on all proteins
- KM curves stratify by protein expression
- Forest plot displays HRs with confidence intervals
- Results match manual survival analysis

---

## 📊 Enhanced Visualizations (Week 8)

### Feature 11: Advanced Heatmap with ComplexHeatmap (Week 8, Days 1-3)
**Estimated effort: 16 hours**

**Why now?**
- Upgrade existing basic heatmap
- ComplexHeatmap offers publication-quality output
- More flexible annotations
- Community detection/clustering

**Implementation**:
- Replace pheatmap with ComplexHeatmap
- Enhanced annotations
- Row/column splitting by clusters
- Custom color scales

**Tasks**:
- [ ] Install `ComplexHeatmap` and `circlize`
- [ ] Replace pheatmap implementation
- [ ] Add annotation tracks (treatment, response, trial)
- [ ] Add clustering method options
- [ ] Implement row/column splitting
- [ ] Custom color palette selector
- [ ] Export cluster membership as CSV
- [ ] High-res export options

**Files to modify**:
- [app.R](app.R) (~50 lines, replacing existing heatmap)

**Dependencies**:
- `ComplexHeatmap` (Bioconductor)
- `circlize` (CRAN)

**Success criteria**:
- Heatmap has publication-quality annotations
- Multiple clustering options work correctly
- Export produces high-resolution figures (300 DPI)

---

### Feature 12: Additional QC Plots (Week 8, Days 4-5)
**Estimated effort: 12 hours**

**Why now?**
- Complete QC suite
- Easy additions (independent plots)
- High utility for data quality assessment

**Implementation**:
- Add 3 new QC plots to Overview tab:
  1. RLE (Relative Log Expression) plot
  2. CV (Coefficient of Variation) plot
  3. Missingness pattern heatmap

**Tasks**:
- [ ] Create RLE plot function
- [ ] Add RLE plot to Overview tab
- [ ] Create CV distribution plot
- [ ] Add CV plot to Overview
- [ ] Create missingness pattern heatmap
- [ ] Add missingness heatmap to Overview
- [ ] Organize Overview tab with subtabs/sections
- [ ] Export buttons for all plots

**Files to modify**:
- [app.R](app.R) (~60 lines in Overview)

**Success criteria**:
- RLE plot centers at zero after normalization
- CV plot distinguishes technical vs biological variation
- Missingness heatmap reveals systematic patterns

---

## 🌐 Network Analysis (Week 9)

### Feature 13: Protein-Protein Interaction Networks (Week 9)
**Estimated effort: 20 hours**

**Why now?**
- Biological context for protein lists
- Interactive visualization
- STRING database well-established
- Community detection adds insight

**Implementation**:
- New subtab under Pathway Enrichment: "Networks"
- STRING API integration
- Interactive visNetwork plot
- Hub protein identification

**Tasks**:
- [ ] Install `httr`, `igraph`, `visNetwork`
- [ ] Create STRING API query function
- [ ] Parse STRING response to igraph object
- [ ] Create network visualization UI
- [ ] Implement visNetwork interactive plot
- [ ] Color nodes by log2FC, size by p-value
- [ ] Implement community detection (Louvain)
- [ ] Create hub analysis table
- [ ] Add confidence score slider
- [ ] Export network as GraphML for Cytoscape

**Files to modify**:
- [app.R](app.R) (~100 lines, new subtab)

**Dependencies**:
- `httr` (CRAN)
- `igraph` (CRAN)
- `visNetwork` (CRAN)

**Success criteria**:
- STRING API returns PPI networks
- Interactive network is responsive
- Community detection identifies functional modules
- Export works for Cytoscape

---

### Feature 14: GSEA Implementation (Week 9, continued)
**Estimated effort: 16 hours**

**Why now?**
- Complements existing ORA
- Detects subtle coordinated changes
- fgsea is fast and well-documented
- Enrichment plots are publication-quality

**Implementation**:
- Add GSEA option to Pathway Enrichment tab
- Use fgsea package
- MSigDB gene sets

**Tasks**:
- [ ] Install `fgsea` and `msigdbr` packages
- [ ] Download MSigDB gene sets (Hallmark, C2, C5)
- [ ] Add GSEA checkbox to UI
- [ ] Implement fgsea wrapper
- [ ] Create GSEA results table
- [ ] Create enrichment score plots
- [ ] Create ridge plot for NES distribution
- [ ] Add leading edge gene extraction
- [ ] Downloadable GSEA results

**Files to modify**:
- [app.R](app.R) (~80 lines in Pathway tab)

**Dependencies**:
- `fgsea` (Bioconductor)
- `msigdbr` (CRAN)

**Success criteria**:
- GSEA runs on ranked protein list
- Enrichment plots match fgsea documentation
- Leading edge genes identified correctly

---

## 📝 Reproducibility Features (Weeks 10-11)

### Feature 15: RMarkdown Report Generation (Week 10)
**Estimated effort: 20 hours**

**Why now?**
- Enables reproducible research
- Methods section auto-generation
- Publication support
- Comprehensive feature set now available

**Implementation**:
- New tab: "Export & Reports"
- Parameterized RMarkdown template
- Methods, results, figures embedded

**Tasks**:
- [ ] Create `inst/rmarkdown/report_template.Rmd`
- [ ] Design report structure (methods, results, figures)
- [ ] Create "Export & Reports" tab UI
- [ ] Collect all analysis parameters
- [ ] Implement report generation function
- [ ] Add preview capability (HTML)
- [ ] Support multiple formats (HTML, PDF, Word)
- [ ] Include all plots as embedded figures
- [ ] Include results tables
- [ ] Auto-generate methods text

**Files to create**:
- `inst/rmarkdown/report_template.Rmd` (new)

**Files to modify**:
- [app.R](app.R) (~80 lines for new tab)

**Dependencies**:
- `rmarkdown` (CRAN)
- `knitr` (CRAN)

**Success criteria**:
- Report generates successfully in all formats
- Methods section accurately describes analysis
- All figures embedded correctly
- Tables formatted properly

---

### Feature 16: Session Management & Code Export (Week 11)
**Estimated effort: 16 hours**

**Why last?**
- Requires all features to be complete
- Enables full reproducibility
- Code export helps users learn R

**Implementation**:
- Save/load session (.RData)
- Export R script of analysis
- Parameter logging (JSON)

**Tasks**:
- [ ] Implement session save functionality
- [ ] Implement session load functionality
- [ ] Track all user actions for code generation
- [ ] Generate R script from UI actions
- [ ] Export parameter JSON
- [ ] Add download buttons
- [ ] Test save/load cycle
- [ ] Validate generated R script runs standalone

**Files to modify**:
- [app.R](app.R) (~70 lines in Export tab, global tracking)

**Success criteria**:
- Saved sessions reload correctly
- Generated R script reproduces all analyses
- Parameter JSON is complete and accurate

---

## 📋 Implementation Summary

### Total Timeline: 11 Weeks (252 hours)

| Week | Features | Focus | Effort |
|------|----------|-------|--------|
| 1 | 1-3 | Quick wins (heatmaps, MA plot) | 18 hours |
| 2 | 4 | Pathway enrichment (ORA) | 16 hours |
| 3 | 5-6 | Batch correction, imputation | 44 hours |
| 4 | 6-7 | Imputation (cont.), ANOVA | 24 hours |
| 5 | 8 | Paired analysis | 18 hours |
| 6 | 9 | Longitudinal trajectories | 18 hours |
| 7 | 10 | Survival analysis | 24 hours |
| 8 | 11-12 | Advanced heatmap, QC plots | 28 hours |
| 9 | 13-14 | Networks, GSEA | 36 hours |
| 10 | 15 | Report generation | 20 hours |
| 11 | 16 | Session management | 16 hours |
| **Total** | **16 features** | | **252 hours** |

---

## 🔄 Development Workflow (Per Feature)

### 1. Plan (30 min)
- Review feature requirements
- Identify files to modify
- Check package dependencies
- Design UI mockup (on paper/whiteboard)

### 2. Implement (80% of time)
- Install required packages
- Update data generators if needed
- Add UI components
- Implement server logic
- Create helper functions if needed
- Test manually during development

### 3. Test (15% of time)
- Test with synthetic data
- Test edge cases (no significant results, all missing, etc.)
- Test exports/downloads
- Check error handling
- Visual inspection of plots

### 4. Document (5% of time)
- Update [CLAUDE.md](CLAUDE.md) with new feature
- Add comments to code
- Update README.md feature list
- Take screenshots for documentation

### 5. Commit & Review
- Git commit with descriptive message
- Optional: Create feature branch and PR
- Demo to stakeholders
- Gather feedback

---

## 📦 Cumulative Package Dependencies

Packages will be installed incrementally:

**Week 1**: `pheatmap`, `corrplot`
**Week 2**: `clusterProfiler`, `org.Hs.eg.db`, `enrichplot` (Bioconductor)
**Week 3**: `sva`, `imp4p` or implement manually, `missForest`
**Week 4**: `limma` (Bioconductor)
**Week 5**: None (use existing data)
**Week 6**: `lme4`, `nlme`
**Week 7**: `survival`, `survminer`
**Week 8**: `ComplexHeatmap`, `circlize` (Bioconductor)
**Week 9**: `httr`, `igraph`, `visNetwork`, `fgsea`, `msigdbr`
**Week 10**: `rmarkdown`, `knitr`
**Week 11**: None

---

## ✅ Success Metrics

### After Quick Wins (Week 2)
- [ ] 4 new visualizations functional
- [ ] Pathway enrichment running on test data
- [ ] Demo to stakeholders completed

### After Statistical Rigor (Week 4)
- [ ] Batch effects visibly reduced in PCA
- [ ] 5 imputation methods working
- [ ] ANOVA handles 3+ groups

### After Clinical Features (Week 7)
- [ ] Paired analysis increases power (verify with test case)
- [ ] Longitudinal models detect time × treatment interactions
- [ ] Survival analysis produces KM curves

### After Visualization (Week 8)
- [ ] 10+ publication-quality visualizations
- [ ] All plots exportable as high-res PNG/PDF

### After Network Analysis (Week 9)
- [ ] PPI networks display correctly
- [ ] GSEA complements ORA results

### After Reproducibility (Week 11)
- [ ] Reports generate in 3 formats
- [ ] Session save/load works reliably
- [ ] Generated R script reproduces analyses

---

## 🚨 Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Bioconductor installation fails | Medium | Provide Docker image, detailed docs |
| Gene ID mapping loses proteins | High | Support multiple ID types, log failures |
| Features take longer than estimated | High | Build buffer time, prioritize ruthlessly |
| Package updates break compatibility | Low | Use `renv` for version locking (optional) |
| User feedback requires redesign | Medium | Implement feedback in next sprint |

---

## 🎓 Learning Resources

For each new package, consult:

**clusterProfiler**: [Official book](https://yulab-smu.top/biomedical-knowledge-mining-book/)
**limma**: [User's Guide](https://bioconductor.org/packages/release/bioc/vignettes/limma/inst/doc/usersguide.pdf)
**sva**: ComBat [tutorial](https://www.bu.edu/jlab/wp-assets/ComBat/Download_files/ComBat_Tutorial.pdf)
**survival**: [Survival Analysis in R](https://www.emilyzabor.com/tutorials/survival_analysis_in_r_tutorial.html)
**ComplexHeatmap**: [Complete Reference](https://jokergoo.github.io/ComplexHeatmap-reference/book/)

---

## 🚀 Next Steps

### This Week: Feature 1 - Sample Correlation Heatmap

1. **Review current app.R** - Understand Overview tab structure
2. **Install pheatmap** - Test on synthetic data outside Shiny
3. **Implement in app** - Add new section to Overview tab
4. **Test thoroughly** - Various metadata configurations
5. **Document** - Update CLAUDE.md
6. **Demo** - Show stakeholders, gather feedback

Then move to Feature 2, and so on.

---

## 📝 Notes

- **Architecture**: Keep single-file app.R for now (modularization can come later if needed)
- **Testing**: Manual testing sufficient for research tool; add unit tests if time permits
- **Deployment**: Continue using local RStudio; consider Shiny Server if multi-user access needed
- **Version control**: Commit after each feature; consider feature branches for complex changes

---

*Document version: 1.0*
*Created: 2026-01-11*
*Focus: Incremental, feature-by-feature implementation of V2 Research Roadmap*

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[
  {
    "content": "Review and understand current app architecture",
    "activeForm": "Reviewing current app architecture",
    "status": "completed"
  },
  {
    "content": "Create incremental V2 implementation plan",
    "activeForm": "Creating incremental V2 implementation plan",
    "status": "in_progress"
  }
]