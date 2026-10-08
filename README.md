# Machine Learning-Driven Discovery of High-Performance Solid Propellants

Replication pipeline for the machine learning and genetic algorithm framework from *Wang et al., ACS Appl. Energy Mater. 2025, 8, 6756–6767*.

## Overview
This repository trains dual-branch Multilayer Perceptrons (MLPs) for predicting key propellant performance characteristics ($I_{sp}$, $T_c$, $C^*$), compares them against baseline algorithms (Ridge Regression, AdaBoost, KNN), performs SHAP interpretability analysis, conducts high-throughput genetic algorithm screening over 1,079 energetic compounds (ECs), and outputs structured performance tables and research figures.

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run the Full Discovery Pipeline
```bash
python run_pipeline.py
```

## Generated Outputs (`outputs/`)
* `paper_pipeline_results.xlsx` (or `paper_pipeline_results_latest.xlsx`):
  * **Top100_and_Final7**: Top 100 screened energetic compounds (left) and Final 7 promising ECs (right).
  * **Curves_and_ROC**: Train vs Test Loss curves and 3-target ROC curves with AUC annotations.
  * **Paper_Figures_and_SHAP**: Embedded Figures 3, 4, 5, 6 (SHAP beeswarm & chemical mechanisms), and 8.
* `train_test_loss.png`: Stabilized train and test MSE loss curves.
* `roc_curves.png`: True Positive Rate vs False Positive Rate ROC curves for $I_{sp}, T_c, C^*$ with annotated AUC.
* `figure3_dataset_distributions.png`: Dataset element counts, EOF, MW, and target violin distributions.
* `figure4_model_comparison.png`: MAE, RMSE, and $R^2$ model comparison.
* `figure5_parity_and_deviations.png`: Parity hexbin scatter plots and test deviation histograms.
* `figure6_shap_and_mechanisms.png`: SHAP beeswarm summary plot and O/C chemical mechanisms.
* `figure8_screening_scatter.png`: High-throughput screening scatter plot ($I_{sp}$ vs Formulation index).
* `all_screened_ecs.csv`, `top100_ecs.csv`, `final7_ecs.csv`: Screened formulations and predicted performance metrics.


