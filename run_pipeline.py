import torch
from pathlib import Path
import pandas as pd

from pipeline.baselines import run_baselines
from pipeline.config import MODEL_DIR, OUTPUT_DIR, QUICK, QUICK_SCREEN_ECS, QUICK_TRAIN_ROWS, SEED, FEATURE_COLUMNS
from pipeline.data import fit_scaler, load_raw, xy_from_frame
from pipeline.export_excel import write_workbook
from pipeline.plots import plot_loss_curves, plot_roc_curves
from pipeline.paper_figures import (
    generate_figure3,
    generate_figure4,
    generate_figure5,
    generate_figure6,
    generate_figure8,
)
from pipeline.screen import screen_all
from pipeline.train_mlp import save_scaler, set_seed, train_one_target


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"Starting Solid Propellant ML Discovery Pipeline | Device: {device}")
    print("=" * 70)

    print("\n[Step 1/6] Loading Dataset (Supplementary Data 1)...")
    df = load_raw()
    train_df = df[df["split"].astype(str).str.lower() == "train"].copy()
    test_df = df[df["split"].astype(str).str.lower() == "test"].copy()
    if QUICK:
        print(f"  [QUICK Mode Enabled] Sampling {QUICK_TRAIN_ROWS} train rows, {QUICK_SCREEN_ECS} ECs")
        train_df = train_df.sample(n=min(QUICK_TRAIN_ROWS, len(train_df)), random_state=SEED)
        test_df = test_df.sample(n=min(max(QUICK_TRAIN_ROWS // 5, 500), len(test_df)), random_state=SEED)
    print(f"  Dataset Statistics: Train={len(train_df):,} | Test={len(test_df):,} | Unique ECs={df['smiles'].nunique():,}")

    X_train, y_train = xy_from_frame(train_df)
    X_test, y_test = xy_from_frame(test_df)
    scaler = fit_scaler(X_train)
    Xtr = scaler.transform(X_train.values)
    Xte = scaler.transform(X_test.values)
    save_scaler(scaler, MODEL_DIR / "ss_X.pkl")

    print("\n[Step 2/6] Training Dual-Branch MLPs for Tc, Isp, C* (Target-Standardized)...")
    mlp_results = {}
    histories = {}
    metric_rows = []
    models = {}
    for target in ["isp", "c_t", "cstar"]:
        print(f"  --> Training MLP for {target.upper()}...")
        out = train_one_target(target, Xtr, y_train[target], Xte, y_test[target], device)
        mlp_results[target] = out
        histories[target] = out["history"]
        models[target] = out["model"]
        metric_rows.append({
            "model": "MLP",
            "target": target,
            "Train_R2": out["train"]["R2"],
            "Train_MAE": out["train"]["MAE"],
            "Train_RMSE": out["train"]["RMSE"],
            "Test_R2": out["test"]["R2"],
            "Test_MAE": out["test"]["MAE"],
            "Test_RMSE": out["test"]["RMSE"],
        })
        print(f"      Test Results: R²={out['test']['R2']:.3f} | MAE={out['test']['MAE']:.3f} | RMSE={out['test']['RMSE']:.3f}")

    print("\n[Step 3/6] Training Baseline Models (Ridge, AdaBoost, KNN) for Comparison...")
    metric_rows.extend(run_baselines(Xtr, y_train, Xte, y_test))

    print("\n[Step 4/6] Generating Loss Curves, ROC Curves, and Paper Figures...")
    loss_png = OUTPUT_DIR / "train_test_loss.png"
    roc_png = OUTPUT_DIR / "roc_curves.png"
    plot_loss_curves(histories, loss_png)
    _, roc_rows = plot_roc_curves(mlp_results, roc_png)
    print(f"  Saved: {loss_png.name}")
    print(f"  Saved: {roc_png.name}")

    fig3_png = OUTPUT_DIR / "figure3_dataset_distributions.png"
    fig4_png = OUTPUT_DIR / "figure4_model_comparison.png"
    fig5_png = OUTPUT_DIR / "figure5_parity_and_deviations.png"
    fig6_png = OUTPUT_DIR / "figure6_shap_and_mechanisms.png"
    fig8_png = OUTPUT_DIR / "figure8_screening_scatter.png"

    generate_figure3(df, fig3_png)
    print(f"  Saved: {fig3_png.name} (Dataset Distributions: Elements, EOF, MW, Targets)")
    generate_figure4(metric_rows, fig4_png)
    print(f"  Saved: {fig4_png.name} (Model Comparisons: MAE, RMSE, R²)")
    generate_figure5(mlp_results, fig5_png)
    print(f"  Saved: {fig5_png.name} (Parity Plots & Deviation Histograms)")

    # Figure 6: SHAP Beeswarm & Chemical Insights
    print("  Generating SHAP value beeswarm and chemical mechanism analysis...")
    generate_figure6(models["isp"], Xtr, Xte, FEATURE_COLUMNS, df, fig6_png, device=device)
    print(f"  Saved: {fig6_png.name} (SHAP Beeswarm & Chemical Mechanisms)")

    print("\n[Step 5/6] High-Throughput Screening of Energetic Compounds (GA Optimization)...")
    all_hits, top100, final7 = screen_all(df, scaler, models, device)
    generate_figure8(all_hits, fig8_png)
    print(f"  Saved: {fig8_png.name} (High-Throughput Screening Distribution)")

    print("\n[Step 6/6] Writing Excel Report with Embedded Visualizations & Metrics Table...")
    paper_figs = {
        "Figure 3: Dataset Distributions (Elements, EOF, MW, Target Distributions)": fig3_png,
        "Figure 4: Model Performance Comparison (MLP vs AdaBoost, KNN, Ridge)": fig4_png,
        "Figure 5: MLP Parity Scatter Plots & Deviation Histograms (Tc, Isp, C*)": fig5_png,
        "Figure 6: SHAP Interpretability Analysis & O/C Ratio Impact on Isp": fig6_png,
        "Figure 8: High-Throughput Energetic Compounds Screening Distribution": fig8_png,
    }
    xlsx = OUTPUT_DIR / "paper_pipeline_results.xlsx"
    write_workbook(all_hits, final7, metric_rows, loss_png, roc_png, paper_figs, xlsx)
    print(f"  Saved Excel: {xlsx}")

    print("\n" + "=" * 85)
    print("MODEL EVALUATION METRICS SUMMARY (R², MAE, RMSE)")
    print("=" * 85)
    mdf = pd.DataFrame(metric_rows)
    mdf["Model"] = mdf["model"].str.upper()
    mdf["Target"] = mdf["target"].replace({"c_t": "Tc (K)", "isp": "Isp (s)", "cstar": "C* (m/s)"})
    cols_order = ["Model", "Target", "Train_R2", "Train_MAE", "Train_RMSE", "Test_R2", "Test_MAE", "Test_RMSE"]
    mdf_display = mdf[cols_order].rename(columns={
        "Train_R2": "Train R²", "Train_MAE": "Train MAE", "Train_RMSE": "Train RMSE",
        "Test_R2": "Test R²", "Test_MAE": "Test MAE", "Test_RMSE": "Test RMSE"
    })
    print(mdf_display.to_string(index=False))

    print("\n" + "=" * 85)
    print("HIGH-THROUGHPUT SCREENING & FINAL CANDIDATES")
    print("=" * 85)
    print(f"Total Unique Molecules Screened: {len(all_hits):,}")
    print(f"Top Isp Compound Identified: {all_hits.iloc[0]['formula']} (Isp = {all_hits.iloc[0]['Isp_s']:.3f} s)")
    print(f"Final 7 Promising ECs Range: {final7['Isp_s'].min():.3f} s - {final7['Isp_s'].max():.3f} s (all > 270 s)")
    print(f"Results Excel Workbook: {xlsx}")
    print("=" * 85)


if __name__ == "__main__":
    main()


