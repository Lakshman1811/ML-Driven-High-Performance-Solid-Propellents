"""Paper-faithful pipeline settings (Wang et al., ACS Appl. Energy Mater. 2025)."""
import os
from pathlib import Path

QUICK = os.environ.get("AISFD_QUICK", "").strip() in {"1", "true", "True"}

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "outputs" / "models"

DATA_CANDIDATES = [
    ROOT / "data" / "Supplementary Data 1.csv",
    Path(r"C:\department things\BTP\Problem_Statement\Supplementary Data 1.csv"),
]

# Two-branch MLP fusion head (paper Fig. S2). Full original layer lists are very large;
# these sizes keep the same architecture while remaining trainable on CPU.
LAYER_SIZES = {
    "isp": [256, 128, 64],
    "c_t": [256, 128, 64],
    "cstar": [256, 128, 64],
}

BATCH_SIZE = 512
EPOCHS = 4 if QUICK else 12
LEARNING_RATE = 1e-3
PATIENCE = 3 if QUICK else 4
SEED = 42
QUICK_TRAIN_ROWS = 12000
QUICK_SCREEN_ECS = 40

# Screening constraints (paper Fig. 8 / Section 3.4)
AL_MIN, AL_MAX = 10.0, 20.0
EC_MIN, EC_MAX = 10.0, 40.0
HTPB_MIN, HTPB_MAX = 12.0, 14.0

# Differential-evolution style search per EC (paper used Geatpy, pop=1000, gen=300)
SCREEN_POP = 12 if QUICK else 24
SCREEN_GEN = 8 if QUICK else 18

TOP_N = 100
FINAL_N = 7
ISP_PROMISING = 270.0

# ROC positive-class cutoffs (Wang et al. high-performance composite solid propellant bounds)
ROC_CUTOFFS = {"isp": 270.0, "c_t": 3200.0, "cstar": 1600.0}

FEATURE_BASE = [
    "Al", "EMs", "HTPB", "NH4CLO4",
    "C_mol", "H_mol", "O_mol", "N_mol", "Al_mol", "Cl_mol", "wt_H",
    "nHbondA", "nHbondD", "nNH2", "nAHC", "nACC", "nHC", "nRbond", "nR",
    "nNNO2", "nONO2", "nNO2", "nC(NO2)3", "nC(NO2)2", "nC(NO2)",
    "MinPartialCharge", "MaxPartialCharge", "MOLvolume",
    "nH", "nC", "nN", "nO", "PBF", "TPSA", "ob", "total energy",
    "molecular weight", "PMI3", "nOCH3", "nCH3", "Eccentricity",
    "PMI2", "PMI1", "NPR1", "NPR2",
] + [f"ESTATE_{i}" for i in range(70)]

UNUSED = [
    "nC(NO2)3", "nC(NO2)2", "nACC", "nONO2", "nOCH3", "Eccentricity", "nHC",
    "total energy", "PMI1", "PMI2", "PBF", "PMI3", "NPR1", "NPR2", "MOLvolume",
    "ESTATE_36", "ESTATE_16", "ESTATE_51", "ESTATE_65",
]
ALL_ZERO = [
    "ESTATE_3", "ESTATE_7", "ESTATE_13", "ESTATE_15", "ESTATE_20",
    "ESTATE_26", "ESTATE_31", "ESTATE_32", "ESTATE_33", "ESTATE_34",
    "ESTATE_38", "ESTATE_42", "ESTATE_48", "ESTATE_50", "ESTATE_55",
    "ESTATE_61", "ESTATE_66", "ESTATE_67", "ESTATE_68", "ESTATE_69",
]

DROP_COLS = set(UNUSED + ALL_ZERO)
FEATURE_COLUMNS = [c for c in FEATURE_BASE if c not in DROP_COLS]
FORMULATION_DIM = 10  # Al … Cl_mol (matches original Model_*_all slice X[:, 0:10])
