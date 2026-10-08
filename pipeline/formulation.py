import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, FORMULATION_DIM

MW_AL = 26.98
MW_HTPB = 86.0
MW_AP = 117.49
HTPB_C, HTPB_H, HTPB_O = 10.0, 15.40, 0.07
AP_H, AP_O, AP_N, AP_CL = 4.0, 4.0, 1.0, 1.0
TOTAL_MASS = 1000.0


def moles_from_mix(al, ems, htpb, ap, nC, nH, nO, nN, mw_ec):
    al = np.asarray(al, dtype=np.float64)
    ems = np.asarray(ems, dtype=np.float64)
    htpb = np.asarray(htpb, dtype=np.float64)
    ap = np.asarray(ap, dtype=np.float64)
    mass_al = TOTAL_MASS * al / 100.0
    mass_ec = TOTAL_MASS * ems / 100.0
    mass_htpb = TOTAL_MASS * htpb / 100.0
    mass_ap = TOTAL_MASS * ap / 100.0
    mol_al = mass_al / MW_AL
    mol_ec = mass_ec / np.maximum(mw_ec, 1e-8)
    mol_htpb = mass_htpb / MW_HTPB
    mol_ap = mass_ap / MW_AP
    C = mol_ec * nC + mol_htpb * HTPB_C
    H = mol_ec * nH + mol_htpb * HTPB_H + mol_ap * AP_H
    O = mol_ec * nO + mol_htpb * HTPB_O + mol_ap * AP_O
    N = mol_ec * nN + mol_ap * AP_N
    Al = mol_al
    Cl = mol_ap * AP_CL
    wt_H = nH * 100.0 / np.maximum(mw_ec, 1e-8)
    return C, H, O, N, Al, Cl, wt_H


def build_feature_matrix(al, ems, htpb, ap, mol_row: pd.Series, template: pd.Series):
    C, H, O, N, Al, Cl, wt_H = moles_from_mix(
        al, ems, htpb, ap,
        mol_row["nC"], mol_row["nH"], mol_row["nO"], mol_row["nN"],
        mol_row["molecular weight"],
    )
    n = np.asarray(al).reshape(-1).size
    frame = pd.DataFrame({col: np.full(n, template[col], dtype=np.float64) for col in FEATURE_COLUMNS})
    frame["Al"] = np.asarray(al, dtype=np.float64).reshape(-1)
    frame["EMs"] = np.asarray(ems, dtype=np.float64).reshape(-1)
    frame["HTPB"] = np.asarray(htpb, dtype=np.float64).reshape(-1)
    frame["NH4CLO4"] = np.asarray(ap, dtype=np.float64).reshape(-1)
    frame["C_mol"] = np.asarray(C, dtype=np.float64).reshape(-1)
    frame["H_mol"] = np.asarray(H, dtype=np.float64).reshape(-1)
    frame["O_mol"] = np.asarray(O, dtype=np.float64).reshape(-1)
    frame["N_mol"] = np.asarray(N, dtype=np.float64).reshape(-1)
    frame["Al_mol"] = np.asarray(Al, dtype=np.float64).reshape(-1)
    frame["Cl_mol"] = np.asarray(Cl, dtype=np.float64).reshape(-1)
    frame["wt_H"] = np.full(n, float(wt_H), dtype=np.float64)
    return frame[FEATURE_COLUMNS].to_numpy(np.float32)


def oc_ratio(al, ems, htpb, ap, mol_row):
    C, H, O, N, Al, Cl, wt_H = moles_from_mix(
        al, ems, htpb, ap,
        mol_row["nC"], mol_row["nH"], mol_row["nO"], mol_row["nN"],
        mol_row["molecular weight"],
    )
    C = float(np.asarray(C).reshape(-1)[0])
    O = float(np.asarray(O).reshape(-1)[0])
    if C <= 1e-12:
        return np.nan
    return O / C


def formula_from_counts(nC, nH, nN, nO):
    def _fmt(sym, n):
        n = float(n)
        if abs(n - round(n)) < 1e-6:
            n = int(round(n))
            return f"{sym}{n}" if n != 1 else sym
        return f"{sym}{n:.2f}"
    return _fmt("C", nC) + _fmt("H", nH) + _fmt("N", nN) + _fmt("O", nO)


assert FEATURE_COLUMNS[:FORMULATION_DIM] == [
    "Al", "EMs", "HTPB", "NH4CLO4", "C_mol", "H_mol", "O_mol", "N_mol", "Al_mol", "Cl_mol"
]
