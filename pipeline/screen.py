import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from .config import (
    AL_MAX,
    AL_MIN,
    EC_MAX,
    EC_MIN,
    FEATURE_COLUMNS,
    FINAL_N,
    HTPB_MAX,
    HTPB_MIN,
    ISP_PROMISING,
    QUICK,
    QUICK_SCREEN_ECS,
    SCREEN_GEN,
    SCREEN_POP,
    SEED,
    TOP_N,
)
from .formulation import build_feature_matrix, formula_from_counts, oc_ratio
from .train_mlp import predict_numpy


def _formula(smiles, nC, nH, nN, nO):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        try:
            return CalcMolFormula(mol)
        except Exception:
            pass
    return formula_from_counts(nC, nH, nN, nO)


def _clip_population(pop):
    al = np.clip(pop[:, 0], AL_MIN, AL_MAX)
    ems = np.clip(pop[:, 1], EC_MIN, EC_MAX)
    htpb = np.clip(pop[:, 2], HTPB_MIN, HTPB_MAX)
    ap = 100.0 - al - ems - htpb
    ap = np.maximum(ap, 0.0)
    total = al + ems + htpb + ap
    al, ems, htpb, ap = al / total * 100, ems / total * 100, htpb / total * 100, ap / total * 100
    return np.column_stack([al, ems, htpb, ap])


@torch.no_grad()
def _isp_of_pop(pop, mol_row, template, scaler, model, device):
    X = build_feature_matrix(pop[:, 0], pop[:, 1], pop[:, 2], pop[:, 3], mol_row, template)
    Xs = scaler.transform(X)
    return predict_numpy(model, Xs, device)


def optimize_ec(mol_row, template, scaler, model_isp, device, rng):
    pop = np.column_stack([
        rng.uniform(AL_MIN, AL_MAX, SCREEN_POP),
        rng.uniform(EC_MIN, EC_MAX, SCREEN_POP),
        rng.uniform(HTPB_MIN, HTPB_MAX, SCREEN_POP),
        np.zeros(SCREEN_POP),
    ])
    pop = _clip_population(pop)
    fitness = _isp_of_pop(pop, mol_row, template, scaler, model_isp, device)
    best = pop[int(np.argmax(fitness))].copy()
    best_f = float(np.max(fitness))

    for _ in range(SCREEN_GEN):
        donor = pop[rng.integers(0, SCREEN_POP, SCREEN_POP)]
        mutant = pop + 0.6 * (donor - pop) + rng.normal(0, 0.4, pop.shape)
        mutant = _clip_population(mutant)
        cross = rng.random(pop.shape) < 0.6
        trial = np.where(cross, mutant, pop)
        trial = _clip_population(trial)
        trial_f = _isp_of_pop(trial, mol_row, template, scaler, model_isp, device)
        improve = trial_f >= fitness
        pop = np.where(improve[:, None], trial, pop)
        fitness = np.where(improve, trial_f, fitness)
        i = int(np.argmax(fitness))
        if fitness[i] > best_f:
            best_f = float(fitness[i])
            best = pop[i].copy()
    return best, best_f


def unique_molecules(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["smiles", "nC", "nH", "nN", "nO", "molecular weight"] + FEATURE_COLUMNS
    cols = list(dict.fromkeys(cols))
    return df[cols].drop_duplicates("smiles").reset_index(drop=True)


def screen_all(df, scaler, models, device):
    rng = np.random.default_rng(SEED)
    mols = unique_molecules(df)
    if QUICK:
        mols = mols.head(QUICK_SCREEN_ECS)
    rows = []
    n = len(mols)
    for i, mol_row in mols.iterrows():
        template = mol_row
        mix, isp = optimize_ec(mol_row, template, scaler, models["isp"], device, rng)
        al, ems, htpb, ap = mix.tolist()
        X = build_feature_matrix([al], [ems], [htpb], [ap], mol_row, template)
        Xs = scaler.transform(X)
        tc = float(predict_numpy(models["c_t"], Xs, device)[0])
        cstar = float(predict_numpy(models["cstar"], Xs, device)[0])
        o_over_c = oc_ratio(al, ems, htpb, ap, mol_row)
        smiles = mol_row["smiles"]
        formula = _formula(smiles, mol_row["nC"], mol_row["nH"], mol_row["nN"], mol_row["nO"])
        rows.append({
            "rank": 0,
            "smiles": smiles,
            "formula": formula,
            "Al_wt%": al,
            "EC_wt%": ems,
            "HTPB_wt%": htpb,
            "AP_wt%": ap,
            "Isp_s": isp,
            "Tc_K": tc,
            "Cstar_m_s": cstar,
            "O_over_C": o_over_c,
            "nC": mol_row["nC"],
            "nH": mol_row["nH"],
            "nN": mol_row["nN"],
            "nO": mol_row["nO"],
            "MW": mol_row["molecular weight"],
        })
        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f"  screened {i + 1}/{n} ECs")

    result = pd.DataFrame(rows).sort_values("Isp_s", ascending=False).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    result["promising_Isp_gt_270"] = result["Isp_s"] > ISP_PROMISING
    top100 = result.head(TOP_N).copy()
    above = result[result["promising_Isp_gt_270"]].copy()
    final7 = (above.head(FINAL_N) if len(above) >= FINAL_N else result.head(FINAL_N)).copy()
    return result, top100, final7
