"""Ultimas dos palancas de solo-video, cada eje modelado por separado (sin joint/multi-task):

  A) Ensamble (soft-voting) de las top-3 configs por eje de nested_cv_final_<dx>.csv,
     cada una con su k ganador ya conocido, evaluado con el mismo esquema honesto
     5x5 RepeatedStratifiedKFold de la etapa de confirmacion.
  B) Grid de k mas amplio (hasta 16) para mutinfo_logreg/mutinfo_rf, por si k<=8
     fue demasiado agresivo.

Uso:  python experimentos_sin_pca/final_checks.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from labels import cargar_labels
from feature_selectors import construir_config, RANDOM_STATE, base_steps, clasificador

from sklearn.feature_selection import SelectKBest
from feature_selectors import _mutual_info_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV, cross_validate

warnings.filterwarnings("ignore")
OUT_DIR = Path(__file__).resolve().parent / "resultados"


def cargar(dx):
    df = pd.read_csv(cfg.DIR_FEATURES / "video_features_v2.csv")
    lab = cargar_labels()[["codigo", dx]].rename(columns={dx: "label"})
    df = df.merge(lab, on="codigo", how="left").dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    feats = [c for c in df.columns if c not in ("participant", "codigo", "label")]
    return df[feats].values, df["label"].values


def evaluar_pipe(pipe, X, y, outer):
    res = cross_validate(pipe, X, y, cv=outer, scoring=["roc_auc", "f1_macro"], n_jobs=-1, error_score="raise")
    return float(np.nanmean(res["test_roc_auc"])), float(np.nanstd(res["test_roc_auc"]))


def ensamble_top3(dx, X, y):
    final_csv = OUT_DIR / f"nested_cv_final_{dx}.csv"
    top3 = pd.read_csv(final_csv).head(3)["config"].tolist()
    outer = RepeatedStratifiedKFold(n_splits=cfg.N_SPLITS, n_repeats=cfg.N_REPEATS, random_state=RANDOM_STATE)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # congela k de cada base-learner con un solo ajuste sobre todos los datos (igual que permutation_test.py)
    estimators = []
    for nombre in top3:
        pipe, grid = construir_config(nombre)
        if grid:
            gs = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc", n_jobs=-1)
            gs.fit(X, y)
            pipe.set_params(**gs.best_params_)
        estimators.append((nombre, pipe))

    voting = VotingClassifier(estimators=estimators, voting="soft")
    auc, std = evaluar_pipe(voting, X, y, outer)
    print(f"  [ensamble top3={top3}] AUC={auc:.3f}+/-{std:.3f}")
    return {"dx": dx, "metodo": "ensamble_top3", "base_learners": ",".join(top3), "auc_mean": round(auc, 4), "auc_std": round(std, 4)}


def wider_k(dx, X, y, nombre_base, k_grid):
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    modelo_nombre = "rf" if "rf" in nombre_base else "logreg"
    pipe = Pipeline(base_steps() + [("sel", SelectKBest(_mutual_info_score)), ("clf", clasificador(modelo_nombre))])
    gs_grid = {"sel__k": k_grid}
    est = GridSearchCV(pipe, gs_grid, cv=inner, scoring="roc_auc", n_jobs=1)
    auc, std = evaluar_pipe(est, X, y, outer)
    print(f"  [{nombre_base} k en {k_grid}] AUC={auc:.3f}+/-{std:.3f}")
    return {"dx": dx, "metodo": f"{nombre_base}_k_amplio", "base_learners": str(k_grid), "auc_mean": round(auc, 4), "auc_std": round(std, 4)}


def main():
    filas = []
    for dx, nombre_base in [("ansiedad", "mutinfo_logreg"), ("depresion", "mutinfo_rf")]:
        X, y = cargar(dx)
        print(f"\n=== {dx.upper()} ===")
        filas.append(ensamble_top3(dx, X, y))
        filas.append(wider_k(dx, X, y, nombre_base, [4, 6, 8, 12, 16, 24, 32]))
    tab = pd.DataFrame(filas)
    tab.to_csv(OUT_DIR / "final_checks.csv", index=False)
    print("\n", tab.to_string(index=False))


if __name__ == "__main__":
    main()
