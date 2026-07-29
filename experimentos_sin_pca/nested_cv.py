"""Comparacion HONESTA (nested CV) de estrategias de seleccion sin PCA.

Por que nested CV y no el barrido plano de v2 (src/train_v2.py): en v2, el
mejor (modelo, k) se elegia mirando el AUC medio de la MISMA cv que se
reportaba como resultado final -> optimismo tipo "winner's curse" (Vabalas
et al. 2019, PLOS ONE). Aqui la eleccion de hiperparametros (k, C, etc.) vive
en un loop INTERNO (GridSearchCV) y el AUC que se reporta es el del loop
EXTERNO, que nunca vio esos hiperparametros siendo elegidos.

Dos etapas (por costo computacional, practica estandar de screening -> confirm):
  1) screening: outer 5x2 (10 folds), inner 3-fold -> barre las 12 configs x 2 ejes.
  2) confirm:   outer 5x5 (25 folds, igual que v1/v2, comparable), inner 5-fold ->
     solo sobre las 3 mejores configs de la etapa 1 por eje, para una estimacion
     mas fina del AUC honesto.

Uso:  python experimentos_sin_pca/nested_cv.py
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
from feature_selectors import CONFIGS, construir_config, RANDOM_STATE

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV, cross_validate

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent / "resultados"
OUT_DIR.mkdir(exist_ok=True)
V2_CSV = cfg.DIR_FEATURES / "video_features_v2.csv"
SCORING = ["roc_auc", "f1_macro", "accuracy"]


def cargar(dx):
    df = pd.read_csv(V2_CSV)
    lab = cargar_labels()[["codigo", dx]].rename(columns={dx: "label"})
    df = df.merge(lab, on="codigo", how="left").dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    feats = [c for c in df.columns if c not in ("participant", "codigo", "label")]
    return df, feats, df[feats].values, df["label"].values


def evaluar_config(nombre, X, y, outer_cv, inner_cv):
    pipe, grid = construir_config(nombre)
    estimator = GridSearchCV(pipe, grid, cv=inner_cv, scoring="roc_auc", n_jobs=1) if grid else pipe
    res = cross_validate(estimator, X, y, cv=outer_cv, scoring=SCORING, n_jobs=-1, error_score="raise")
    return {
        "config": nombre,
        "auc_mean": round(float(np.nanmean(res["test_roc_auc"])), 4),
        "auc_std": round(float(np.nanstd(res["test_roc_auc"])), 4),
        "f1_mean": round(float(np.nanmean(res["test_f1_macro"])), 4),
        "acc_mean": round(float(np.nanmean(res["test_accuracy"])), 4),
        "n_outer_folds": len(res["test_roc_auc"]),
    }


def screening(dx, X, y):
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    filas = []
    for nombre in CONFIGS:
        r = evaluar_config(nombre, X, y, outer, inner)
        filas.append(r)
        print(f"  [screen] {nombre:22s} AUC={r['auc_mean']:.3f}+/-{r['auc_std']:.3f}  F1={r['f1_mean']:.3f}")
    tab = pd.DataFrame(filas).sort_values("auc_mean", ascending=False).reset_index(drop=True)
    tab.to_csv(OUT_DIR / f"screening_{dx}.csv", index=False)
    return tab


def confirmacion(dx, X, y, top_configs):
    outer = RepeatedStratifiedKFold(n_splits=cfg.N_SPLITS, n_repeats=cfg.N_REPEATS, random_state=RANDOM_STATE)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    filas = []
    for nombre in top_configs:
        r = evaluar_config(nombre, X, y, outer, inner)
        filas.append(r)
        print(f"  [confirm] {nombre:22s} AUC={r['auc_mean']:.3f}+/-{r['auc_std']:.3f}  F1={r['f1_mean']:.3f}  (outer {r['n_outer_folds']} folds)")
    tab = pd.DataFrame(filas).sort_values("auc_mean", ascending=False).reset_index(drop=True)
    tab.to_csv(OUT_DIR / f"nested_cv_final_{dx}.csv", index=False)
    return tab


def main():
    for dx in cfg.DXS:
        df, feats, X, y = cargar(dx)
        print(f"\n=== {dx.upper()} ===  n={len(y)}  clases={dict(zip(*np.unique(y, return_counts=True)))}  features={len(feats)}")
        print(" -- etapa 1: screening (10 folds externos x 12 configs) --")
        tab_screen = screening(dx, X, y)
        top3 = tab_screen.head(3)["config"].tolist()
        print(f" -- etapa 2: confirmacion (25 folds externos) sobre el top-3: {top3} --")
        tab_final = confirmacion(dx, X, y, top3)
        ganador = tab_final.iloc[0]
        print(f"  >> GANADOR {dx}: {ganador['config']}  AUC honesto={ganador['auc_mean']:.3f}+/-{ganador['auc_std']:.3f}")


if __name__ == "__main__":
    main()
