"""Evaluacion honesta del enfoque multi-instancia (ventanas de 10s).

Diferencia clave frente a nested_cv.py: la unidad de muestreo para el CV es la
VENTANA, pero la unidad de PERSONA sigue siendo el participante (~20-40
ventanas correlacionadas por persona). Un K-Fold plano sobre ventanas dejaria
ventanas del MISMO participante en train y test -> fuga severa y AUC
falsamente alto. Aqui se usa `StratifiedGroupKFold` (group=codigo) tanto en el
loop externo como en el interno (seleccion de k), y la metrica que se reporta
es el AUC a nivel PARTICIPANTE tras promediar la probabilidad de sus ventanas
(out-of-fold), no el AUC a nivel ventana (que se reporta aparte, solo a modo
informativo -- typicamente mas alto y NO es el numero honesto).

Uso:  python experimentos_sin_pca/windowed_nested_cv.py
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
from feature_selectors import construir_config, RANDOM_STATE

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent / "resultados"
WIN_CSV = OUT_DIR / "window_features.csv"
CONFIGS_A_PROBAR = ["mutinfo_logreg", "mutinfo_rf", "anova_rf", "l1logreg", "anovacorr_rf", "rfimportance_rf"]
N_REPEATS = 5
N_SPLITS_OUTER = 5
N_SPLITS_INNER = 3


def cargar(dx):
    df = pd.read_csv(WIN_CSV)
    lab = cargar_labels()[["codigo", dx]].rename(columns={dx: "label"})
    df = df.merge(lab, on="codigo", how="left").dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    feats = [c for c in df.columns if c not in ("participant", "codigo", "window", "label")]
    return df, feats


def una_repeticion(nombre, X, y, codigos, seed):
    outer = StratifiedGroupKFold(n_splits=N_SPLITS_OUTER, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr_idx, te_idx in outer.split(X, y, groups=codigos):
        pipe, grid = construir_config(nombre)
        if grid:
            inner = StratifiedGroupKFold(n_splits=N_SPLITS_INNER, shuffle=True, random_state=seed)
            gs = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc", n_jobs=1)
            gs.fit(X[tr_idx], y[tr_idx], groups=codigos[tr_idx])
            modelo = gs.best_estimator_
        else:
            modelo = pipe.fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = modelo.predict_proba(X[te_idx])[:, 1]
    return oof


def evaluar(dx):
    df, feats = cargar(dx)
    X = df[feats].values
    y = df["label"].values
    codigos = df["codigo"].values
    n_part = df["codigo"].nunique()
    print(f"\n=== {dx.upper()} (multi-instancia) === {n_part} participantes, {len(y)} ventanas, "
          f"clases_ventana={dict(zip(*np.unique(y, return_counts=True)))}")

    filas = []
    for nombre in CONFIGS_A_PROBAR:
        auc_win_list, auc_part_list = [], []
        for rep in range(N_REPEATS):
            seed = RANDOM_STATE + rep
            oof = una_repeticion(nombre, X, y, codigos, seed)
            auc_win_list.append(roc_auc_score(y, oof))
            part = pd.DataFrame({"codigo": codigos, "y": y, "proba": oof}).groupby("codigo").agg(
                y=("y", "first"), proba=("proba", "mean"))
            auc_part_list.append(roc_auc_score(part["y"], part["proba"]))
        fila = {"config": nombre,
               "auc_ventana_mean": round(float(np.mean(auc_win_list)), 4),
               "auc_ventana_std": round(float(np.std(auc_win_list)), 4),
               "auc_participante_mean": round(float(np.mean(auc_part_list)), 4),
               "auc_participante_std": round(float(np.std(auc_part_list)), 4)}
        filas.append(fila)
        print(f"  {nombre:20s} AUC_ventana={fila['auc_ventana_mean']:.3f}+/-{fila['auc_ventana_std']:.3f}  "
              f"AUC_PARTICIPANTE(honesto)={fila['auc_participante_mean']:.3f}+/-{fila['auc_participante_std']:.3f}")

    tab = pd.DataFrame(filas).sort_values("auc_participante_mean", ascending=False).reset_index(drop=True)
    tab.to_csv(OUT_DIR / f"windowed_final_{dx}.csv", index=False)
    print(f"  >> GANADOR (por AUC participante honesto): {tab.iloc[0]['config']} = {tab.iloc[0]['auc_participante_mean']:.3f}")
    return tab


def main():
    for dx in cfg.DXS:
        evaluar(dx)


if __name__ == "__main__":
    main()
