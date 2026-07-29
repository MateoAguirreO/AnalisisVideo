"""Test de permutacion sobre el pipeline GANADOR de cada eje (post nested_cv.py).

Pregunta que responde: si el AUC honesto (nested CV) fuera pura casualidad de
N=80 y clase positiva de 19-24, deberia verse un AUC parecido barajando las
etiquetas al azar. Vabalas et al. (2019) y Varoquaux (2018) documentan que con
muestras asi de chicas los intervalos de un solo CV pueden ser enganosos; el
permutation test es la forma estandar de chequearlo (Ojala & Garriga 2010).

Metodo (equivalente a `sklearn.model_selection.permutation_test_score`, que es
lo que se usa aqui): se fijan los hiperparametros ganadores (un solo
GridSearchCV sobre TODOS los datos reales, needed solo para congelar la
configuracion -- no para el AUC reportado, ese sale de la CV con permutacion),
y se evalua el pipeline CONGELADO con Stratified 5-fold sobre las etiquetas
reales (1 vez) y sobre N_PERM barajes de las etiquetas.

p-valor = (1 + #{AUC_permutado >= AUC_real}) / (N_PERM + 1)

Uso:  python experimentos_sin_pca/permutation_test.py
"""
import json
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

from sklearn.model_selection import StratifiedKFold, GridSearchCV, permutation_test_score

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent / "resultados"
N_PERMUTATIONS = 300


def cargar(dx):
    df = pd.read_csv(cfg.DIR_FEATURES / "video_features_v2.csv")
    lab = cargar_labels()[["codigo", dx]].rename(columns={dx: "label"})
    df = df.merge(lab, on="codigo", how="left").dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    feats = [c for c in df.columns if c not in ("participant", "codigo", "label")]
    return df[feats].values, df["label"].values


def congelar_hiperparametros(nombre, X, y):
    """Un unico ajuste sobre TODOS los datos solo para fijar (k, C, ...) ganadores.
    No es de aqui de donde sale el AUC reportado (ese es de permutation_test_score)."""
    pipe, grid = construir_config(nombre)
    if not grid:
        return pipe, {}
    gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                      scoring="roc_auc", n_jobs=-1)
    gs.fit(X, y)
    pipe.set_params(**gs.best_params_)
    return pipe, gs.best_params_


def main():
    filas = []
    for dx in cfg.DXS:
        final_csv = OUT_DIR / f"nested_cv_final_{dx}.csv"
        if not final_csv.exists():
            print(f"[{dx}] falta {final_csv}, corre nested_cv.py primero"); continue
        ganador = pd.read_csv(final_csv).iloc[0]
        nombre = ganador["config"]
        X, y = cargar(dx)
        print(f"\n=== {dx.upper()} === ganador nested CV: {nombre} (AUC honesto={ganador['auc_mean']:.3f})")

        pipe, params = congelar_hiperparametros(nombre, X, y)
        print(f"  hiperparametros congelados: {params if params else '(sin grid, ej. stability)'}")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        score, perm_scores, pvalue = permutation_test_score(
            pipe, X, y, scoring="roc_auc", cv=cv, n_permutations=N_PERMUTATIONS,
            n_jobs=-1, random_state=RANDOM_STATE)

        print(f"  AUC real (5-fold, hiperparams congelados) = {score:.3f}")
        print(f"  AUC nulo (permutado): media={perm_scores.mean():.3f}  p90={np.percentile(perm_scores,90):.3f}")
        print(f"  p-valor = {pvalue:.4f}  ({'SIGNIFICATIVO' if pvalue < 0.05 else 'no significativo'} a alpha=0.05)")

        filas.append({"dx": dx, "config": nombre, "auc_real_5fold": round(float(score), 4),
                      "auc_nulo_media": round(float(perm_scores.mean()), 4),
                      "auc_nulo_p90": round(float(np.percentile(perm_scores, 90)), 4),
                      "auc_nulo_p95": round(float(np.percentile(perm_scores, 95)), 4),
                      "p_valor": round(float(pvalue), 4), "n_permutaciones": N_PERMUTATIONS})

        with open(OUT_DIR / f"best_params_{dx}.json", "w") as f:
            json.dump({"config": nombre, "params": params}, f, indent=2)

    pd.DataFrame(filas).to_csv(OUT_DIR / "permutation_test.csv", index=False)


if __name__ == "__main__":
    main()
