"""Nucleo de features ESTABLE por eje (ansiedad/depresion) via stability selection.

Corre el bootstrap (Meinshausen & Buhlmann 2010) sobre TODOS los datos (80
participantes) -- esto es para REPORTAR el conjunto de features robusto que
alimentara el SHAP final, no para medir desempeno (eso ya lo hace, sin fuga,
`nested_cv.py` embebiendo el mismo selector dentro de cada fold). Es el mismo
principio que v2 usaba en `interpretar()` con f_classif sobre todos los datos:
un ranking/seleccion final para INTERPRETACION se calcula con todos los datos
disponibles; lo que nunca debe verse contaminado por todos los datos es la
ESTIMACION DE DESEMPENO (AUC), que vive aparte en nested_cv.py.

Uso:  python experimentos_sin_pca/stability_selection.py
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
from feature_selectors import StabilitySelector

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent / "resultados"
OUT_DIR.mkdir(exist_ok=True)
V2_CSV = cfg.DIR_FEATURES / "video_features_v2.csv"

N_BOOTSTRAP = 200
THRESHOLD = 0.6


def cargar(dx):
    df = pd.read_csv(V2_CSV)
    lab = cargar_labels()[["codigo", dx]].rename(columns={dx: "label"})
    df = df.merge(lab, on="codigo", how="left").dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    feats = [c for c in df.columns if c not in ("participant", "codigo", "label")]
    return feats, df[feats].values, df["label"].values


def main():
    for dx in cfg.DXS:
        feats, X, y = cargar(dx)
        Xi = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(X)
        var = VarianceThreshold(0.0)
        Xv = var.fit_transform(Xi)
        feats_v = [feats[i] for i in var.get_support(indices=True)]

        sel = StabilitySelector(n_bootstrap=N_BOOTSTRAP, threshold=THRESHOLD, C=0.5, random_state=cfg.RANDOM_STATE)
        sel.fit(Xv, y)

        rank = pd.DataFrame({"feature": feats_v, "freq_seleccion": sel.selection_freq_}).sort_values(
            "freq_seleccion", ascending=False)
        rank.to_csv(OUT_DIR / f"stability_{dx}.csv", index=False)

        nucleo = rank[rank["freq_seleccion"] >= THRESHOLD]
        print(f"\n=== {dx.upper()} === nucleo estable (freq>={THRESHOLD}, B={N_BOOTSTRAP}): {len(nucleo)} features")
        print(nucleo.to_string(index=False))
        if len(nucleo) < sel.min_features:
            print(f"  (ninguna/pocas alcanzaron el umbral; se retienen las top-{sel.min_features} por frecuencia)")
            print(rank.head(sel.min_features).to_string(index=False))


if __name__ == "__main__":
    main()
