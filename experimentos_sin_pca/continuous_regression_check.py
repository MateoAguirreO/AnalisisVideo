"""Ultimo chequeo: PHQ-9/GAD-7 como variable CONTINUA (regresion) en vez de riesgo binario.

Motivacion: quizas el corte binario (Sin Riesgo vs Riesgo, o el corte clinico
>=5 ya probado en check_label_quality.py) tira informacion que si existe en el
gradiente continuo de sintomas. Se prueba con RepeatedKFold + GroupKFold no
aplica aqui (1 fila por participante, no hay grupos), pero SI hay que evitar
que la seleccion de features vea el target completo -> mismo principio de
nested CV que el resto del proyecto, con Ridge/RF regressor + seleccion
embebida por informacion mutua (regresion) o correlacion.

Metrica: correlacion de Spearman entre prediccion OOF y puntaje real (mas
robusta que R2 con N tan chico), + su p-valor.

Uso:  python experimentos_sin_pca/continuous_regression_check.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import config as cfg

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_regression, f_regression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
OUT_DIR = Path(__file__).resolve().parent / "resultados"
MAPA = {"Para nada.": 0, "Varios días.": 1, "Más de la mitad de los días.": 2,
        "Casi todos los días.": 3, "Nunca.": 0}


def cargar_puntajes():
    df = pd.read_excel(cfg.RUTA_LABELS)
    col_cod = df.columns[0]
    phq_cols = [c for c in df.columns if str(c).startswith("PHQ")]
    gad_cols = [c for c in df.columns if str(c).startswith("GAD")]
    out = pd.DataFrame({
        "codigo": pd.to_numeric(df[col_cod], errors="coerce"),
        "phq_total": df[phq_cols].apply(lambda c: c.map(MAPA)).sum(axis=1, skipna=False),
        "gad_total": df[gad_cols].apply(lambda c: c.map(MAPA)).sum(axis=1, skipna=False),
    }).dropna(subset=["codigo"])
    out["codigo"] = out["codigo"].astype(int)
    return out.drop_duplicates(subset="codigo", keep="first")


def pipe_reg(nombre, k):
    sel = SelectKBest(mutual_info_regression, k=k) if nombre == "mutinfo" else SelectKBest(f_regression, k=k)
    modelo = RandomForestRegressor(n_estimators=400, max_depth=4, min_samples_leaf=3,
                                   random_state=RANDOM_STATE, n_jobs=1) if "rf" in nombre else Ridge(alpha=1.0)
    return Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("var", VarianceThreshold(0.0)),
        ("sc", StandardScaler()),
        ("sel", sel),
        ("clf", modelo),
    ])


def evaluar(dx, col_score):
    puntajes = cargar_puntajes()
    feats_df = pd.read_csv(cfg.DIR_FEATURES / "video_features_v2.csv")
    feats = [c for c in feats_df.columns if c not in ("participant", "codigo", "label")]
    merged = feats_df.merge(puntajes, on="codigo", how="left").dropna(subset=[col_score])
    X = merged[feats].values
    yscore = merged[col_score].values
    print(f"\n=== {dx.upper()} (regresion continua sobre {col_score}) === n={len(yscore)}  "
          f"rango={yscore.min():.0f}-{yscore.max():.0f}  media={yscore.mean():.1f}")

    outer = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    inner = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    filas = []
    for nombre in ["mutinfo_ridge", "mutinfo_rf", "freg_ridge", "freg_rf"]:
        pipe = pipe_reg(nombre, k=6)
        grid = {"sel__k": [4, 6, 8, 12]}
        gs = GridSearchCV(pipe, grid, cv=inner, scoring="r2", n_jobs=-1)
        pred = cross_val_predict(gs, X, yscore, cv=outer, n_jobs=1)
        rho, p = spearmanr(yscore, pred)
        filas.append({"dx": dx, "config": nombre, "spearman_rho": round(float(rho), 4), "p_valor": round(float(p), 4)})
        print(f"  {nombre:14s} Spearman rho={rho:.3f}  p={p:.4f}  {'*significativo*' if p < 0.05 else ''}")
    tab = pd.DataFrame(filas)
    tab.to_csv(OUT_DIR / f"regresion_continua_{dx}.csv", index=False)
    return tab


def main():
    evaluar("ansiedad", "gad_total")
    evaluar("depresion", "phq_total")


if __name__ == "__main__":
    main()
