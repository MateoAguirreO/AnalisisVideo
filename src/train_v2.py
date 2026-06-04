"""v2 - Entrenamiento con SELECCION DE FEATURES EMBEBIDA en la CV (sin fuga).

Pipeline por fold: imputacion -> VarianceThreshold -> escala -> SelectKBest(f_classif, k)
-> clasificador. Barrido de k y de modelos. CV RepeatedStratifiedKFold 5x5.
Reporta AUC/F1 y compara con la v1. Escribe resultados/metricas_v2_*.csv y SHAP v2.

No toca la v1.

Uso:  python src/train_v2.py --dx all
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from labels import cargar_labels

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

warnings.filterwarnings("ignore")

V2_CSV = cfg.DIR_FEATURES / "video_features_v2.csv"
KS = [8, 12, 16, 24]


def modelo(nombre):
    if nombre == "l1logreg":
        return LogisticRegression(penalty="l1", solver="liblinear", C=0.5,
                                  class_weight="balanced", max_iter=2000,
                                  random_state=cfg.RANDOM_STATE)
    if nombre == "rf":
        return RandomForestClassifier(n_estimators=400, max_depth=4, min_samples_leaf=3,
                                      class_weight="balanced", random_state=cfg.RANDOM_STATE,
                                      n_jobs=-1)
    if nombre == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=250, max_depth=2, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.8, reg_lambda=2.0,
                             eval_metric="logloss", random_state=cfg.RANDOM_STATE, n_jobs=-1)
    raise ValueError(nombre)


def pipe(nombre, k):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("var", VarianceThreshold(0.0)),
        ("sc", StandardScaler()),
        ("sel", SelectKBest(f_classif, k=k)),
        ("clf", modelo(nombre)),
    ])


def cargar(dx):
    df = pd.read_csv(V2_CSV)
    lab = cargar_labels()[["codigo", dx]].rename(columns={dx: "label"})
    df = df.merge(lab, on="codigo", how="left").dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    feats = [c for c in df.columns if c not in ("participant", "codigo", "label")]
    return df, feats, df[feats].values, df["label"].values


def v1_ref(dx):
    f = cfg.DIR_RESULTADOS / f"metricas_{dx}.csv"
    if f.exists():
        t = pd.read_csv(f)
        return float(t["roc_auc_mean"].max())
    return None


def evaluar(dx):
    df, feats, X, y = cargar(dx)
    n_feats = X.shape[1]
    print(f"\n=== {dx.upper()} (v2) ===  n={len(y)}  clases={dict(zip(*np.unique(y, return_counts=True)))}  features_totales={n_feats}")
    cv = RepeatedStratifiedKFold(n_splits=cfg.N_SPLITS, n_repeats=cfg.N_REPEATS,
                                 random_state=cfg.RANDOM_STATE)
    filas = []
    for m in ["l1logreg", "rf", "xgb"]:
        for k in KS:
            kk = min(k, n_feats)
            res = cross_validate(pipe(m, kk), X, y, cv=cv,
                                 scoring=["roc_auc", "f1_macro", "accuracy"], n_jobs=-1)
            filas.append({"modelo": m, "k": kk,
                          "auc": round(float(np.nanmean(res["test_roc_auc"])), 3),
                          "auc_std": round(float(np.nanstd(res["test_roc_auc"])), 3),
                          "f1": round(float(np.nanmean(res["test_f1_macro"])), 3),
                          "acc": round(float(np.nanmean(res["test_accuracy"])), 3)})
    tab = pd.DataFrame(filas).sort_values("auc", ascending=False).reset_index(drop=True)
    tab.to_csv(cfg.DIR_RESULTADOS / f"metricas_v2_{dx}.csv", index=False)
    ref = v1_ref(dx)
    best = tab.iloc[0]
    print(tab.head(6).to_string(index=False))
    print(f"  >> MEJOR v2: {best['modelo']} k={int(best['k'])}  AUC={best['auc']}  (v1 mejor AUC={ref})")
    return df, feats, X, y, int(best["k"])


def interpretar(dx, df, feats, X, y, k):
    """Top features por f_classif (univariado) sobre todos los datos -> interpretacion."""
    from sklearn.impute import SimpleImputer as Imp
    Xi = Imp(strategy="median", keep_empty_features=True).fit_transform(X)
    F, _ = f_classif(Xi, y)
    rank = pd.DataFrame({"feature": feats, "f_score": F}).sort_values("f_score", ascending=False)
    rank.to_csv(cfg.DIR_RESULTADOS / f"v2_top_features_{dx}.csv", index=False)
    print(f"  top features ({dx}):", ", ".join(rank.head(8)["feature"].tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", choices=cfg.DXS + ["all"], default="all")
    args = ap.parse_args()
    dxs = cfg.DXS if args.dx == "all" else [args.dx]
    for dx in dxs:
        df, feats, X, y, k = evaluar(dx)
        interpretar(dx, df, feats, X, y, k)


if __name__ == "__main__":
    main()
