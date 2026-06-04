"""Modelos interpretables + XAI sobre features de video (espejo del pipeline de audio).

- Evaluacion: RepeatedStratifiedKFold 5x5 (25 folds) a nivel de participante.
- Modelos: RandomForest, XGBoost, SVM-RBF.
- Metricas: accuracy, precision/recall/F1 macro, ROC-AUC.
- XAI: SHAP (TreeExplainer sobre RandomForest) -> importancia global + plot.

Uso:
    python src/train_xai.py --dx ansiedad
    python src/train_xai.py --dx depresion --modelo rf
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

warnings.filterwarnings("ignore")


def construir_modelo(nombre):
    if nombre == "rf":
        clf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                     random_state=cfg.RANDOM_STATE, n_jobs=-1)
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("clf", clf)])
    if nombre == "svm":
        clf = SVC(kernel="rbf", class_weight="balanced", probability=True,
                  random_state=cfg.RANDOM_STATE)
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()), ("clf", clf)])
    if nombre == "xgb":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                            subsample=0.9, colsample_bytree=0.8, eval_metric="logloss",
                            random_state=cfg.RANDOM_STATE, n_jobs=-1)
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("clf", clf)])
    raise ValueError(nombre)


# scorers nativos de sklearn (roc_auc usa correctamente predict_proba de la clase positiva)
SCORERS = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
    "roc_auc": "roc_auc",
}


def evaluar(dx, modelos):
    ds = cfg.DIR_FEATURES / f"dataset_{dx}.csv"
    if not ds.exists():
        print(f"[{dx}] falta {ds} (corre build_dataset.py primero)"); return None
    df = pd.read_csv(ds)
    feats = [c for c in df.columns if c not in ("participant", "label")]
    X = df[feats].values
    y = df["label"].values
    print(f"\n=== {dx.upper()} ===  n={len(y)}  clases={dict(zip(*np.unique(y, return_counts=True)))}  features={len(feats)}")
    cv = RepeatedStratifiedKFold(n_splits=cfg.N_SPLITS, n_repeats=cfg.N_REPEATS,
                                 random_state=cfg.RANDOM_STATE)
    filas = []
    for m in modelos:
        res = cross_validate(construir_modelo(m), X, y, cv=cv, scoring=SCORERS, n_jobs=-1)
        fila = {"dx": dx, "modelo": m}
        for k in SCORERS:
            v = res[f"test_{k}"]
            fila[f"{k}_mean"] = round(float(np.mean(v)), 4)
            fila[f"{k}_std"] = round(float(np.std(v)), 4)
        filas.append(fila)
        print(f"  {m:4s}  F1={fila['f1_macro_mean']:.3f}±{fila['f1_macro_std']:.3f}  "
              f"AUC={fila['roc_auc_mean']:.3f}  acc={fila['accuracy_mean']:.3f}")
    tabla = pd.DataFrame(filas)
    out = cfg.DIR_RESULTADOS / f"metricas_{dx}.csv"
    tabla.to_csv(out, index=False)
    print(f"  -> {out}")
    return df, feats, X, y


def shap_analisis(dx, df, feats, X, y):
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model = construir_modelo("rf").fit(X, y)
    Xi = model.named_steps["imp"].transform(X)
    explainer = shap.TreeExplainer(model.named_steps["clf"])
    sv = explainer.shap_values(Xi)
    # normalizar a forma (n_muestras, n_features) para la clase positiva (1):
    #  - shap antiguo: lista [clase0, clase1]
    #  - shap nuevo:   ndarray (n, features) o (n, features, n_clases)
    if isinstance(sv, list):
        vals = np.asarray(sv[1])
    else:
        vals = np.asarray(sv)
        if vals.ndim == 3:
            vals = vals[:, :, 1]
    imp = np.abs(vals).mean(axis=0)
    rank = pd.DataFrame({"feature": feats, "shap_importance": imp}).sort_values(
        "shap_importance", ascending=False)
    rank.to_csv(cfg.DIR_RESULTADOS / f"shap_importancia_{dx}.csv", index=False)
    print(f"\n  SHAP top-8 ({dx}):")
    for _, r in rank.head(8).iterrows():
        print(f"     {r['feature']:<22} {r['shap_importance']:.4f}")
    try:
        shap.summary_plot(vals, Xi, feature_names=feats, show=False, plot_type="bar")
        plt.tight_layout()
        plt.savefig(cfg.DIR_RESULTADOS / f"shap_{dx}.png", dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  -> resultados/shap_{dx}.png")
    except Exception as e:
        print(f"  (plot SHAP omitido: {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", choices=cfg.DXS + ["all"], default="all")
    ap.add_argument("--modelo", default="rf,xgb,svm", help="csv: rf,xgb,svm")
    ap.add_argument("--no-shap", action="store_true")
    args = ap.parse_args()
    modelos = [m.strip() for m in args.modelo.split(",") if m.strip()]
    dxs = cfg.DXS if args.dx == "all" else [args.dx]
    for dx in dxs:
        r = evaluar(dx, modelos)
        if r and not args.no_shap:
            df, feats, X, y = r
            shap_analisis(dx, df, feats, X, y)


if __name__ == "__main__":
    main()
