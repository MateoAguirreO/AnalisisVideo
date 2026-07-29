"""Sustenta dos afirmaciones del informe mensual:

  (1) UNIVARIADO + FDR: ANOVA F-test (f_classif) de cada una de las 256 features
      contra la etiqueta, por eje, con correccion de comparaciones multiples
      Benjamini-Hochberg (FDR). Reporta cuantas features quedan significativas
      (q<0.05) -> respalda "ninguna feature es significativa de forma univariada
      tras correccion FDR" (o lo corrige si resulta falso).

  (2) ANOVA ENTRE MODELOS: one-way ANOVA sobre las distribuciones de AUC de los
      3 clasificadores (RF, XGB, L1-LogReg) a lo largo de los 25 folds del
      RepeatedStratifiedKFold 5x5, con la MISMA seleccion embebida (ANOVA k=8)
      para aislar el efecto del clasificador. Respalda "los tres modelos son
      estadisticamente equivalentes entre si" (o lo corrige). Se complementa con
      Kruskal-Wallis (no parametrico) por robustez con n=25.

Uso:  python experimentos_sin_pca/anova_fdr_check.py
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
from labels import cargar_labels

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from scipy.stats import f_oneway, kruskal

warnings.filterwarnings("ignore")
OUT_DIR = Path(__file__).resolve().parent / "resultados"


def fdr_bh(p, alpha=0.05):
    """Benjamini-Hochberg. Devuelve (rechazado_bool, q_valores) alineados al orden de entrada."""
    p = np.asarray(p, dtype=float)
    n = p.size
    orden = np.argsort(p)
    p_ord = p[orden]
    # q ajustado = min acumulado desde el mayor: p*n/rank
    q_ord = p_ord * n / (np.arange(n) + 1)
    q_ord = np.minimum.accumulate(q_ord[::-1])[::-1]
    q_ord = np.clip(q_ord, 0, 1)
    q = np.empty(n)
    q[orden] = q_ord
    return q < alpha, q


def cargar(dx):
    df = pd.read_csv(cfg.DIR_FEATURES / "video_features_v2.csv")
    lab = cargar_labels()[["codigo", dx]].rename(columns={dx: "label"})
    df = df.merge(lab, on="codigo", how="left").dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    feats = [c for c in df.columns if c not in ("participant", "codigo", "label")]
    return df, feats, df[feats].values, df["label"].values


def modelo(nombre):
    if nombre == "l1logreg":
        return LogisticRegression(penalty="l1", solver="liblinear", C=0.5,
                                  class_weight="balanced", max_iter=2000, random_state=cfg.RANDOM_STATE)
    if nombre == "rf":
        return RandomForestClassifier(n_estimators=400, max_depth=4, min_samples_leaf=3,
                                      class_weight="balanced", random_state=cfg.RANDOM_STATE, n_jobs=1)
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=250, max_depth=2, learning_rate=0.05, subsample=0.9,
                         colsample_bytree=0.8, reg_lambda=2.0, eval_metric="logloss",
                         random_state=cfg.RANDOM_STATE, n_jobs=1)


def pipe(nombre, k=8):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("var", VarianceThreshold(0.0)),
        ("sc", StandardScaler()),
        ("sel", SelectKBest(f_classif, k=k)),
        ("clf", modelo(nombre)),
    ])


def univariado_fdr(dx, feats, X, y):
    Xi = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(X)
    var = VarianceThreshold(0.0)
    Xv = var.fit_transform(Xi)
    feats_v = [feats[i] for i in var.get_support(indices=True)]
    F, p = f_classif(Xv, y)
    rej, q = fdr_bh(p, alpha=0.05)
    tab = pd.DataFrame({"feature": feats_v, "F": F, "p_valor": p, "q_valor_FDR": q,
                        "signif_FDR": rej}).sort_values("p_valor")
    tab.to_csv(OUT_DIR / f"anova_univariado_{dx}.csv", index=False)
    n_sig_raw = int((p < 0.05).sum())
    n_sig_fdr = int(rej.sum())
    print(f"\n=== {dx.upper()} — ANOVA univariado F-test (n={len(y)}, {len(feats_v)} features) ===")
    print(f"  features con p<0.05 SIN corregir: {n_sig_raw}/{len(feats_v)}")
    print(f"  features significativas tras FDR (Benjamini-Hochberg, q<0.05): {n_sig_fdr}/{len(feats_v)}")
    print(f"  menor p-valor: {p.min():.4f}  ->  menor q-valor FDR: {q.min():.4f}")
    print(f"  top-5 por p-valor:")
    print(tab.head(5)[["feature", "F", "p_valor", "q_valor_FDR"]].to_string(index=False))
    return n_sig_fdr, float(q.min())


def anova_entre_modelos(dx, X, y):
    cv = RepeatedStratifiedKFold(n_splits=cfg.N_SPLITS, n_repeats=cfg.N_REPEATS, random_state=cfg.RANDOM_STATE)
    aucs = {}
    for m in ["rf", "xgb", "l1logreg"]:
        aucs[m] = cross_val_score(pipe(m, k=8), X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    F, p_anova = f_oneway(aucs["rf"], aucs["xgb"], aucs["l1logreg"])
    H, p_kw = kruskal(aucs["rf"], aucs["xgb"], aucs["l1logreg"])
    print(f"\n=== {dx.upper()} — ANOVA entre modelos (25 folds, seleccion ANOVA k=8 fija) ===")
    for m in ["rf", "xgb", "l1logreg"]:
        print(f"  {m:9s} AUC medio={aucs[m].mean():.3f} +/- {aucs[m].std():.3f}")
    print(f"  one-way ANOVA:     F={F:.3f}  p={p_anova:.4f}  -> {'equivalentes' if p_anova>=0.05 else 'DIFIEREN'}")
    print(f"  Kruskal-Wallis:    H={H:.3f}  p={p_kw:.4f}  -> {'equivalentes' if p_kw>=0.05 else 'DIFIEREN'}")
    return p_anova, p_kw


def main():
    resumen = []
    for dx in cfg.DXS:
        df, feats, X, y = cargar(dx)
        n_sig_fdr, qmin = univariado_fdr(dx, feats, X, y)
        p_anova, p_kw = anova_entre_modelos(dx, X, y)
        resumen.append({"dx": dx, "features_signif_FDR": n_sig_fdr, "q_min_FDR": round(qmin, 4),
                        "anova_modelos_p": round(float(p_anova), 4), "kruskal_modelos_p": round(float(p_kw), 4)})
    pd.DataFrame(resumen).to_csv(OUT_DIR / "anova_fdr_resumen.csv", index=False)
    print("\n\n===== RESUMEN PARA EL INFORME =====")
    print(pd.DataFrame(resumen).to_string(index=False))


if __name__ == "__main__":
    main()
