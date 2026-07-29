"""Diagnostico: la etiqueta 'DX ... IA' vs el puntaje PHQ-9/GAD-7 calculado.

Hallazgo motivador (ver conversacion / REPORTE.md Seccion 10): las columnas
PHQ 9_* y GAD7_* del Excel de campo NO estan vacias (contrario a lo que decia
el docstring de labels.py) -- tienen las respuestas Likert item por item. Al
sumarlas (escala estandar 0-3 por item) y cruzar contra la etiqueta categorica
'DX ... IA' que se usa como ground truth en TODO el pipeline (audio y video),
la correlacion punto-biserial es ~0 (dep: r=-0.04 p=0.75; ans: r=0.15 p=0.22).
Esto sugiere que el techo de AUC no es un problema del canal de video, sino de
un ground truth con poca relacion con el instrumento estandarizado.

Este script prueba la hipotesis empiricamente: reemplaza el target por un
binario derivado directamente del puntaje PHQ-9/GAD-7 (corte estandar >=5 =
'leve o peor') y corre el MISMO tipo de comparacion (nested-ish, screening 10
folds) que `nested_cv.py`, sobre las MISMAS 256 features de video, para ver si
el AUC sube cuando el target es mas consistente con el instrumento.

Uso:  python experimentos_sin_pca/check_label_quality.py
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
from feature_selectors import CONFIGS, construir_config, RANDOM_STATE

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV, cross_validate
from scipy.stats import pointbiserialr

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent / "resultados"
MAPA = {"Para nada.": 0, "Varios días.": 1, "Más de la mitad de los días.": 2,
        "Casi todos los días.": 3, "Nunca.": 0}
CORTE = 5  # PHQ-9 y GAD-7: 0-4 minimo, >=5 ya es "leve" o peor (cutoff clinico estandar)


def cargar_puntajes():
    df = pd.read_excel(cfg.RUTA_LABELS)
    col_cod = df.columns[0]
    phq_cols = [c for c in df.columns if str(c).startswith("PHQ")]
    gad_cols = [c for c in df.columns if str(c).startswith("GAD")]
    phq_num = df[phq_cols].apply(lambda c: c.map(MAPA))
    gad_num = df[gad_cols].apply(lambda c: c.map(MAPA))
    out = pd.DataFrame({
        "codigo": pd.to_numeric(df[col_cod], errors="coerce"),
        "phq_total": phq_num.sum(axis=1, skipna=False),
        "gad_total": gad_num.sum(axis=1, skipna=False),
    }).dropna(subset=["codigo"])
    out["codigo"] = out["codigo"].astype(int)
    return out.drop_duplicates(subset="codigo", keep="first")


def cargar_dx_label(dx):
    from labels import cargar_labels
    return cargar_labels()[["codigo", dx]].rename(columns={dx: "label_dx"})


def evaluar_config(nombre, X, y, outer_cv, inner_cv):
    pipe, grid = construir_config(nombre)
    estimator = GridSearchCV(pipe, grid, cv=inner_cv, scoring="roc_auc", n_jobs=1) if grid else pipe
    res = cross_validate(estimator, X, y, cv=outer_cv, scoring=["roc_auc", "f1_macro"], n_jobs=-1, error_score="raise")
    return float(np.nanmean(res["test_roc_auc"])), float(np.nanstd(res["test_roc_auc"])), float(np.nanmean(res["test_f1_macro"]))


def main():
    puntajes = cargar_puntajes()
    feats_df = pd.read_csv(cfg.DIR_FEATURES / "video_features_v2.csv")
    feats = [c for c in feats_df.columns if c not in ("participant", "codigo", "label")]

    for dx, col_score, corte_col in [("depresion", "phq_total", "phq_total"), ("ansiedad", "gad_total", "gad_total")]:
        dx_label = cargar_dx_label(dx)
        merged = feats_df.merge(puntajes, on="codigo", how="left").merge(dx_label, on="codigo", how="left")

        # -- 1) consistencia: DX label (existente) vs puntaje calculado, en el universo de 80 con video --
        chk = merged.dropna(subset=["label_dx", col_score])
        chk["label_dx"] = chk["label_dx"].astype(int)
        r, p = pointbiserialr(chk["label_dx"], chk[col_score])
        print(f"\n=== {dx.upper()} ===")
        print(f"  n con DX label + {col_score} completo: {len(chk)}")
        print(f"  correlacion punto-biserial DX-label vs {col_score}: r={r:.3f} (p={p:.4f})")
        pos_bajo = ((chk['label_dx'] == 1) & (chk[col_score] < CORTE)).sum()
        neg_alto = ((chk['label_dx'] == 0) & (chk[col_score] >= CORTE)).sum()
        print(f"  Etiquetados RIESGO con {col_score}<{CORTE} (minimo): {pos_bajo}/{(chk.label_dx==1).sum()}")
        print(f"  Etiquetados SIN RIESGO con {col_score}>={CORTE} (leve+): {neg_alto}/{(chk.label_dx==0).sum()}")

        # -- 2) re-etiquetar con el puntaje y re-correr el mismo tipo de comparacion --
        alt = merged.dropna(subset=[col_score]).copy()
        alt["label_alt"] = (alt[col_score] >= CORTE).astype(int)
        X = alt[feats].values
        y_alt = alt["label_alt"].values
        y_dx = alt["label_dx"] if "label_dx" in alt else None
        print(f"  n para re-entrenar con target alterno: {len(alt)}  clases={dict(zip(*np.unique(y_alt, return_counts=True)))}")

        outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        filas = []
        for nombre in CONFIGS:
            try:
                auc, std, f1 = evaluar_config(nombre, X, y_alt, outer, inner)
            except Exception as e:
                print(f"    [{nombre}] ERROR: {e}"); continue
            filas.append({"config": nombre, "auc_mean": round(auc, 4), "auc_std": round(std, 4), "f1_mean": round(f1, 4)})
        tab = pd.DataFrame(filas).sort_values("auc_mean", ascending=False).reset_index(drop=True)
        tab.to_csv(OUT_DIR / f"label_swap_{dx}.csv", index=False)
        print(f"  -- top 5 con target PHQ/GAD-binario (screening 10 folds) --")
        print(tab.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
