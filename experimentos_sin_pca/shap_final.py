"""SHAP sobre el pipeline GANADOR de cada eje, entrenado en TODOS los datos.

Usa la configuracion (selector + modelo + hiperparametros) que gano la
comparacion honesta de `nested_cv.py` y que paso el test de permutacion en
`permutation_test.py` (lee `resultados/best_params_<dx>.json`). Como ningun
selector usado es PCA, SHAP explica features ORIGINALES con nombre clinico
(ej. `brow_inner_up_mean`), no componentes abstractos -- ese es el punto de
todo este ejercicio.

Uso:  python experimentos_sin_pca/shap_final.py
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
from feature_selectors import construir_config

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent / "resultados"


def cargar(dx):
    df = pd.read_csv(cfg.DIR_FEATURES / "video_features_v2.csv")
    lab = cargar_labels()[["codigo", dx]].rename(columns={dx: "label"})
    df = df.merge(lab, on="codigo", how="left").dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    feats = [c for c in df.columns if c not in ("participant", "codigo", "label")]
    return feats, df[feats].values, df["label"].values


def _indices_paso(step):
    if hasattr(step, "get_support"):
        return step.get_support(indices=True)
    if hasattr(step, "support_"):
        return np.asarray(step.support_)
    return None  # paso que no reduce columnas (imputer, scaler, clasificador)


def nombres_finales(pipe, feats):
    idx = np.arange(len(feats))
    for nombre_paso, step in pipe.steps[:-1]:  # todo menos el clasificador
        sub_idx = _indices_paso(step)
        if sub_idx is not None:
            idx = idx[sub_idx]
    return [feats[i] for i in idx]


def _shap_values_clase_positiva(explainer, Xt):
    sv = explainer(Xt)
    vals = sv.values if hasattr(sv, "values") else sv
    if isinstance(vals, list):
        vals = np.asarray(vals[1])
    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    return vals


def main():
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for dx in cfg.DXS:
        params_json = OUT_DIR / f"best_params_{dx}.json"
        if not params_json.exists():
            print(f"[{dx}] falta {params_json}, corre permutation_test.py primero"); continue
        with open(params_json) as f:
            info = json.load(f)
        nombre, params = info["config"], info["params"]

        feats, X, y = cargar(dx)
        pipe, _ = construir_config(nombre)
        if params:
            pipe.set_params(**params)
        pipe.fit(X, y)

        feats_finales = nombres_finales(pipe, feats)
        Xt = pipe[:-1].transform(X)
        clf = pipe.named_steps["clf"]

        print(f"\n=== {dx.upper()} === config={nombre}  params={params}  features finales={len(feats_finales)}")
        print("  ", ", ".join(feats_finales))

        explainer = shap.Explainer(clf, Xt, feature_names=feats_finales)
        vals = _shap_values_clase_positiva(explainer, Xt)
        imp = np.abs(vals).mean(axis=0)
        rank = pd.DataFrame({"feature": feats_finales, "shap_importance": imp}).sort_values(
            "shap_importance", ascending=False)
        rank.to_csv(OUT_DIR / f"shap_importancia_{dx}.csv", index=False)
        print(f"  SHAP top-{min(8, len(rank))}:")
        for _, r in rank.head(8).iterrows():
            print(f"     {r['feature']:<28} {r['shap_importance']:.4f}")

        try:
            shap.summary_plot(vals, Xt, feature_names=feats_finales, show=False, plot_type="bar")
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"shap_{dx}.png", dpi=130, bbox_inches="tight")
            plt.close()
            print(f"  -> {OUT_DIR / f'shap_{dx}.png'}")
        except Exception as e:
            print(f"  (plot SHAP omitido: {e})")


if __name__ == "__main__":
    main()
