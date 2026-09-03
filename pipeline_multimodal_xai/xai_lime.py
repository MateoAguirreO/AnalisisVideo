"""LIME local + chequeo de consistencia SHAP/LIME sobre la fusion multimodal.

Sobre la matriz de EARLY FUSION (1 fila / participante, features aud__* y vid__*):
  1) elige 3 participantes ilustrativos por sus predicciones OOF -> un verdadero
     positivo, un falso positivo y un verdadero negativo;
  2) LIME explica cada uno: una sola lista rankeada que mezcla voz y rostro
     ("por que este paciente");
  3) para los mismos 3 casos calcula SHAP local (KernelExplainer sobre la misma
     funcion predict_proba y el mismo espacio de features) y reporta la
     concordancia con LIME (Jaccard del top-10 y Spearman del ranking).

Uso:  python pipeline_multimodal_xai/xai_lime.py   (requiere best_config_<dx>.json)
"""
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict

import config_mm as cfg
from data_multimodal import cargar
from fusion import matriz_early_fusion, construir_early_config

warnings.filterwarnings("ignore")
RS = cfg.RANDOM_STATE
K_TOP = 10


def _pipe_early(Me, early_cfg):
    feats = [c for c in Me.columns if c != "label"]
    X, y = Me[feats].values, Me["label"].values
    pipe, grid = construir_early_config(early_cfg)
    if grid:
        pipe = GridSearchCV(pipe, grid, cv=5, scoring="roc_auc", n_jobs=-1).fit(X, y).best_estimator_
    pipe.fit(X, y)
    return pipe, X, y, feats


def _casos(pipe, Me, early_cfg, X, y):
    """OOF predict_proba sobre la matriz early -> indices de un TP, un FP, un TN."""
    pipe0, grid = construir_early_config(early_cfg)
    est = GridSearchCV(pipe0, grid, cv=3, scoring="roc_auc", n_jobs=-1) if grid else pipe0
    oof = cross_val_predict(est, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=RS),
                            method="predict_proba", n_jobs=-1)[:, 1]
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    tp = pos[np.argmax(oof[pos])]
    fp = neg[np.argmax(oof[neg])]
    tn = neg[np.argmin(oof[neg])]
    return {"TP": int(tp), "FP": int(fp), "TN": int(tn)}, oof


def _barh(pares, path, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pares = pares[::-1]
    labels = [p[0] for p in pares]
    vals = [p[1] for p in pares]
    colores = ["#c0392b" if v > 0 else "#2c7fb8" for v in vals]
    plt.figure(figsize=(7, 4))
    plt.barh(range(len(vals)), vals, color=colores)
    plt.yticks(range(len(vals)), labels, fontsize=8)
    plt.axvline(0, color="k", lw=0.8)
    plt.title(titulo, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()


def main():
    from lime.lime_tabular import LimeTabularExplainer
    import shap

    filas_cons = []
    for dx in cfg.DXS:
        pj = cfg.DIR_RESULTADOS / f"best_config_{dx}.json"
        if not pj.exists():
            print(f"[{dx}] falta {pj.name}; corre eval_multimodal.py primero"); continue
        best = json.loads(pj.read_text(encoding="utf-8"))
        vid, aud = cargar(dx)
        Me = matriz_early_fusion(vid, aud)
        pipe, X, y, feats = _pipe_early(Me, best["early_config"])
        casos, oof = _casos(pipe, Me, best["early_config"], X, y)
        pids = list(vid.index)
        print(f"\n=== {dx.upper()} === early_config={best['early_config']}  "
              f"casos (pid): TP={pids[casos['TP']]} FP={pids[casos['FP']]} TN={pids[casos['TN']]}")

        lime_expl = LimeTabularExplainer(
            X, feature_names=feats, class_names=["sin riesgo", "riesgo"],
            discretize_continuous=True, mode="classification", random_state=RS)
        bg = shap.sample(X, 40, random_state=RS)
        shap_expl = shap.KernelExplainer(lambda z: pipe.predict_proba(z)[:, 1], bg)

        for nombre, idx in casos.items():
            exp = lime_expl.explain_instance(X[idx], pipe.predict_proba, num_features=K_TOP,
                                             num_samples=3000)
            lime_pairs = exp.as_list()
            with open(cfg.DIR_RESULTADOS / f"lime_{dx}_{nombre}.txt", "w", encoding="utf-8") as f:
                f.write(f"{dx} — caso {nombre} (pid {pids[idx]}), y={y[idx]}, "
                        f"p_oof={oof[idx]:.3f}, p_full={pipe.predict_proba(X[idx:idx+1])[0,1]:.3f}\n\n")
                for feat, w in lime_pairs:
                    mod = "voz  " if "aud__" in feat else "rostro"
                    f.write(f"  [{mod}] {feat:<45} {w:+.4f}\n")
            _barh([(p[0][:38], p[1]) for p in lime_pairs],
                  cfg.DIR_RESULTADOS / f"lime_{dx}_{nombre}.png",
                  f"{dx} — LIME caso {nombre} (pid {pids[idx]}, y={y[idx]})")

            sv = np.asarray(shap_expl.shap_values(X[idx:idx+1], nsamples=200)).ravel()
            shap_rank = pd.Series(np.abs(sv), index=feats).sort_values(ascending=False)
            lime_names = [f for f, _ in _clean_lime_names(lime_pairs, feats)]
            shap_names = list(shap_rank.head(K_TOP).index)
            jac = len(set(lime_names) & set(shap_names)) / len(set(lime_names) | set(shap_names))
            comunes = [f for f in feats if f in set(lime_names) | set(shap_names)]
            lime_full = dict(_clean_lime_names(lime_pairs, feats))
            rho, _ = spearmanr([abs(lime_full.get(f, 0)) for f in comunes],
                               [abs(shap_rank.get(f, 0)) for f in comunes])
            filas_cons.append({"dx": dx, "caso": nombre, "pid": pids[idx],
                               "jaccard_top10": round(jac, 3),
                               "spearman_rank": round(float(rho), 3)})
            print(f"  {nombre}: Jaccard(top10)={jac:.2f}  Spearman={rho:+.2f}")

    if filas_cons:
        pd.DataFrame(filas_cons).to_csv(
            cfg.DIR_RESULTADOS / "consistencia_shap_lime.csv", index=False)


def _clean_lime_names(lime_pairs, feats):
    """LIME devuelve 'feat <= 0.3' etc.; recupera el nombre de feature crudo."""
    out = []
    fset = set(feats)
    for cond, w in lime_pairs:
        cand = [f for f in fset if f in cond]
        if cand:
            out.append((max(cand, key=len), w))
    return out


if __name__ == "__main__":
    main()
