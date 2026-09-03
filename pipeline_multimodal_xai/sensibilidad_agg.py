"""Analisis de sensibilidad: como afecta la AGREGACION de la probabilidad de
segmento (media / mediana / p80) al AUC de la fusion ganadora (`soft_vote`).

La eleccion `AUDIO_PROB_AGG = "mean"` de config_mm.py es un parametro libre; este
script chequea que no cambie la conclusion. Reutiliza el mismo modelo de segmento
por fold (solo cambia como se resumen sus probabilidades) -> barato.

Uso:  python pipeline_multimodal_xai/sensibilidad_agg.py
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV

import config_mm as cfg
from data_multimodal import cargar
from audio_branch import RamaAudio, construir_estimador_segmento, agregar_a_participante
from fusion import construir_video_config

warnings.filterwarnings("ignore")
RS = cfg.RANDOM_STATE
AGGS = ["mean", "median", "p80"]


def main():
    filas = []
    for dx in cfg.DXS:
        pj = cfg.DIR_RESULTADOS / f"best_config_{dx}.json"
        if not pj.exists():
            print(f"[{dx}] falta {pj.name}"); continue
        best = json.loads(pj.read_text(encoding="utf-8"))
        vid, aud = cargar(dx)
        y = vid["label"].values
        Xv = vid.drop(columns="label").values
        pids = vid.index.values

        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RS)
        auc = {a: [] for a in AGGS}
        for tr, te in cv.split(pids, y):
            pids_tr, pids_te = pids[tr], pids[te]
            # video (soft_vote no necesita OOF de train)
            vp, vg = construir_video_config(best["video_config"])
            vest = GridSearchCV(vp, vg, cv=StratifiedKFold(5, shuffle=True, random_state=RS),
                                scoring="roc_auc", n_jobs=-1) if vg else vp
            vest.fit(Xv[tr], y[tr])
            vbest = vest.best_estimator_ if hasattr(vest, "best_estimator_") else vest
            v_te = vbest.fit(Xv[tr], y[tr]).predict_proba(Xv[te])[:, 1]
            # audio: un solo modelo de segmento, prob por segmento en test
            seg_tr = aud[aud.pid.isin(pids_tr)]
            seg_te = aud[aud.pid.isin(pids_te)]
            fa = [c for c in aud.columns if c not in ("pid", cfg.AUDIO_SEG_COL, "label")]
            sm = construir_estimador_segmento(best["audio_config"]).fit(seg_tr[fa].values, seg_tr["label"].values)
            p_seg = sm.predict_proba(seg_te[fa].values)[:, 1]
            for a in AGGS:
                a_te = agregar_a_participante(seg_te["pid"].values, p_seg, a).reindex(pids_te).values
                fused = 0.5 * a_te + 0.5 * v_te
                auc[a].append(roc_auc_score(y[te], fused))

        for a in AGGS:
            filas.append({"dx": dx, "agg": a, "auc_soft_vote_mean": round(float(np.mean(auc[a])), 4),
                          "auc_std": round(float(np.std(auc[a])), 4), "n_folds": len(auc[a])})
        print(f"[{dx}] " + "  ".join(f"{a}={np.mean(auc[a]):.3f}" for a in AGGS))

    tab = pd.DataFrame(filas)
    tab.to_csv(cfg.DIR_RESULTADOS / "sensibilidad_agg.csv", index=False)
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
