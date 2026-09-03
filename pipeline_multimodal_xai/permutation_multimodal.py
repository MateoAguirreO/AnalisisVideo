"""Test de permutacion sobre el pipeline multimodal GANADOR de cada eje.

Pregunta: si el AUC honesto de la fusion fuera casualidad de N=79 y ~20 positivos,
barajando las etiquetas se veria un AUC parecido. Metodo estandar (Ojala & Garriga
2010), mismo criterio que experimentos_sin_pca/permutation_test.py.

Se congela la ARQUITECTURA ganadora (config de audio + config de video + variante de
meta-fusion, de best_config_<dx>.json) en un estimador a nivel participante y se
evalua con StratifiedKFold(5) sobre las etiquetas reales (1x) y sobre N_PERM barajes.

p = (1 + #{AUC_perm >= AUC_real}) / (N_PERM + 1)

Nota de costo: la rama de audio se reentrena en cada barajada; el OOF interno del
meta-modelo usa 3 folds (en vez de 5) para acotar el tiempo. Mismo trato para las
etiquetas reales y las permutadas.

Uso:  python pipeline_multimodal_xai/permutation_multimodal.py
"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold,
                                     cross_val_predict, permutation_test_score)

import config_mm as cfg
from data_multimodal import cargar
from audio_branch import construir_estimador_segmento, agregar_a_participante
from fusion import construir_video_config, MetaFusion

warnings.filterwarnings("ignore")
RS = cfg.RANDOM_STATE
N_PERM = 200


class MultimodalEstimator(ClassifierMixin, BaseEstimator):
    """Estimador a nivel participante: X = columna de pids.

    Los DataFrames vid_df/aud_df viajan como parametros del estimador para que
    `permutation_test_score` con n_jobs>1 (loky) los tenga en cada worker.
    """

    _estimator_type = "classifier"

    def __init__(self, vid_df=None, aud_df=None, audio_config="anova_xgb",
                 video_config="mutinfo_rf", meta="stack_logreg", n_inner=3):
        self.vid_df = vid_df
        self.aud_df = aud_df
        self.audio_config = audio_config
        self.video_config = video_config
        self.meta = meta
        self.n_inner = n_inner

    def _seg(self, pids):
        d = self.aud_df
        return d[d.pid.isin(pids)]

    def _vid(self, pids):
        v = self.vid_df.loc[list(pids)]
        return v.drop(columns="label").values

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        pids = np.asarray(X).ravel()
        y = np.asarray(y, int)
        entrena_meta = self.meta not in ("soft_vote", "hard_vote")

        # rama video
        vp, vg = construir_video_config(self.video_config)
        from sklearn.model_selection import GridSearchCV
        self.vid_ = (GridSearchCV(vp, vg, cv=3, scoring="roc_auc", n_jobs=1)
                     .fit(self._vid(pids), y).best_estimator_ if vg else vp.fit(self._vid(pids), y))

        # rama audio
        seg = self._seg(pids)
        feats = [c for c in seg.columns if c not in ("pid", cfg.AUDIO_SEG_COL, "label")]
        Xs = seg[feats].values
        ys = seg["pid"].map(dict(zip(pids, y))).values
        gs = seg["pid"].values
        self.aud_ = construir_estimador_segmento(self.audio_config).fit(Xs, ys)

        if entrena_meta:
            # el meta-modelo se entrena con scores OOF (sin fuga); soft/hard voting no
            # entrenan -> se salta este bloque (3-4x mas rapido en la permutacion).
            v_oof = cross_val_predict(self.vid_, self._vid(pids), y,
                                      cv=StratifiedKFold(self.n_inner, shuffle=True, random_state=RS),
                                      method="predict_proba")[:, 1]
            ns = min(self.n_inner, int(np.min(np.bincount(y))))
            p_seg = cross_val_predict(construir_estimador_segmento(self.audio_config), Xs, ys,
                                      cv=StratifiedGroupKFold(max(ns, 2), shuffle=True, random_state=RS),
                                      groups=gs, method="predict_proba", n_jobs=1)[:, 1]
            a_oof = agregar_a_participante(gs, p_seg, cfg.AUDIO_PROB_AGG).reindex(pids).values
            self.meta_ = MetaFusion(self.meta).fit(np.column_stack([a_oof, v_oof]), y)
        else:
            self.meta_ = MetaFusion(self.meta).fit(np.zeros((len(pids), 2)), y)
        return self

    def predict_proba(self, X):
        pids = np.asarray(X).ravel()
        seg = self._seg(pids)
        feats = [c for c in seg.columns if c not in ("pid", cfg.AUDIO_SEG_COL, "label")]
        p_seg = self.aud_.predict_proba(seg[feats].values)[:, 1]
        a = agregar_a_participante(seg["pid"].values, p_seg, cfg.AUDIO_PROB_AGG).reindex(pids).values
        v = self.vid_.predict_proba(self._vid(pids))[:, 1]
        p = self.meta_.predict_proba(np.column_stack([a, v]))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", choices=cfg.DXS + ["all"], default="all")
    ap.add_argument("--nperm", type=int, default=N_PERM)
    args = ap.parse_args()
    nperm = args.nperm
    dxs = cfg.DXS if args.dx == "all" else [args.dx]

    out_csv = cfg.DIR_RESULTADOS / "permutation_multimodal.csv"
    filas = []
    if out_csv.exists() and dxs != cfg.DXS:
        filas = [r for r in pd.read_csv(out_csv).to_dict("records") if r["dx"] not in dxs]
    for dx in dxs:
        pj = cfg.DIR_RESULTADOS / f"best_config_{dx}.json"
        if not pj.exists():
            print(f"[{dx}] falta {pj.name}; corre eval_multimodal.py primero"); continue
        best = json.loads(pj.read_text(encoding="utf-8"))
        vid, aud = cargar(dx)

        meta = best["ganador"] if best["ganador"] != "early_fusion" else "stack_logreg"
        est = MultimodalEstimator(vid_df=vid, aud_df=aud, audio_config=best["audio_config"],
                                  video_config=best["video_config"], meta=meta)
        pids = vid.index.values.reshape(-1, 1)
        y = vid["label"].values
        print(f"\n=== {dx.upper()} === arquitectura congelada: "
              f"audio={best['audio_config']} video={best['video_config']} meta={meta}")

        score, perm, p = permutation_test_score(
            est, pids, y, scoring="roc_auc",
            cv=StratifiedKFold(5, shuffle=True, random_state=RS),
            n_permutations=nperm, n_jobs=3, random_state=RS)
        print(f"  AUC real (5-fold) = {score:.3f}")
        print(f"  AUC nulo: media={perm.mean():.3f}  p95={np.percentile(perm,95):.3f}")
        print(f"  p-valor = {p:.4f}  ({'SIGNIFICATIVO' if p < 0.05 else 'no significativo'} a alpha=0.05)")
        filas.append({"dx": dx, "meta": meta, "auc_real_5fold": round(float(score), 4),
                      "auc_nulo_media": round(float(perm.mean()), 4),
                      "auc_nulo_p95": round(float(np.percentile(perm, 95)), 4),
                      "p_valor": round(float(p), 4), "n_permutaciones": nperm})
    if filas:
        orden = {d: i for i, d in enumerate(cfg.DXS)}
        filas = sorted(filas, key=lambda r: orden.get(r["dx"], 9))
        pd.DataFrame(filas).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
