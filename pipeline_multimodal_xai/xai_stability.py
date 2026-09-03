"""Estabilidad de las explicaciones frente al remuestreo (bootstrap, B=200).

Con N=79 una sola explicacion SHAP puede ser artefacto del split. Aqui se mide,
por eje:
  1) frecuencia con que cada feature de VIDEO entra en la seleccion (bootstrap de
     participantes, refit del pipeline de video ganador);
  2) idem para las features de AUDIO (bootstrap de participantes + sus segmentos,
     refit del modelo de segmento);
  3) estabilidad del peso de modalidad: bootstrap de los pares (score_audio,
     score_video) OOF + refit del meta-logistico y de la ponderacion AUC ->
     distribucion de |coef| por modalidad y de alpha.

Uso:  python pipeline_multimodal_xai/xai_stability.py   (requiere best_config_<dx>.json + xai_shap ya corrido para los OOF... se recalculan si faltan)
"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

import config_mm as cfg
from data_multimodal import cargar
from audio_branch import construir_estimador_segmento
from fusion import construir_video_config, MetaFusion
from xai_shap import nombres_finales, oof_scores_full

warnings.filterwarnings("ignore")
RS = cfg.RANDOM_STATE
B = 200


def _sel_video(vid, video_cfg, rng):
    feats = [c for c in vid.columns if c != "label"]
    X, y = vid[feats].values, vid["label"].values
    idx = resample(np.arange(len(y)), replace=True, stratify=y, random_state=rng.randint(1 << 30))
    if len(np.unique(y[idx])) < 2:
        return []
    pipe, grid = construir_video_config(video_cfg)
    if "sel__k" in grid:
        pipe.set_params(sel__k=6)
    pipe.fit(X[idx], y[idx])
    return nombres_finales(pipe, feats)


def _sel_audio(aud, rng):
    feats = [c for c in aud.columns if c not in ("pid", cfg.AUDIO_SEG_COL, "label")]
    pids = aud["pid"].unique()
    ypid = aud.groupby("pid")["label"].first()
    bpid = resample(pids, replace=True, stratify=ypid.values, random_state=rng.randint(1 << 30))
    parts = [aud[aud.pid == p] for p in bpid]
    d = pd.concat(parts, ignore_index=True)
    if d["label"].nunique() < 2:
        return []
    pipe = construir_estimador_segmento("anova_rf")
    pipe.fit(d[feats].values, d["label"].values)
    return nombres_finales(pipe, feats)


def _freq(counter, universo, B):
    return (pd.Series(counter).reindex(universo).fillna(0) / B).sort_values(ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", choices=cfg.DXS + ["all"], default="all")
    dxs = cfg.DXS if ap.parse_args().dx == "all" else [ap.parse_args().dx]
    for dx in dxs:
        pj = cfg.DIR_RESULTADOS / f"best_config_{dx}.json"
        if not pj.exists():
            print(f"[{dx}] falta {pj.name}; corre eval_multimodal.py primero"); continue
        best = json.loads(pj.read_text(encoding="utf-8"))
        vid, aud = cargar(dx)
        rng = np.random.RandomState(RS)
        print(f"\n=== {dx.upper()} === B={B}  (video={best['video_config']})")

        cv, ca = {}, {}
        for b in range(B):
            for f in _sel_video(vid, best["video_config"], rng):
                cv[f] = cv.get(f, 0) + 1
            for f in _sel_audio(aud, rng):
                ca[f] = ca.get(f, 0) + 1

        fv = _freq(cv, [c for c in vid.columns if c != "label"], B)
        fa = _freq(ca, [c for c in aud.columns if c not in ("pid", cfg.AUDIO_SEG_COL, "label")], B)
        pd.DataFrame({"feature": fv.index, "freq_seleccion": fv.values, "modalidad": "video"}).to_csv(
            cfg.DIR_RESULTADOS / f"estabilidad_video_{dx}.csv", index=False)
        pd.DataFrame({"feature": fa.index, "freq_seleccion": fa.values, "modalidad": "audio"}).to_csv(
            cfg.DIR_RESULTADOS / f"estabilidad_audio_{dx}.csv", index=False)
        print("  video - nucleo estable (freq>=0.5):", list(fv[fv >= 0.5].index) or "(ninguna)")
        print("  audio - nucleo estable (freq>=0.5):", list(fa[fa >= 0.5].index) or "(ninguna)")

        # --- estabilidad del peso de modalidad ---
        a, v, y = oof_scores_full(dx, vid, aud, best["audio_config"], best["video_config"])
        coefs, alphas = [], []
        for b in range(B):
            ii = resample(np.arange(len(y)), replace=True, stratify=y, random_state=rng.randint(1 << 30))
            if len(np.unique(y[ii])) < 2:
                continue
            S = np.column_stack([a[ii], v[ii]])
            lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(S, y[ii])
            c = np.abs(lr.coef_.ravel())
            coefs.append(c / (c.sum() + 1e-9))
            alphas.append(MetaFusion("pond_auc").fit(S, y[ii]).alpha_)
        coefs = np.array(coefs)
        res = pd.DataFrame({
            "metrica": ["coef_rel_audio", "coef_rel_video", "alpha_pond_auc (peso audio)"],
            "media": np.round([coefs[:, 0].mean(), coefs[:, 1].mean(), np.mean(alphas)], 3),
            "std": np.round([coefs[:, 0].std(), coefs[:, 1].std(), np.std(alphas)], 3),
            "p05": np.round([np.percentile(coefs[:, 0], 5), np.percentile(coefs[:, 1], 5),
                             np.percentile(alphas, 5)], 3),
            "p95": np.round([np.percentile(coefs[:, 0], 95), np.percentile(coefs[:, 1], 95),
                             np.percentile(alphas, 95)], 3),
        })
        res.to_csv(cfg.DIR_RESULTADOS / f"estabilidad_modalidad_{dx}.csv", index=False)
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
