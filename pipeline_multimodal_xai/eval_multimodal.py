"""Evaluacion honesta del pipeline multimodal: baselines + nested CV + late/early fusion.

Dos etapas (espejo de experimentos_sin_pca/nested_cv.py):
  1) SCREENING  (RepeatedStratifiedKFold 5x2 = 10 folds): elige la mejor config de
     rama-audio, rama-video y early-fusion por su AUC de participante (solo test scores).
  2) CONFIRMACION (5x5 = 25 folds): con esas configs fijas, evalua todos los baselines
     y todas las variantes de fusion. El AUC reportado es el del loop externo.

Sin fuga: splits por participante en todos los loops; la rama de audio agrega OOF con
StratifiedGroupKFold (grupo=participante); seleccion de features e imputacion dentro de
cada pipeline; los scores base que alimentan al meta-modelo son out-of-fold.

Uso:  python pipeline_multimodal_xai/eval_multimodal.py [--dx ansiedad|depresion|all]
"""
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     GridSearchCV, cross_val_predict)

import config_mm as cfg
from data_multimodal import cargar
from audio_branch import RamaAudio, AUDIO_CONFIGS
from fusion import (construir_video_config, VIDEO_CONFIGS, matriz_early_fusion,
                    construir_early_config, EARLY_CONFIGS, MetaFusion, META_VARIANTES)

warnings.filterwarnings("ignore")
RS = cfg.RANDOM_STATE


# ---------------------------------------------------------------------------
# ramas base -> scores de participante (OOF para train, ajuste normal para test)
# ---------------------------------------------------------------------------
def video_scores(Xv, y, tr, te, config, inner=5):
    pipe, grid = construir_video_config(config)
    est = GridSearchCV(pipe, grid, cv=StratifiedKFold(inner, shuffle=True, random_state=RS),
                       scoring="roc_auc", n_jobs=-1) if grid else pipe
    est.fit(Xv[tr], y[tr])
    best = est.best_estimator_ if hasattr(est, "best_estimator_") else est
    oof = cross_val_predict(best, Xv[tr], y[tr],
                            cv=StratifiedKFold(inner, shuffle=True, random_state=RS),
                            method="predict_proba", n_jobs=-1)[:, 1]
    test = best.fit(Xv[tr], y[tr]).predict_proba(Xv[te])[:, 1]
    return oof, test


def audio_scores(aud, pids, y, tr, te, config, agg=cfg.AUDIO_PROB_AGG):
    pids_tr, pids_te = pids[tr], pids[te]
    ra = RamaAudio(config=config, agg=agg).fit(aud[aud.pid.isin(pids_tr)])
    s_oof = ra.scores_oof_participante(aud[aud.pid.isin(pids_tr)]).reindex(pids_tr).values
    s_te = ra.scores_test_participante(aud[aud.pid.isin(pids_te)]).reindex(pids_te).values
    return s_oof, s_te


def early_scores(Me, y, tr, te, config, inner=5):
    Xe = Me.drop(columns="label").values
    pipe, grid = construir_early_config(config)
    est = GridSearchCV(pipe, grid, cv=StratifiedKFold(inner, shuffle=True, random_state=RS),
                       scoring="roc_auc", n_jobs=-1) if grid else pipe
    est.fit(Xe[tr], y[tr])
    best = est.best_estimator_ if hasattr(est, "best_estimator_") else est
    return best.fit(Xe[tr], y[tr]).predict_proba(Xe[te])[:, 1]


# ---------------------------------------------------------------------------
def _metricas(y_true, score):
    pred = (score >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y_true, score) if len(np.unique(y_true)) == 2 else np.nan,
        "f1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "acc": accuracy_score(y_true, pred),
        "bacc": balanced_accuracy_score(y_true, pred),
    }


def _agg_folds(filas, etiqueta):
    d = pd.DataFrame(filas)
    return {
        "metodo": etiqueta,
        "auc_mean": round(float(np.nanmean(d["auc"])), 4),
        "auc_std": round(float(np.nanstd(d["auc"])), 4),
        "f1_mean": round(float(np.nanmean(d["f1_macro"])), 4),
        "bacc_mean": round(float(np.nanmean(d["bacc"])), 4),
        "n_folds": len(d),
    }


# ---------------------------------------------------------------------------
def screening(dx, vid, aud, Me):
    y = vid["label"].values
    Xv = vid.drop(columns="label").values
    pids = vid.index.values
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RS)
    splits = list(cv.split(pids, y))

    def _eval(kind, config):
        filas = []
        for tr, te in splits:
            if kind == "audio":
                _, s = audio_scores(aud, pids, y, tr, te, config)
            elif kind == "video":
                _, s = video_scores(Xv, y, tr, te, config)
            else:
                s = early_scores(Me, y, tr, te, config)
            filas.append(_metricas(y[te], s))
        return _agg_folds(filas, f"{kind}:{config}")

    filas = []
    for c in AUDIO_CONFIGS:
        filas.append(_eval("audio", c)); print("  ", filas[-1])
    for c in VIDEO_CONFIGS:
        filas.append(_eval("video", c)); print("  ", filas[-1])
    for c in EARLY_CONFIGS:
        filas.append(_eval("early", c)); print("  ", filas[-1])
    tab = pd.DataFrame(filas).sort_values("auc_mean", ascending=False)
    tab.to_csv(cfg.DIR_RESULTADOS / f"screening_{dx}.csv", index=False)

    def _best(pref):
        sub = tab[tab["metodo"].str.startswith(pref)]
        return sub.iloc[0]["metodo"].split(":", 1)[1]

    return _best("audio"), _best("video"), _best("early"), tab


def confirmacion(dx, vid, aud, Me, audio_cfg, video_cfg, early_cfg):
    y = vid["label"].values
    Xv = vid.drop(columns="label").values
    pids = vid.index.values
    prior = y.mean()
    cv = RepeatedStratifiedKFold(n_splits=cfg.N_SPLITS, n_repeats=cfg.N_REPEATS, random_state=RS)

    acc = {m: [] for m in ["mayoria", "solo_audio", "solo_video", "early_fusion", *META_VARIANTES]}
    alphas = {m: [] for m in ("pond_grid", "pond_auc")}

    for i, (tr, te) in enumerate(cv.split(pids, y), 1):
        a_oof, a_te = audio_scores(aud, pids, y, tr, te, audio_cfg)
        v_oof, v_te = video_scores(Xv, y, tr, te, video_cfg)
        e_te = early_scores(Me, y, tr, te, early_cfg)

        acc["mayoria"].append(_metricas(y[te], np.full(len(te), prior)))
        acc["solo_audio"].append(_metricas(y[te], a_te))
        acc["solo_video"].append(_metricas(y[te], v_te))
        acc["early_fusion"].append(_metricas(y[te], e_te))

        S_tr = np.column_stack([a_oof, v_oof])
        S_te = np.column_stack([a_te, v_te])
        for m in META_VARIANTES:
            mf = MetaFusion(m).fit(S_tr, y[tr])
            acc[m].append(_metricas(y[te], mf.predict_proba(S_te)))
            if m in alphas:
                alphas[m].append(mf.alpha_)
        if i % 5 == 0:
            print(f"    fold {i}/{cv.get_n_splits()}")

    filas = [_agg_folds(v, k) for k, v in acc.items()]
    tab = pd.DataFrame(filas).sort_values("auc_mean", ascending=False).reset_index(drop=True)
    for m, xs in alphas.items():
        tab.loc[tab["metodo"] == m, "alpha_mean"] = round(float(np.mean(xs)), 3)
    tab.to_csv(cfg.DIR_RESULTADOS / f"metricas_multimodal_{dx}.csv", index=False)

    fusion_rows = tab[tab["metodo"].isin(["early_fusion", *META_VARIANTES])]
    ganador = fusion_rows.iloc[0]
    mejor_sola = tab[tab["metodo"].isin(["solo_audio", "solo_video"])]["auc_mean"].max()
    best = {
        "dx": dx, "audio_config": audio_cfg, "video_config": video_cfg,
        "early_config": early_cfg, "ganador": ganador["metodo"],
        "auc_ganador": float(ganador["auc_mean"]), "auc_mejor_modalidad_sola": float(mejor_sola),
        "delta_fusion": round(float(ganador["auc_mean"] - mejor_sola), 4),
    }
    if ganador["metodo"] in alphas:
        best["alpha_mean"] = float(np.mean(alphas[ganador["metodo"]]))
    with open(cfg.DIR_RESULTADOS / f"best_config_{dx}.json", "w") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)
    return tab, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", choices=cfg.DXS + ["all"], default="all")
    args = ap.parse_args()
    dxs = cfg.DXS if args.dx == "all" else [args.dx]
    for dx in dxs:
        t0 = time.time()
        vid, aud = cargar(dx)
        Me = matriz_early_fusion(vid, aud)
        print(f"\n=== {dx.upper()} ===  n={len(vid)}  "
              f"pos={int(vid['label'].sum())}  seg={len(aud)}")
        print(" -- screening (10 folds) --")
        a_cfg, v_cfg, e_cfg, _ = screening(dx, vid, aud, Me)
        print(f"  best -> audio={a_cfg}  video={v_cfg}  early={e_cfg}")
        print(" -- confirmacion (25 folds) --")
        tab, best = confirmacion(dx, vid, aud, Me, a_cfg, v_cfg, e_cfg)
        print(tab.to_string(index=False))
        print(f"  GANADOR fusion: {best['ganador']}  AUC={best['auc_ganador']:.3f}  "
              f"(mejor modalidad sola={best['auc_mejor_modalidad_sola']:.3f}, "
              f"delta={best['delta_fusion']:+.3f})")
        print(f"  [{dx}] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
