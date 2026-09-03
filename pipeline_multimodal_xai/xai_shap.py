"""SHAP sobre el pipeline multimodal ganador de cada eje (reentrenado en los 79).

Produce tres vistas complementarias:
  1) NIVEL MODALIDAD  -> cuanto aporta la voz vs. el rostro. Se calcula con un
     meta-modelo logistico sobre [score_audio, score_video] OOF (LinearExplainer);
     es la vista canonica "voz vs rostro" independientemente de que variante de
     fusion haya ganado en eval_multimodal.py. Si el ganador es una ponderacion se
     reporta ademas el alpha.
  2) INTRA-MODALIDAD (audio)  -> |SHAP| por feature eGeMAPS (TreeExplainer sobre el
     modelo de segmento), agregado y agrupado por familia acustica.
  3) INTRA-MODALIDAD (video)  -> |SHAP| por Action Unit / pose / emocion, agrupado
     por familia facial.
  + early fusion: SHAP unico sobre la matriz concatenada, agrupado por prefijo
    aud__/vid__, como chequeo de consistencia entre arquitecturas.

Uso:  python pipeline_multimodal_xai/xai_shap.py   (requiere best_config_<dx>.json)
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import config_mm as cfg
from data_multimodal import cargar
from audio_branch import RamaAudio, construir_estimador_segmento
from fusion import construir_video_config, matriz_early_fusion, construir_early_config
from eval_multimodal import audio_scores, video_scores

warnings.filterwarnings("ignore")
RS = cfg.RANDOM_STATE


# --- helpers de forma de SHAP (calcados de experimentos_sin_pca/shap_final.py) ---
def _indices_paso(step):
    if hasattr(step, "get_support"):
        return step.get_support(indices=True)
    if hasattr(step, "support_"):
        return np.asarray(step.support_)
    return None


def nombres_finales(pipe, feats):
    idx = np.arange(len(feats))
    for _, step in pipe.steps[:-1]:
        sub = _indices_paso(step)
        if sub is not None:
            idx = idx[sub]
    return [feats[i] for i in idx]


def _shap_pos(explainer, Xt):
    sv = explainer(Xt)
    vals = sv.values if hasattr(sv, "values") else sv
    if isinstance(vals, list):
        vals = np.asarray(vals[1])
    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    return vals


def _guardar_bar(vals, Xt, feats, path, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    try:
        shap.summary_plot(vals, Xt, feature_names=feats, show=False, plot_type="bar")
        plt.title(titulo)
        plt.tight_layout()
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  (plot omitido: {e})")


def _ranking_familias(rank, familias):
    rank = rank.copy()
    rank["familia"] = rank["feature"].map(lambda f: cfg.familia_de(f, familias))
    fam = rank.groupby("familia")["shap_importance"].sum().sort_values(ascending=False)
    return (fam / fam.sum()).rename("aporte_relativo").reset_index()


# ---------------------------------------------------------------------------
def oof_scores_full(dx, vid, aud, audio_cfg, video_cfg, n_splits=5):
    """Scores OOF de audio y video para los 79 participantes (para el SHAP de modalidad)."""
    y = vid["label"].values
    Xv = vid.drop(columns="label").values
    pids = vid.index.values
    a = np.zeros(len(y)); v = np.zeros(len(y))
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=RS)
    for tr, te in cv.split(pids, y):
        _, a[te] = audio_scores(aud, pids, y, tr, te, audio_cfg)
        _, v[te] = video_scores(Xv, y, tr, te, video_cfg)
    return a, v, y


def shap_modalidad(dx, a, v, y, best):
    import shap
    from sklearn.linear_model import LogisticRegression
    S = np.column_stack([a, v])
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RS).fit(S, y)
    expl = shap.LinearExplainer(clf, S)
    vals = _shap_pos(expl, S)
    imp = np.abs(vals).mean(axis=0)
    rel = imp / imp.sum()
    out = pd.DataFrame({"modalidad": ["audio (voz)", "video (rostro)"],
                        "shap_abs_mean": np.round(imp, 4),
                        "aporte_relativo": np.round(rel, 3)})
    out.to_csv(cfg.DIR_RESULTADOS / f"shap_modalidad_{dx}.csv", index=False)
    _guardar_bar(vals, S, ["audio (voz)", "video (rostro)"],
                 cfg.DIR_RESULTADOS / f"shap_modalidad_{dx}.png",
                 f"{dx} — aporte por modalidad (SHAP, meta-logistico)")
    print(f"  [modalidad] voz={rel[0]:.0%}  rostro={rel[1]:.0%}"
          + (f"  | alpha_ganador={best.get('alpha_mean'):.2f}" if best.get("alpha_mean") is not None else ""))
    return out


def shap_audio(dx, aud, audio_cfg):
    import shap
    df = aud
    feats = [c for c in df.columns if c not in ("pid", cfg.AUDIO_SEG_COL, "label")]
    X, yv = df[feats].values, df["label"].values
    # el SHAP intra-modalidad necesita un modelo explicable; si el ganador es lineal
    # (l1logreg/anova_logreg) se explica igual, si es de arbol se usa TreeExplainer.
    pipe = construir_estimador_segmento(audio_cfg).fit(X, yv)
    ff = nombres_finales(pipe, feats)
    Xt = pipe[:-1].transform(X)
    clf = pipe.named_steps["clf"]
    try:
        expl = shap.TreeExplainer(clf)
        vals = _shap_pos(expl, Xt)
    except Exception:
        expl = shap.Explainer(clf, Xt, feature_names=ff)
        vals = _shap_pos(expl, Xt)
    imp = np.abs(vals).mean(axis=0)
    rank = pd.DataFrame({"feature": ff, "shap_importance": imp}).sort_values(
        "shap_importance", ascending=False)
    rank.to_csv(cfg.DIR_RESULTADOS / f"shap_audio_{dx}.csv", index=False)
    fam = _ranking_familias(rank, cfg.FAMILIAS_AUDIO)
    fam.to_csv(cfg.DIR_RESULTADOS / f"shap_audio_familias_{dx}.csv", index=False)
    _guardar_bar(vals, Xt, ff, cfg.DIR_RESULTADOS / f"shap_audio_{dx}.png",
                 f"{dx} — features de voz (eGeMAPS, SHAP nivel segmento)")
    print(f"  [audio] top: {', '.join(rank['feature'].head(5))}")
    print(f"  [audio] familias: {dict(zip(fam['familia'], fam['aporte_relativo'].round(2)))}")


def shap_video(dx, vid, video_cfg):
    import shap
    feats = [c for c in vid.columns if c != "label"]
    X, y = vid[feats].values, vid["label"].values
    pipe, grid = construir_video_config(video_cfg)
    if grid:
        from sklearn.model_selection import GridSearchCV
        pipe = GridSearchCV(pipe, grid, cv=5, scoring="roc_auc", n_jobs=-1).fit(X, y).best_estimator_
    pipe.fit(X, y)
    ff = nombres_finales(pipe, feats)
    Xt = pipe[:-1].transform(X)
    clf = pipe.named_steps["clf"]
    expl = shap.Explainer(clf, Xt, feature_names=ff)
    vals = _shap_pos(expl, Xt)
    imp = np.abs(vals).mean(axis=0)
    rank = pd.DataFrame({"feature": ff, "shap_importance": imp}).sort_values(
        "shap_importance", ascending=False)
    rank.to_csv(cfg.DIR_RESULTADOS / f"shap_video_{dx}.csv", index=False)
    fam = _ranking_familias(rank, cfg.FAMILIAS_VIDEO)
    fam.to_csv(cfg.DIR_RESULTADOS / f"shap_video_familias_{dx}.csv", index=False)
    _guardar_bar(vals, Xt, ff, cfg.DIR_RESULTADOS / f"shap_video_{dx}.png",
                 f"{dx} — features de rostro (AU/pose/emocion, SHAP)")
    print(f"  [video] top: {', '.join(rank['feature'].head(5))}")
    print(f"  [video] familias: {dict(zip(fam['familia'], fam['aporte_relativo'].round(2)))}")


def shap_early(dx, Me, early_cfg):
    import shap
    from sklearn.model_selection import GridSearchCV
    feats = [c for c in Me.columns if c != "label"]
    X, y = Me[feats].values, Me["label"].values
    pipe, grid = construir_early_config(early_cfg)
    if grid:
        pipe = GridSearchCV(pipe, grid, cv=5, scoring="roc_auc", n_jobs=-1).fit(X, y).best_estimator_
    pipe.fit(X, y)
    ff = nombres_finales(pipe, feats)
    Xt = pipe[:-1].transform(X)
    clf = pipe.named_steps["clf"]
    try:
        expl = shap.Explainer(clf, Xt, feature_names=ff)
        vals = _shap_pos(expl, Xt)
    except Exception:
        expl = shap.KernelExplainer(lambda z: clf.predict_proba(z)[:, 1], shap.sample(Xt, 30))
        vals = np.asarray(expl.shap_values(Xt))
    imp = np.abs(vals).mean(axis=0)
    rank = pd.DataFrame({"feature": ff, "shap_importance": imp})
    rank["modalidad"] = rank["feature"].map(lambda f: "audio" if f.startswith("aud__") else "video")
    rank = rank.sort_values("shap_importance", ascending=False)
    rank.to_csv(cfg.DIR_RESULTADOS / f"shap_early_{dx}.csv", index=False)
    bal = rank.groupby("modalidad")["shap_importance"].sum()
    bal = (bal / bal.sum()).round(3)
    print(f"  [early] balance modalidad: audio={bal.get('audio',0):.0%}  video={bal.get('video',0):.0%}")
    print(f"  [early] top: {', '.join(rank['feature'].head(6))}")
    return rank


def main():
    for dx in cfg.DXS:
        pj = cfg.DIR_RESULTADOS / f"best_config_{dx}.json"
        if not pj.exists():
            print(f"[{dx}] falta {pj.name}; corre eval_multimodal.py primero"); continue
        best = json.loads(pj.read_text(encoding="utf-8"))
        vid, aud = cargar(dx)
        Me = matriz_early_fusion(vid, aud)
        print(f"\n=== {dx.upper()} === ganador={best['ganador']}  "
              f"(audio={best['audio_config']}, video={best['video_config']}, early={best['early_config']})")
        a, v, y = oof_scores_full(dx, vid, aud, best["audio_config"], best["video_config"])
        shap_modalidad(dx, a, v, y, best)
        shap_audio(dx, aud, best["audio_config"])
        shap_video(dx, vid, best["video_config"])
        shap_early(dx, Me, best["early_config"])


if __name__ == "__main__":
    main()
