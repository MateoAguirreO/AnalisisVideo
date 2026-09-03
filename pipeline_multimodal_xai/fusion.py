"""Fusion multimodal: rama de video, early fusion y meta-modelos de late fusion.

- Rama video: pipeline sin PCA de feature_selectors.py a nivel participante.
- Early fusion: agrega los segmentos de audio a un vector por participante
  (media/std/percentiles), lo concatena con las features de video (prefijos
  aud__ / vid__) y aplica un unico pipeline de seleccion + modelo.
- Late fusion: meta-modelos sobre [score_audio, score_video] (2-D):
  soft/hard voting, ponderado (grid y AUC-proporcional), stacking-logreg, stacking-ANN.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import config_mm as cfg
from feature_selectors import construir_config, base_steps, clasificador  # noqa: F401
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

SEG_COL = cfg.AUDIO_SEG_COL

# ----------------------------------------------------------------------------
# Rama de video (participante)
# ----------------------------------------------------------------------------
VIDEO_CONFIGS = ["mutinfo_rf", "mutinfo_logreg", "anova_rf", "anova_logreg", "l1logreg"]


def construir_video_config(nombre):
    """(pipe, grid) para GridSearchCV interno; k de un solo digito (EPV ~20)."""
    pipe, grid = construir_config(nombre)
    if "sel__k" in grid:
        grid = {**grid, "sel__k": cfg.K_GRID}
    return pipe, grid


# ----------------------------------------------------------------------------
# Early fusion
# ----------------------------------------------------------------------------
def _agg_col(g: pd.DataFrame, feats, modos):
    out = {}
    for f in feats:
        v = g[f].values
        for m in modos:
            if m == "mean":
                out[f"aud__{f}__mean"] = np.mean(v)
            elif m == "std":
                out[f"aud__{f}__std"] = np.std(v)
            elif m.startswith("p"):
                out[f"aud__{f}__{m}"] = np.percentile(v, float(m[1:]))
    return pd.Series(out)


def matriz_early_fusion(vid_df: pd.DataFrame, aud_df: pd.DataFrame,
                        modos=None) -> pd.DataFrame:
    """DataFrame index=pid, columnas aud__*/vid__*, + 'label'.

    La agregacion de audio se calcula por-participante (solo con sus propios
    segmentos) -> no hay fuga entre participantes ni entre folds.
    """
    modos = modos or cfg.EARLY_FUSION_AGG
    feats_a = [c for c in aud_df.columns if c not in ("pid", SEG_COL, "label")]
    ag = aud_df.groupby("pid").apply(lambda g: _agg_col(g, feats_a, modos))
    feats_v = [c for c in vid_df.columns if c != "label"]
    vv = vid_df[feats_v].rename(columns={c: f"vid__{c}" for c in feats_v})
    out = vv.join(ag, how="inner")
    out["label"] = vid_df["label"]
    return out


def construir_early_config(nombre):
    """Como construir_video_config pero admite tambien 'anova_mlp' (ANN sobre features crudas)."""
    if nombre == "anova_mlp":
        pipe = Pipeline(base_steps() + [
            ("sel", SelectKBest(f_classif)),
            ("clf", MLPClassifier(hidden_layer_sizes=(8,), alpha=1.0, max_iter=800,
                                  early_stopping=True, random_state=cfg.RANDOM_STATE)),
        ])
        return pipe, {"sel__k": cfg.K_GRID}
    return construir_video_config(nombre)


EARLY_CONFIGS = ["mutinfo_rf", "anova_logreg", "l1logreg", "anova_mlp"]


# ----------------------------------------------------------------------------
# Late fusion: meta-modelos sobre [score_audio, score_video]
# ----------------------------------------------------------------------------
META_VARIANTES = ["soft_vote", "hard_vote", "pond_grid", "pond_auc",
                  "stack_logreg", "stack_ann"]


class MetaFusion:
    """Combina dos scores (prob. clase positiva) en una prediccion final.

    fit(S, y): S = array (n, 2) = [score_audio, score_video].
    """

    def __init__(self, variante="stack_logreg", random_state=cfg.RANDOM_STATE):
        self.variante = variante
        self.random_state = random_state

    def fit(self, S, y):
        S = np.asarray(S, float)
        y = np.asarray(y, int)
        self.alpha_ = 0.5
        if self.variante in ("soft_vote", "hard_vote"):
            pass
        elif self.variante == "pond_grid":
            grid = np.linspace(0, 1, 11)
            cvp = self._alpha_cv_scores(S, y, grid)
            self.alpha_ = float(grid[int(np.argmax(cvp))])
        elif self.variante == "pond_auc":
            a = self._safe_auc(y, S[:, 0]); v = self._safe_auc(y, S[:, 1])
            a, v = max(a - 0.5, 0) + 1e-6, max(v - 0.5, 0) + 1e-6
            self.alpha_ = float(a / (a + v))
        elif self.variante == "stack_logreg":
            self.clf_ = LogisticRegression(class_weight="balanced", max_iter=2000,
                                           random_state=self.random_state).fit(S, y)
        elif self.variante == "stack_ann":
            # N pequeno (~60 en train): red minima, fuerte regularizacion L2, sin
            # early_stopping (no alcanza para un hold-out interno estable), entradas escaladas.
            self.clf_ = make_pipeline(
                StandardScaler(),
                MLPClassifier(hidden_layer_sizes=(3,), alpha=3.0, max_iter=3000,
                              random_state=self.random_state),
            ).fit(S, y)
        else:
            raise ValueError(self.variante)
        return self

    @staticmethod
    def _safe_auc(y, s):
        return roc_auc_score(y, s) if len(np.unique(y)) == 2 else 0.5

    def _alpha_cv_scores(self, S, y, grid):
        cv = StratifiedKFold(5, shuffle=True, random_state=self.random_state)
        scores = np.zeros(len(grid))
        for tr, te in cv.split(S, y):
            for i, a in enumerate(grid):
                p = a * S[te, 0] + (1 - a) * S[te, 1]
                scores[i] += self._safe_auc(y[te], p)
        return scores / cv.get_n_splits()

    def predict_proba(self, S):
        S = np.asarray(S, float)
        if self.variante == "soft_vote":
            p = S.mean(axis=1)
        elif self.variante == "hard_vote":
            p = ((S[:, 0] > 0.5).astype(float) + (S[:, 1] > 0.5).astype(float)) / 2.0
        elif self.variante in ("pond_grid", "pond_auc"):
            p = self.alpha_ * S[:, 0] + (1 - self.alpha_) * S[:, 1]
        else:
            p = self.clf_.predict_proba(S)[:, 1]
        return np.clip(p, 0, 1)
