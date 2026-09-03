"""Rama de audio: clasificador eGeMAPS a nivel SEGMENTO + agregacion a participante.

El audio viene segmentado (23-157 segmentos por participante). El modelo se entrena
a nivel segmento; para obtener una prediccion por PARTICIPANTE se agrega la
probabilidad de sus segmentos (media por defecto).

Sin fuga:
- El score de audio a nivel participante para el conjunto de ENTRENAMIENTO de un
  fold externo se obtiene con `cross_val_predict` + `StratifiedGroupKFold`
  (grupo = participante): ningun segmento de un participante contribuye a su propia
  probabilidad agregada.
- El score para el conjunto de TEST sale del modelo ajustado con TODOS los segmentos
  de entrenamiento.

Configuraciones de segmento: se reutilizan las de
`experimentos_sin_pca/feature_selectors.py` (mismos selectores sin PCA).
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

import config_mm as cfg
from feature_selectors import construir_config

SEG_COL = cfg.AUDIO_SEG_COL

# configs de segmento a considerar (subconjunto barato de feature_selectors.CONFIGS:
# a nivel segmento hay miles de muestras -> la seleccion pesada aporta poco y cuesta caro)
AUDIO_CONFIGS = ["anova_rf", "anova_xgb", "anova_logreg", "l1logreg"]


def _agg(valores: np.ndarray, modo: str) -> float:
    if modo == "mean":
        return float(np.mean(valores))
    if modo == "median":
        return float(np.median(valores))
    if modo.startswith("p"):
        return float(np.percentile(valores, float(modo[1:])))
    raise ValueError(modo)


def agregar_a_participante(pids: np.ndarray, probas: np.ndarray, modo: str) -> pd.Series:
    """probas: prob. clase positiva por segmento -> Series index=pid."""
    s = pd.DataFrame({"pid": pids, "p": probas}).groupby("pid")["p"].apply(
        lambda v: _agg(v.values, modo))
    return s


def construir_estimador_segmento(nombre: str):
    """Pipeline de segmento con hiperparametros fijos (k=8) para acotar el costo.

    A nivel segmento hay ~4000 muestras de entrenamiento por fold: el grid de k no
    cambia mucho el resultado y multiplica el costo por el numero de folds externos.
    Se fija k=8 (documentado) y se deja el grid solo para el early fusion / video,
    que si son de N pequeno.
    """
    pipe, grid = construir_config(nombre)
    if "sel__k" in grid:
        pipe.set_params(sel__k=8)
    elif "sel__max_features" in grid:
        pipe.set_params(sel__max_features=8)
    return pipe


class RamaAudio:
    """Modelo de segmento + agregacion. Se ajusta con un DataFrame de segmentos."""

    def __init__(self, config="anova_rf", agg=cfg.AUDIO_PROB_AGG, n_inner=5,
                 random_state=cfg.RANDOM_STATE):
        self.config = config
        self.agg = agg
        self.n_inner = n_inner
        self.random_state = random_state

    def _xy(self, df):
        feats = [c for c in df.columns if c not in ("pid", SEG_COL, "label")]
        return df[feats].values, df["label"].values, df["pid"].values, feats

    def fit(self, df_train):
        X, y, _, feats = self._xy(df_train)
        self.feats_ = feats
        self.estimator_ = construir_estimador_segmento(self.config)
        self.estimator_.fit(X, y)
        return self

    def scores_test_participante(self, df_test) -> pd.Series:
        X, _, pids, _ = self._xy(df_test)
        p = self.estimator_.predict_proba(X)[:, 1]
        return agregar_a_participante(pids, p, self.agg)

    def scores_oof_participante(self, df_train) -> pd.Series:
        """OOF a nivel segmento (GroupKFold por participante) -> agregado a participante."""
        X, y, pids, _ = self._xy(df_train)
        n_grupos = len(np.unique(pids))
        n_splits = min(self.n_inner, n_grupos, int(np.min(np.bincount(y))))
        n_splits = max(n_splits, 2)
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        est = construir_estimador_segmento(self.config)
        p = cross_val_predict(est, X, y, cv=cv, groups=pids, method="predict_proba", n_jobs=-1)[:, 1]
        return agregar_a_participante(pids, p, self.agg)
