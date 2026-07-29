"""Selectores/reductores de dimensionalidad SIN PCA + fabricas de pipelines.

Motivacion (ver resultados/../REPORTE.md): con N=80 (19-24 positivos por eje) y
p=256 features de video, PCA es una mala eleccion para esta tesis porque (a) es
NO supervisado -> no hay garantia de que capture la direccion clinicamente
relevante cuando la senal ya es debil, y (b) convierte cada feature en una
combinacion lineal de las 256 originales, rompiendo la trazabilidad
feature -> SHAP que exige el hilo XAI de la tesis. Aqui solo se usan metodos
que preservan la identidad de cada feature original.

Selectores implementados:
  - CorrelationFilter : filtro de redundancia (Pearson), igual a "Beyond
    Questionnaires" (Sahu et al. 2025).
  - SelectKBest(f_classif)      : ANOVA F-test univariado (ya usado en v2).
  - SelectKBest(mutual_info_classif) : dependencia no lineal univariada.
  - SelectFromModel(RandomForest)    : importancia por impureza, como
    "AnxietyFaceTrack" (Sahu et al. 2025).
  - StabilitySelector            : bootstrap + L1-logistica, Meinshausen &
    Buhlmann (2010); da un nucleo de features ESTABLE frente al resample,
    condicion necesaria para que el SHAP final no sea artefacto del split.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42
K_GRID = [4, 6, 8]  # EPV: con ~20 positivos, features finales de un solo digito


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Descarta features redundantes por correlacion de Pearson (umbral fijo).

    Recorre las columnas en orden y descarta la j-esima si su |corr| con
    alguna columna ya conservada supera el umbral. Se ajusta SOLO con el
    fold de entrenamiento (va dentro del Pipeline) -> sin fuga.
    """

    def __init__(self, threshold=0.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n_feats = X.shape[1]
        std = X.std(axis=0)
        keep = []
        for j in range(n_feats):
            if std[j] < 1e-12:
                continue  # constante, no aporta y evita corr NaN
            redundante = False
            for k in keep:
                c = np.corrcoef(X[:, j], X[:, k])[0, 1]
                if np.isfinite(c) and abs(c) > self.threshold:
                    redundante = True
                    break
            if not redundante:
                keep.append(j)
        self.support_ = np.array(keep, dtype=int)
        if self.support_.size == 0:  # salvaguarda: no debería pasar con datos reales
            self.support_ = np.arange(min(3, n_feats))
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]


class StabilitySelector(BaseEstimator, TransformerMixin):
    """Stability selection (Meinshausen & Buhlmann 2010): bootstrap + L1-logistica.

    Conserva las features cuya frecuencia de seleccion (coeficiente != 0) a
    traves de B remuestreos bootstrap supera `threshold`. Umbral fijado a
    priori (no se tunea por grid) para no gastar otro grado de libertad con
    N=80: 0.6 es el valor estandar recomendado en la literatura de origen.
    """

    def __init__(self, n_bootstrap=150, threshold=0.6, C=0.5, min_features=3, random_state=RANDOM_STATE):
        self.n_bootstrap = n_bootstrap
        self.threshold = threshold
        self.C = C
        self.min_features = min_features
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_feats = X.shape[1]
        counts = np.zeros(n_feats)
        rng = np.random.RandomState(self.random_state)
        for b in range(self.n_bootstrap):
            seed = rng.randint(0, 1_000_000)
            Xb, yb = resample(X, y, replace=True, stratify=y, random_state=seed)
            if len(np.unique(yb)) < 2:
                continue
            clf = Pipeline([
                ("sc", StandardScaler()),
                ("lr", LogisticRegression(penalty="l1", solver="liblinear", C=self.C,
                                          class_weight="balanced", random_state=seed, max_iter=2000)),
            ])
            clf.fit(Xb, yb)
            coefs = clf.named_steps["lr"].coef_.ravel()
            counts += (np.abs(coefs) > 1e-9).astype(float)
        self.selection_freq_ = counts / self.n_bootstrap
        support = np.where(self.selection_freq_ >= self.threshold)[0]
        if support.size < self.min_features:
            support = np.argsort(-self.selection_freq_)[: self.min_features]
        self.support_ = np.sort(support)
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]


def _mutual_info_score(X, y):
    """Top-level (picklable) wrapper: mutual_info_classif con random_state fijo."""
    return mutual_info_classif(X, y, random_state=RANDOM_STATE)


def base_steps(incluir_corr=False, corr_umbral=0.75):
    steps = [
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("var", VarianceThreshold(0.0)),
    ]
    if incluir_corr:
        steps.append(("corr", CorrelationFilter(threshold=corr_umbral)))
    steps.append(("sc", StandardScaler()))
    return steps


def clasificador(nombre):
    if nombre == "logreg":
        return LogisticRegression(penalty="l2", C=1.0, class_weight="balanced",
                                  max_iter=2000, random_state=RANDOM_STATE)
    if nombre == "l1logreg":
        return LogisticRegression(penalty="l1", solver="liblinear", C=0.5,
                                  class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)
    if nombre == "rf":
        # n_jobs=1: la paralelizacion vive en el loop externo de CV (evita sobre-suscripcion)
        return RandomForestClassifier(n_estimators=400, max_depth=4, min_samples_leaf=3,
                                      class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1)
    if nombre == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=250, max_depth=2, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.8, reg_lambda=2.0,
                             eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1)
    raise ValueError(nombre)


def construir_config(nombre):
    """Devuelve (Pipeline, param_grid) para GridSearchCV interno. Ningun paso usa PCA."""

    if nombre == "l1logreg":
        pipe = Pipeline(base_steps() + [("clf", clasificador("l1logreg"))])
        grid = {"clf__C": [0.05, 0.1, 0.3, 0.5, 1.0]}
        return pipe, grid

    if nombre == "anova_logreg":
        pipe = Pipeline(base_steps() + [("sel", SelectKBest(f_classif)), ("clf", clasificador("logreg"))])
        return pipe, {"sel__k": K_GRID, "clf__C": [0.3, 1.0]}

    if nombre == "anova_rf":
        pipe = Pipeline(base_steps() + [("sel", SelectKBest(f_classif)), ("clf", clasificador("rf"))])
        return pipe, {"sel__k": K_GRID}

    if nombre == "anova_xgb":
        pipe = Pipeline(base_steps() + [("sel", SelectKBest(f_classif)), ("clf", clasificador("xgb"))])
        return pipe, {"sel__k": K_GRID}

    if nombre == "anovacorr_logreg":
        pipe = Pipeline(base_steps(incluir_corr=True) + [("sel", SelectKBest(f_classif)), ("clf", clasificador("logreg"))])
        return pipe, {"sel__k": K_GRID}

    if nombre == "anovacorr_rf":
        pipe = Pipeline(base_steps(incluir_corr=True) + [("sel", SelectKBest(f_classif)), ("clf", clasificador("rf"))])
        return pipe, {"sel__k": K_GRID}

    if nombre == "mutinfo_logreg":
        pipe = Pipeline(base_steps() + [("sel", SelectKBest(_mutual_info_score)), ("clf", clasificador("logreg"))])
        return pipe, {"sel__k": K_GRID}

    if nombre == "mutinfo_rf":
        pipe = Pipeline(base_steps() + [("sel", SelectKBest(_mutual_info_score)), ("clf", clasificador("rf"))])
        return pipe, {"sel__k": K_GRID}

    if nombre == "rfimportance_logreg":
        sel = SelectFromModel(RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                      random_state=RANDOM_STATE, n_jobs=1),
                              threshold=-np.inf, max_features=6)
        pipe = Pipeline(base_steps() + [("sel", sel), ("clf", clasificador("logreg"))])
        return pipe, {"sel__max_features": K_GRID}

    if nombre == "rfimportance_rf":
        sel = SelectFromModel(RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                      random_state=RANDOM_STATE, n_jobs=1),
                              threshold=-np.inf, max_features=6)
        pipe = Pipeline(base_steps() + [("sel", sel), ("clf", clasificador("rf"))])
        return pipe, {"sel__max_features": K_GRID}

    if nombre == "stability_logreg":
        pipe = Pipeline(base_steps() + [("sel", StabilitySelector()), ("clf", clasificador("logreg"))])
        return pipe, {}  # umbral fijo a priori, ver docstring de StabilitySelector

    if nombre == "stability_rf":
        pipe = Pipeline(base_steps() + [("sel", StabilitySelector()), ("clf", clasificador("rf"))])
        return pipe, {}

    raise ValueError(nombre)


CONFIGS = ["l1logreg", "anova_logreg", "anova_rf", "anova_xgb",
           "anovacorr_logreg", "anovacorr_rf", "mutinfo_logreg", "mutinfo_rf",
           "rfimportance_logreg", "rfimportance_rf", "stability_logreg", "stability_rf"]
