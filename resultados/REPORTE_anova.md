# Informe ANOVA — Features de Video (MediaPipe / Google)

_Generado: 2026-06-10 08:30_

---

## 1. Metodología

- **Extracción de features**: MediaPipe Tasks API (Google) con modelos `face_landmarker`, `pose_landmarker_lite` y `hand_landmarker`. Features incluyen blendshapes faciales (parpadeo, apertura ocular, sonrisa, ceño, posición de cejas, apertura de boca), orientación de cabeza (yaw/pitch/roll), movimiento corporal y de manos.
- **ANOVA univariado (features)**: F-test de una vía para cada feature, comparando grupo positivo vs. negativo por diagnóstico. Valores faltantes imputados con mediana antes del test.
- **Corrección de comparaciones múltiples**: Benjamini-Hochberg (FDR α = 5%). Una feature es *significativa* si p_adj < 0.05.
- **Tamaño de efecto**: η² (eta-cuadrado). Umbrales: trivial < 0.01 ≤ pequeño < 0.06 ≤ mediano < 0.14 ≤ grande.
- **ANOVA entre modelos**: F-test de una vía sobre las 25 puntuaciones AUC-ROC de la CV RepeatedStratifiedKFold (5 folds × 5 repeticiones). Modelos comparados: L1-LogReg, RandomForest, XGBoost, cada uno con su mejor `k` de selección de features según `metricas_v2_{dx}.csv`.
- **Post-hoc**: t-tests independientes entre pares de modelos, corrección Bonferroni (α/3 ≈ 0.0167).

---

## 2. Depresion

**Muestra**: n = 80 (positivos = 21 · negativos = 59)

### 2.1 ANOVA por Feature

- Features analizadas: **256**
- Significativas (p_adj < 0.05): **0**

_Ninguna feature resultó significativa tras corrección FDR._

### 2.2 ANOVA entre Modelos

| Modelo | AUC medio | AUC std | Mejor k |
|--------|----------:|--------:|--------:|
| l1logreg | 0.605 | 0.110 | 24 |
| rf | 0.561 | 0.128 | 24 |
| xgb | 0.546 | 0.167 | 12 |

**F = 1.211, p = 0.3040** — ✗ No hay diferencia significativa entre modelos (p ≥ 0.05).

---

## 2. Ansiedad

**Muestra**: n = 80 (positivos = 19 · negativos = 61)

### 2.1 ANOVA por Feature

- Features analizadas: **256**
- Significativas (p_adj < 0.05): **0**

_Ninguna feature resultó significativa tras corrección FDR._

### 2.2 ANOVA entre Modelos

| Modelo | AUC medio | AUC std | Mejor k |
|--------|----------:|--------:|--------:|
| l1logreg | 0.541 | 0.117 | 16 |
| rf | 0.536 | 0.117 | 24 |
| xgb | 0.483 | 0.129 | 24 |

**F = 1.654, p = 0.1984** — ✗ No hay diferencia significativa entre modelos (p ≥ 0.05).

---

## 3. Conclusiones

### Depresion

- Ninguna feature de video discrimina significativamente entre grupos para depresion tras corrección FDR. Las features univariadas tienen poder discriminativo limitado.
- Los tres modelos (L1-LogReg, RF, XGBoost) **no difieren** significativamente en AUC (F = 1.21, p = 0.3040). El mejor modelo es **l1logreg** (AUC = 0.605).

### Ansiedad

- Ninguna feature de video discrimina significativamente entre grupos para ansiedad tras corrección FDR. Las features univariadas tienen poder discriminativo limitado.
- Los tres modelos (L1-LogReg, RF, XGBoost) **no difieren** significativamente en AUC (F = 1.65, p = 0.1984). El mejor modelo es **l1logreg** (AUC = 0.541).

---

_Archivos generados:_

- `anova_features_depresion.csv` — ANOVA completo por feature
- `anova_modelos_depresion_folds.csv` — AUC por fold
- `anova_modelos_depresion_pairwise.csv` — post-hoc
- `anova_features_ansiedad.csv` — ANOVA completo por feature
- `anova_modelos_ansiedad_folds.csv` — AUC por fold
- `anova_modelos_ansiedad_pairwise.csv` — post-hoc
