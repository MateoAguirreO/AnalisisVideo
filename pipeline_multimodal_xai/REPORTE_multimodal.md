# Fusión multimodal audio + video con explicabilidad (SHAP + LIME)

**Fecha:** 2026-09-03 · **Datos:** 79 participantes (audio ∩ video), cruce por **código**.
**Modalidades:** voz = 88 funcionales eGeMAPS v02 a nivel segmento (`features_<dx>_egemaps.csv`);
rostro = 66 features de Action Units / pose de cabeza / prototipos de emoción a nivel
participante (`dataset_au_features.csv`).
**Motivación:** `experimentos_sin_pca/REPORTE.md` §8 concluye que, agotada la vía de solo-video
(depresión AUC≈0.66 p=0.003; ansiedad AUC≈0.57 no significativo), **la palanca que queda es la
fusión multimodal** — la expectativa era que ayudara sobre todo a ansiedad. Este documento la
implementa y la evalúa con la misma batería honesta (nested CV, permutación, N pequeño),
añadiendo el hilo XAI de la tesis: SHAP a nivel modalidad (voz vs. rostro) y por familia de
features, y LIME local por paciente. *(Spoiler: la fusión termina ayudando más a depresión que
a ansiedad — ver §3.3 y §10.)*

**No toca los componentes previos** (audio solo, video solo `v1/v2/v3-sin-PCA`): los combina.

---

## 1. Datos y alineación

| | valor |
|---|---|
| Participantes audio ∩ video | **79** (el 66 tiene video pero no audio → fuera del análisis multimodal) |
| Segmentos de audio | 5224 (ansiedad) / 5225 (depresión); 23–157 por participante, mediana 62 |
| Features de voz | 88 (eGeMAPS v02 funcionales, por segmento) |
| Features de rostro | 66 (AU mean/std, pose de cabeza, prototipos de emoción). Se descartan `error` (100% NaN) y `n_frames_detected` (control de calidad, posible confusor) |
| Clases ansiedad | 60 sin riesgo / **19** riesgo |
| Clases depresión | 58 sin riesgo / **21** riesgo |
| Consistencia de etiqueta | `target_<dx>` (video) coincide **100%** con el `label` de segmento (audio) por participante |

EPV ≈ 19–21 positivos: mismo régimen que solo-video. Cualquier modelo debe terminar con un
puñado de features (k ∈ {4,6,8}); el meta-modelo de fusión vive en 2 dimensiones.

---

## 2. Arquitectura

**Ramas base (independientes por eje — nunca conjunto ansiedad/depresión):**

- **Voz:** clasificador eGeMAPS a nivel **segmento** (configs sin PCA de
  `feature_selectors.py`: `SelectKBest`+modelo con k=8, o L1-logística embebida) → se agrega
  la probabilidad de los segmentos de cada participante (media). Los scores de entrenamiento
  para el meta-modelo son **out-of-fold** (`StratifiedGroupKFold`, grupo = participante):
  ningún segmento contribuye a la probabilidad agregada de su propio participante.
- **Rostro:** pipeline sin PCA a nivel participante (imputer → varianza → escala →
  `SelectKBest` → clasificador), `GridSearchCV` interno sobre k.

**Late fusion** — meta-modelo sobre `[score_voz, score_rostro]` (2-D):

| variante | qué hace |
|---|---|
| `soft_vote` | media de las dos probabilidades (sin entrenar) |
| `hard_vote` | regla sobre umbral 0.5 |
| `pond_grid` | `P = α·P_voz + (1−α)·P_rostro`, α ∈ {0,…,1} por AUC en CV interna |
| `pond_auc` | α = AUC_voz / (AUC_voz + AUC_rostro) (ecuación cerrada, estilo AVEC) |
| `stack_logreg` | regresión logística sobre los 2 scores |
| `stack_ann` | `MLPClassifier(3 neuronas, α=3.0)` sobre los 2 scores (entradas escaladas) |

**Early fusion** (comparación): se agregan los segmentos de audio a un vector por participante
(media/std/p20/p50/p80 → 440 cols), se concatena con las 66 de rostro (prefijos `aud__`/`vid__`,
506 features), un único pipeline de selección + modelo (incluye una ANN sobre features crudas).

**Evaluación:** *screening* `RepeatedStratifiedKFold(5,2)` = 10 folds elige la config de cada
rama; *confirmación* `(5,5)` = 25 folds evalúa todos los baselines y variantes. El AUC
reportado es el del loop externo. Todo a nivel participante, por eje, `RANDOM_STATE=42`.

**Guardas anti-fuga:** splits por participante en todos los loops; agregación de audio OOF con
GroupKFold; selección de features e imputación dentro de cada pipeline; los scores base que
alimentan al meta-modelo son out-of-fold; la agregación de audio es por-participante.

---

## 3. Resultados (AUC honesto, 25 folds externos)

### 3.0 Screening de configs de cada rama (10 folds) — top 4 por eje

| Ansiedad | AUC | | Depresión | AUC |
|---|---|---|---|---|
| `video:anova_logreg` | 0.656 | | `video:anova_rf` | 0.657 |
| `audio:anova_xgb` | 0.585 | | `video:mutinfo_rf` | 0.648 |
| `video:l1logreg` | 0.583 | | `audio:l1logreg` | 0.644 |
| `audio:anova_logreg` | 0.574 | | `video:l1logreg` | 0.638 |

Se congela la mejor config de rama-voz, rama-rostro y early por eje (tabla completa en
`resultados/screening_<dx>.csv`) y se pasa a la confirmación de 25 folds. El early fusion
nunca supera 0.56 (ansiedad) / 0.60 (depresión) en screening.

### 3.1 Ansiedad — configs ganadoras: voz `anova_xgb`, rostro `anova_logreg`, early `anova_logreg`

| método | AUC | ±std | F1-macro | bAcc |
|---|---|---|---|---|
| **`soft_vote` (fusión)** | **0.668** | 0.160 | 0.542 | 0.562 |
| solo rostro | 0.661 | 0.153 | 0.562 | 0.603 |
| `pond_auc` (α_voz≈0.21) | 0.641 | 0.167 | 0.553 | 0.577 |
| `pond_grid` (α_voz≈0.24) | 0.611 | 0.148 | 0.542 | 0.570 |
| `hard_vote` | 0.594 | 0.125 | 0.554 | 0.597 |
| solo voz | 0.572 | 0.149 | 0.426 | 0.488 |
| `stack_logreg` | 0.557 | 0.212 | 0.523 | 0.558 |
| `stack_ann` | 0.537 | 0.189 | 0.453 | 0.503 |
| early fusion | 0.511 | 0.156 | 0.498 | 0.530 |
| mayoría | 0.500 | 0.000 | 0.432 | 0.500 |

**Δ fusión vs. mejor modalidad sola = +0.007.** Marginal.

### 3.2 Depresión — configs ganadoras: voz `l1logreg`, rostro `anova_rf`, early `anova_mlp`

| método | AUC | ±std | F1-macro | bAcc |
|---|---|---|---|---|
| **`soft_vote` (fusión)** | **0.674** | 0.114 | 0.606 | 0.610 |
| `pond_grid` (α_voz≈0.28) | 0.649 | 0.117 | 0.564 | 0.573 |
| `pond_auc` (α_voz≈0.25) | 0.642 | 0.123 | 0.566 | 0.575 |
| solo rostro | 0.638 | 0.111 | 0.547 | 0.554 |
| `stack_logreg` | 0.626 | 0.135 | 0.560 | 0.590 |
| early fusion | 0.619 | 0.147 | 0.502 | 0.554 |
| solo voz | 0.593 | 0.123 | 0.533 | 0.551 |
| `hard_vote` | 0.587 | 0.120 | 0.527 | 0.574 |
| `stack_ann` | 0.583 | 0.132 | 0.485 | 0.523 |
| mayoría | 0.500 | 0.000 | 0.423 | 0.500 |

**Δ fusión vs. mejor modalidad sola = +0.036.**

### 3.3 Lecturas transversales

1. **`soft_vote` (promedio simple de las 2 probabilidades) gana en ambos ejes.** Todo
   meta-modelo *entrenado* (`stack_logreg`, `stack_ann`, `pond_grid`, `pond_auc`) queda por
   debajo: con ~63 participantes de entrenamiento y un espacio de 2 features, ajustar el peso
   de combinación gasta un grado de libertad que no compensa. Es el mismo hallazgo que la
   literatura AVEC de N pequeño (el promedio no ponderado es un baseline difícil de batir).
   La ANN (`stack_ann`, red de 3 neuronas con regularización fuerte) queda 0.09–0.13 de AUC
   por debajo de `soft_vote` — como se anticipó en el plan, no hay datos para entrenar una
   red y se incluye solo por completitud metodológica.
2. **Early fusion (506 features, 79 muestras) rinde por debajo de late fusion** en ambos
   ejes: la concatenación diluye la señal y la selección no logra recuperar un núcleo útil.
   La ANN sobre features crudas (`anova_mlp`, que gana el *screening* de early para depresión)
   no cambia el cuadro.
3. **La fusión ayuda a depresión (+0.036) más que a ansiedad (+0.007).** Para ansiedad, la
   rama de rostro con este set de AU ya llega a 0.66 sola (mejor que el 0.57 histórico con
   las 256 features temporales — ver §9), y la voz aporta poco margen extra.

---

## 4. Test de permutación (¿real o ruido de N=79?)

Arquitectura ganadora congelada (`soft_vote` sobre la config de voz + rostro de cada eje),
`StratifiedKFold(5)`, barajes de etiqueta. p = (1 + #{AUC_perm ≥ AUC_real}) / (n+1).

| Eje | config congelada | AUC real (5-fold) | AUC nulo (media) | AUC nulo p95 | n perm | **p-valor** |
|---|---|---|---|---|---|---|
| Ansiedad | voz `anova_xgb` + rostro `anova_logreg` | 0.676 | 0.492 | 0.653 | 200 | **0.045 — significativo** |
| Depresión | voz `l1logreg` + rostro `anova_rf` | 0.674 | 0.482 | 0.665 | 200 | **0.040 — significativo** |

**Lectura.** Los dos ejes superan el azar de forma estadísticamente defendible, aunque con
poco margen: el AUC real cae apenas por encima del p95 de la distribución nula (0.653/0.665).
Con N=79 esto es esperable. Consistente con `experimentos_sin_pca` (depresión-solo-video daba
p=0.003 con las 256 features temporales; aquí, con menos features de rostro pero la voz
sumada, la evidencia es más justa pero sigue del lado significativo). El AUC real de la
permutación (0.674 depresión, 0.676 ansiedad) **coincide con el de la confirmación nested CV**
(§3) — el pipeline congelado no pierde desempeño.
*Nota de costo:* la permutación reentrena la rama de audio en cada barajada (n=200 barajes por
eje; granularidad mínima de p ≈ 0.005).

---

## 5. SHAP — nivel modalidad (¿cuánto aporta la voz vs. el rostro?)

Meta-modelo logístico sobre `[score_voz, score_rostro]` OOF de los 79 participantes,
`LinearExplainer`. |SHAP| medio normalizado:

| Eje | voz | rostro |
|---|---|---|
| Ansiedad | **18 %** | **82 %** |
| Depresión | **45 %** | **55 %** |

En el early fusion (SHAP sobre las 506 features concatenadas, agrupado por prefijo):
ansiedad audio 54 % / video 46 %; depresión audio 24 % / video 76 %. Las dos arquitecturas
coinciden en que para ansiedad el rostro domina claramente; para depresión discrepan (late:
parejo; early: video 76 %). El balance exacto **no es estable** (§7.2): estos porcentajes son
una tendencia, no una medición.

---

## 6. SHAP — features con nombre clínico (intra-modalidad)

### 6.1 Voz (eGeMAPS, SHAP a nivel segmento, agrupado por familia acústica)

| Eje | modelo de voz (ganador) | familias (aporte relativo) | top features |
|---|---|---|---|
| Ansiedad | `anova_xgb` (árbol, TreeExplainer) | F0/prosodia 0.41 · loudness 0.37 · MFCC 0.18 · espectral 0.04 | `logRelF0-H1-H2_amean`, `loudness_p20`, `logRelF0-H1-A3_amean`, `loudness_p50`, `mfcc4_amean` |
| Depresión | `l1logreg` (lineal L1, coef.) | formantes 0.49 · espectral 0.17 · MFCC 0.14 · F0 0.08 · loudness 0.06 | `F3amplitudeLogRelF0_stddevNorm`, `F2amplitudeLogRelF0_amean`, `F1amplitudeLogRelF0_stddevNorm`, `alphaRatioV_amean`, `hammarbergIndexV_amean` |

**Lectura clínica.** *Ansiedad:* `logRelF0-H1-H2` / `logRelF0-H1-A3` (relación de armónicos,
marcador de fonación tensa vs. soplada) + percentiles de `loudness` (energía vocal) — coherente
con tensión laríngea y control de la intensidad bajo estrés. *Depresión:* dominan las
**amplitudes de formantes relativas a F0** (`FxamplitudeLogRelF0`, definición de resonancias
vocálicas) y el **balance espectral** (`alphaRatio`, `hammarbergIndex`, energía grave vs.
aguda) — marcadores de articulación reducida y voz "apagada"/poco proyectada, consistente con
el aplanamiento prosódico descrito en la literatura de depresión. (El modelo de voz de
depresión es lineal L1, así que estas son las features con coeficiente no nulo.)

### 6.2 Rostro (AU / pose / emoción, SHAP a nivel participante, agrupado por familia)

| Eje | familias (aporte relativo) | top features (SHAP) |
|---|---|---|
| Ansiedad | cara inferior 0.56 · cara superior 0.44 | `AU15_std` (0.63), `AU20_std` (0.53), `AU01_mean` (0.40), `AU05_mean` (0.33), `AU04_mean` (0.19) |
| Depresión | emoción 0.54 · cara superior 0.46 | `anger_mean` (0.12), `AU01_mean` (0.10), `anger_std` (0.05), `AU05_std` (0.04) |

**Lectura clínica.** Ansiedad: variabilidad de `AU15` (depresor de la comisura labial) y
`AU20` (estirador de labios) + `AU01`/`AU04`/`AU05` (tensión frontal y de párpado) — tensión
perioral y frontal, coherente con Giannakakis et al. (2017/2024). Depresión: el prototipo de
**enojo** (`anger`) y `AU01` (elevador interno de ceja, marcador clásico de tristeza) + `AU05` —
afecto negativo facial, coherente con el patrón "afecto negativo / ceño" de la literatura de
depresión.

---

## 7. Estabilidad de las explicaciones (bootstrap B=200)

### 7.1 Núcleo de features estable (frecuencia de selección en 200 bootstraps ≥ 0.5)

| Eje | rostro | voz |
|---|---|---|
| Ansiedad | `AU15_std` (0.66), `AU01_mean` (0.56) | `logRelF0-H1-H2_amean` (0.88), `mfcc4_amean` (0.53) |
| Depresión | `anger_mean` (0.86), `anger_std` (0.79), `AU01_mean` (0.67) | `logRelF0-H1-H2_amean` (0.66), `F1frequency_amean` (0.53) |

El núcleo estable **coincide con el top de SHAP** (§6) para el rostro en ambos ejes y para la
voz en ansiedad → esa parte de la explicación no es un artefacto de un solo split. (El probe
de estabilidad usa un selector RF común para los dos ejes, así que para la voz en depresión
señala `logRelF0-H1-H2` / `F1frequency` como lo robustamente informativo, en vez de las
amplitudes de formante que elige el modelo lineal L1 ganador — dos lentes, ambas apuntan a
resonancia/calidad vocal.)

### 7.2 Estabilidad del peso de modalidad (200 bootstraps de los scores OOF)

| Eje | coef. rel. voz (media ± std) | p05–p95 (coef. voz) | α ponderado-AUC (peso voz) |
|---|---|---|---|
| Ansiedad | 0.41 ± 0.21 | 0.07–0.82 | 0.50 ± 0.21 |
| Depresión | 0.50 ± 0.20 | 0.17–0.88 | 0.53 ± 0.25 |

**El peso de modalidad es muy inestable** (el intervalo p05–p95 cubre casi todo [0,1]). Con
N=79 la pregunta "¿cuánto pesa la voz vs. el rostro?" no tiene una respuesta puntual
defendible — solo una tendencia (en ansiedad el rostro pesa algo más; en depresión están
parejos). Esto **refuerza** por qué `soft_vote` (peso fijo 50/50) gana: cualquier peso
aprendido está sobreajustado al split. Nótese que el SHAP de modalidad (§5) da rostro 82 %
para ansiedad mientras el bootstrap da ~59 % — la diferencia es que el SHAP pondera además
por la varianza de cada score, y ninguna de las dos cifras es estable; la conclusión
robusta es solo el orden (rostro ≥ voz en ansiedad).

---

## 8. LIME — explicación local por paciente + consistencia con SHAP

Sobre la matriz de early fusion (1 fila/participante, features `aud__`/`vid__`), 3
participantes elegidos por su predicción OOF: un verdadero positivo, un falso positivo y un
verdadero negativo. Cada uno produce una lista rankeada única que mezcla voz y rostro
(`resultados/lime_<dx>_{TP,FP,TN}.{txt,png}`).

**Consistencia SHAP vs. LIME** (mismos casos, misma función `predict_proba`, mismo espacio de
506 features; Jaccard del top-10, Spearman del ranking sobre las features que alguno de los
dos marca):

| Eje | caso (pid) | Jaccard top-10 | Spearman |
|---|---|---|---|
| Ansiedad | TP (61) | 0.11 | −0.25 |
| Ansiedad | FP (44) | 0.25 | +0.28 |
| Ansiedad | TN (54) | 0.11 | −0.25 |
| Depresión | TP (35) | 0.11 | −0.25 |
| Depresión | FP (15) | 0.18 | −0.27 |
| Depresión | TN (9) | 0.18 | +0.02 |

**Matiz importante:** SHAP y LIME **coinciden en la feature dominante** de cada caso
(ansiedad: `vid__AU15_std`; depresión: `vid__anger_mean`, que también encabezan el SHAP del
early fusion), pero **discrepan en la cola** (features de rango 2–10), que es donde vive el
ruido — de ahí el Jaccard bajo (≈ 0.1–0.25) y el Spearman cercano a 0. Ejemplos
(`resultados/lime_<dx>_TP.txt`):

- Ansiedad, TP (pid 61, y=1, p=0.76): `AU15_std` en rango medio (+0.070) y
  `logRelF0-H1-H2` de la voz (+0.037) empujan a "riesgo".
- Depresión, TP (pid 35, y=1, p=0.99): `anger_mean > 0.02` domina por completo (+0.29); el
  resto de features (voz incluida) aportan < 0.03 cada una.

Es el comportamiento esperado de la literatura XAI con N pequeño y features correlacionadas
(Salih et al. 2026): la **feature principal** de una explicación local es fiable; el orden
fino no. La lectura defendible para la tesis es la **global** (§5–§6), respaldada por el
bootstrap (§7); LIME se reporta como ilustración cualitativa por-paciente, no como evidencia.

---

## 9. Limitaciones (honestidad ante todo)

1. **N=79, 19–21 positivos.** Los ±0.11–0.16 entre folds siguen siendo anchos; el 0.674 de
   depresión y el 0.638 de rostro solo tienen intervalos que se solapan. La mejora de la
   fusión es defendible para depresión (Δ+0.036, y el eje pasa permutación) pero modesta.
2. **La ganancia de la fusión es chica.** `soft_vote` supera a la mejor modalidad sola por
   +0.007 (ansiedad) y +0.036 (depresión). No es el salto que se esperaba; el techo de
   ~0.65–0.68 por AUC se mantiene, ahora con dos modalidades en vez de una.
3. **El CSV de AU (66 features) es una extracción distinta** de las 256 features temporales
   de `video_features_v2.csv` usadas en `experimentos_sin_pca/`. El baseline solo-rostro de
   aquí (ansiedad 0.66, depresión 0.64) **no es comparable** con el número canónico de video
   de la tesis (ansiedad 0.57, depresión 0.66). Que ansiedad suba a 0.66 con AU es un
   hallazgo a verificar por separado, no una conclusión de este documento.
4. **La agregación de audio (media de probabilidades de segmento) es un parámetro libre**,
   no optimizado dentro de la CV. Análisis de sensibilidad (`sensibilidad_agg.py`,
   `soft_vote`, 15 folds): ansiedad `mean`/`median`/`p80` = 0.681 / 0.685 / 0.687 (idénticas);
   depresión = 0.689 / 0.683 / 0.671 (`mean` mejor, `p80` cae ~0.02). La elección de `mean`
   no cambia la conclusión.
5. **Validez del ground truth**, heredada de `experimentos_sin_pca/REPORTE.md` §7.2: la
   etiqueta `DX ... IA` no correlaciona con el PHQ-9/GAD-7 de la misma encuesta. Aplica
   igual a las dos modalidades.
6. **El balance de modalidad (§5) no es estable (§7.2).** Los porcentajes "voz X % / rostro
   Y %" son una tendencia, no una medición.

---

## 10. Conclusión

- **Fusión tardía por promedio simple (`soft_vote`)** es la mejor arquitectura para ambos
  ejes: AUC **0.668 (ansiedad)** y **0.674 (depresión)**, superando a todo meta-modelo
  entrenado (`stack_logreg`, `stack_ann`, `pond_grid`, `pond_auc`) y al early fusion.
- **Depresión:** la fusión aporta una mejora sobre la mejor modalidad sola (+0.036) y el eje
  pasa el test de permutación (p=0.040). Voz y rostro pesan parejo.
- **Ansiedad:** la fusión ≈ rostro solo (+0.007); el rostro (con features de AU) es el canal
  dominante (SHAP 82 %) y pasa el test de permutación (p=0.045).
- **XAI:** la voz aporta **calidad de fonación / balance espectral** (`logRelF0-H1-H2`,
  `alphaRatio`, amplitudes de formantes); el rostro aporta **tensión perioral/frontal**
  (ansiedad: `AU15`, `AU20`, `AU01`) y **afecto negativo** (depresión: prototipo `anger`,
  `AU01`). La lectura global es estable (bootstrap); la local (LIME) solo en su feature
  principal.

---

## 11. Reproducibilidad

```
pip install -r pipeline_multimodal_xai/requirements.txt
python pipeline_multimodal_xai/run_all.py
```

Archivos: `config_mm.py`, `data_multimodal.py`, `audio_branch.py`, `fusion.py`,
`eval_multimodal.py`, `permutation_multimodal.py`, `xai_shap.py`, `xai_lime.py`,
`xai_stability.py`, `sensibilidad_agg.py`, `run_all.py`; salidas en `resultados/`.

## 12. Próximos pasos

1. Verificar por separado si el set de AU (66 features) realmente mejora ansiedad respecto a
   `video_features_v2.csv` (repetir `experimentos_sin_pca` con ese CSV).
2. Si se consigue más muestra (sobre todo casos moderados/severos), repetir: tanto el AUC
   como el peso de modalidad deberían estabilizarse con folds más grandes.
3. Explorar agregación de audio dentro de la CV (mediana/p80/atención) como hiperparámetro.
4. rPPG facial (ritmo cardiaco) como tercera señal — ver `experimentos_sin_pca/REPORTE.md` §8.
