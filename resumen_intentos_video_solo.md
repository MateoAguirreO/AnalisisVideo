# Resumen ejecutivo: 7 intentos de mejorar la clasificación de solo-video (para descartar esta vía como suficiente)

**Fecha:** 2026-07-06 · **Objetivo del documento:** dejar registro, para el informe mensual, de todos los intentos realizados para subir el desempeño del componente de **video** (ansiedad y depresión, N=80) antes de concluir que el canal de video por sí solo no es suficiente y de pasar a la fusión multimodal audio+video.

> Detalle técnico completo (código, CSVs, gráficos SHAP) en `experimentos_sin_pca/REPORTE.md`. Este documento es la síntesis para reporte.

---

## 1. Punto de partida

| Versión | Features | Selección | AUC ansiedad | AUC depresión |
|---|---|---|---|---|
| v1 | 48 (agregados simples) | ninguna | 0.575 | 0.528 |
| v2 | 256 (temporales ricas) | ANOVA F-test (`SelectKBest`) embebido en CV | 0.541 | 0.605 |

v2 ya mostró que el ANOVA F-test embebido en la validación cruzada (sin fuga de datos) ayudaba algo a depresión, pero con una limitación metodológica: el mejor `(modelo, k)` se elegía mirando el mismo número que se reportaba como resultado final (sesgo optimista tipo "winner's curse", Vabalas et al. 2019). A partir de ahí se hicieron 7 intentos adicionales, documentados abajo, para (a) corregir ese sesgo y (b) agotar razonablemente las alternativas de solo-video antes de descartarlo.

---

## 2. Los 6 intentos

### Intento 1 — Comparación honesta de 12 estrategias de selección de features (incluyendo ANOVA), con nested cross-validation

**Método:** se compararon 12 configuraciones de selección/reducción de dimensionalidad — **ANOVA F-test** (la de v2), ANOVA + filtro de correlación de Pearson, **información mutua**, importancia embebida de Random Forest, L1-logística embebida y *stability selection* (bootstrap, Meinshausen & Bühlmann 2010) — cada una combinada con Regresión Logística / Random Forest / XGBoost. Se descartó explícitamente **PCA** (revisión de literatura: ningún paper de referencia lo usa para seleccionar features conductuales; rompe la trazabilidad SHAP que exige la tesis). Evaluación con **nested CV** (elección de hiperparámetros en un loop interno, AUC reportado del loop externo, sin el sesgo de v2) en dos etapas: screening (10 folds) y confirmación (25 folds). El resultado ganador se validó además con un **test de permutación** (300 barajes de etiqueta).

**Resultado:**

| Eje | Ganador | AUC honesto (nested CV) | Test de permutación |
|---|---|---|---|
| Ansiedad | Información mutua + Regresión Logística (k=4) | 0.573 ± 0.127 | p=0.093 — **no significativo** |
| Depresión | Información mutua + Random Forest (k=4) | **0.662 ± 0.148** | p=0.0033 — **significativo** |

**Sobre ANOVA específicamente:** el ANOVA F-test (`anova_logreg`, `anova_rf`, `anovacorr_*`) quedó consistentemente por debajo de la información mutua en las 12 configuraciones probadas para ambos ejes (ansiedad: mejor variante ANOVA=0.517 vs. 0.573 de información mutua; depresión: mejor variante ANOVA=0.514 vs. 0.662). Razón: el ANOVA F-test solo detecta relaciones **lineales** entre cada feature y la etiqueta; la información mutua también captura dependencia **no lineal**, que aparentemente es donde está parte de la señal en estos datos.

**Conclusión parcial:** con selección honesta, depresión mejora de forma **estadísticamente defendible** (0.605→0.662, p=0.0033); ansiedad no.

### Intento 2 — Auditoría de calidad de la etiqueta (ground truth)

**Método:** se descubrió que las columnas de respuesta ítem-por-ítem del PHQ-9/GAD-7 en el Excel de campo sí están disponibles (documentación previa decía lo contrario). Se sumaron con la escala estándar (0-3 por ítem) y se cruzaron contra la etiqueta categórica `DX ... IA` que usa todo el pipeline como ground truth.

**Resultado:** correlación punto-biserial prácticamente nula (depresión r=-0.04, p=0.75; ansiedad r=0.15, p=0.22). 12 de 18 personas etiquetadas "en riesgo" de depresión tienen PHQ-9 en rango mínimo; 17 de 51 "sin riesgo" tienen PHQ-9 leve o peor. Se probó re-entrenar usando el puntaje PHQ-9/GAD-7 directo como etiqueta alterna: **el AUC no mejora** (cae a ≈0.556 depresión, ≈0.559 ansiedad).

**Conclusión parcial:** hay una inconsistencia real entre el ground truth categórico y el autorreporte estandarizado (limitación a documentar, afecta también al componente de audio), pero **no es la causa** del techo de AUC — ninguna de las dos definiciones de etiqueta se relaciona fuertemente con el comportamiento facial.

### Intento 3 — Auditoría de calidad de extracción/tracking

**Método:** se revisó `face_detect_rate` y `hand_visible_frac` por participante para descartar errores de video/tracking.

**Resultado:** `face_detect_rate` mínimo = 0.95 en los 80 participantes (sin outliers de calidad). `hand_visible_frac` es bajo en general (encuadre tipo selfie), pero es una limitación de captura, no un bug.

**Conclusión parcial:** la extracción de features no tiene errores que expliquen el techo.

### Intento 4 — Multi-instancia por ventanas de 10 segundos

**Método:** en vez de 1 vector por participante (toda la entrevista, ~4.4 min), se probó partir en ventanas de 10s (enfoque de *AnxietyFaceTrack*, Sahu et al. 2025) — 1956 ventanas de 80 participantes. Validado con `StratifiedGroupKFold` (agrupando por participante) para no filtrar información entre train y test, agregando la probabilidad de las ventanas de cada persona para obtener el AUC honesto a nivel participante.

**Resultado:** ansiedad 0.585 ± 0.054 (≈ igual a 0.573); depresión 0.563 ± 0.025 (**peor** que 0.662).

**Conclusión parcial:** no ayuda. Las features que más aportan a depresión son tendencias a lo largo de **toda** la entrevista (aplanamiento progresivo), que una ventana aislada de 10s no puede capturar.

### Intento 5 — Detección de outliers

**Método:** `IsolationForest` sobre las 256 features de los 80 participantes, para identificar y evaluar la remoción de casos atípicos.

**Resultado:** los participantes más atípicos son mayoritariamente negativos en ambos ejes (removerlos solo reduciría la clase mayoritaria); el único caso de ansiedad "Alto Riesgo" también resulta atípico, pero removerlo eliminaría la única muestra de ansiedad severa disponible.

**Conclusión parcial:** no hay ningún caso concreto y justificable de remover.

### Intento 6 — Ensamble de modelos y regularización más amplia

**Método:** (a) ensamble por *soft-voting* de las 3 mejores configuraciones de cada eje; (b) ampliar el número de features seleccionadas (k) hasta 32 en vez de limitarlo a 4-8.

**Resultado:** el ensamble rinde **peor** que el mejor modelo individual (ansiedad 0.519, depresión 0.609); el k más amplio da un AUC prácticamente **igual** (ansiedad 0.567, depresión 0.658) — confirma que k=4 no dejaba desempeño sobre la mesa.

**Conclusión parcial:** ninguna de las dos palancas mejora el resultado de referencia.

### Intento 7 — Regresión sobre el puntaje continuo PHQ-9/GAD-7 (en vez de clasificación binaria)

**Método:** en lugar de la etiqueta binaria de riesgo, se usó el puntaje **continuo** del PHQ-9 (depresión) y del GAD-7 (ansiedad) como objetivo de **regresión**, por si el corte binario estuviera destruyendo información presente en el gradiente de síntomas. Modelos de regresión (Ridge y Random Forest regressor) con selección de features embebida (información mutua y F-test de regresión) en nested CV. Métrica: correlación de **Spearman** entre la predicción out-of-fold y el puntaje real (más robusta que R² con N pequeño).

**Resultado:**

| Eje | Mejor Spearman ρ | p-valor |
|---|---|---|
| Ansiedad (GAD-7) | −0.19 | 0.12 — no significativo (y signo negativo) |
| Depresión (PHQ-9) | 0.16 | 0.18 — no significativo |

**Conclusión parcial:** el gradiente continuo tampoco produce una relación estadísticamente significativa con el comportamiento facial. Es incluso el escenario más favorable posible (se regresa contra el propio puntaje del cuestionario, más informativo que la etiqueta categórica) y aun así no hay señal. Refuerza que el techo no depende de cómo se defina o discretice el objetivo.

---

## 3. Tabla resumen de los 7 intentos

| # | Intento | Resultado clave | ¿Mejoró el AUC? |
|---|---|---|---|
| 1 | 12 selectores (ANOVA, correlación, info. mutua, RF-importance, L1, stability) + nested CV + permutación | Depresión 0.662 (p=0.0033, **significativo**); ansiedad 0.573 (p=0.093, no signif.) | Sí, respecto a v2 (0.605→0.662 depresión) |
| 2 | Calidad del ground truth (`DX IA` vs. PHQ-9/GAD-7) | Etiquetas inconsistentes entre sí (r≈0), pero re-etiquetar no sube el AUC | No |
| 3 | Calidad de tracking/extracción | Sin errores (face_detect_rate ≥0.95 en el 100% de los casos) | No aplica (se descarta como causa) |
| 4 | Ventanas de 10s (multi-instancia) | Ansiedad ≈igual (0.585); depresión **peor** (0.563) | No |
| 5 | Detección de outliers | Ningún caso justificable de remover | No |
| 6 | Ensamble / k más amplio | Ensamble peor; k amplio ≈igual | No |
| 7 | Regresión sobre puntaje continuo PHQ-9/GAD-7 | Spearman ρ no significativo (dep 0.16 p=0.18; ans −0.19 p=0.12) | No |

**Resultado de referencia final de solo-video (tras los 7 intentos): depresión AUC 0.662 (significativo), ansiedad AUC ≈0.57-0.59 (no significativo).**

---

## 4. Conclusión

Se agotaron siete vías de mejora metodológicamente distintas — selección de features (con ANOVA F-test como método de partida y también evaluado explícitamente frente a 5 alternativas), calidad del ground truth, calidad de la extracción, granularidad temporal (ventanas vs. entrevista completa), robustez a atípicos, ensamblado/regularización, y objetivo continuo (regresión sobre el puntaje en vez de clasificación binaria) — y en ninguna se superó el techo alcanzado en el primer intento. Con N=80 (19-21 casos positivos por eje, mayoritariamente de severidad **leve**), el video por sí solo:

- **Sí aporta señal real y estadísticamente defendible para depresión** (AUC 0.662, p=0.0033), pero de magnitud moderada.
- **No aporta señal estadísticamente defendible para ansiedad** (AUC ≈0.57, no distinguible de azar con confianza al 95%).

**Esto descarta el video como fuente única y suficiente de clasificación de riesgo**, tanto para ansiedad como (en menor medida) para depresión, y confirma que la vía correcta es la **fusión multimodal audio+video**, donde cada canal débil por separado puede aportar información complementaria al otro.

---

## 5. Referencias metodológicas citadas en los intentos

- N. Meinshausen, P. Bühlmann, "Stability Selection," *J. R. Stat. Soc. B*, 2010.
- A. Vabalas, E. Gowen, E. Poliakoff, A. J. Casson, "Machine learning algorithm validation with a limited sample size," *PLOS ONE* 14(11):e0224365, 2019.
- N. K. Sahu, S. Gupta, H. R. Lone, "AnxietyFaceTrack: A Smartphone-Based Non-Intrusive Approach for Detecting Social Anxiety Using Facial Features," arXiv:2502.16106, 2025.
- Detalle técnico completo, código y CSVs de resultados: `experimentos_sin_pca/REPORTE.md`.
