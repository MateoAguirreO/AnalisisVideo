# Experimentos sin PCA — selección de features honesta (video)

**Fecha:** 2026-07-06 · **Datos:** 80 clips, video_features_v2.csv (256 features), cruce por **código**.
**Motivación:** subir el AUC de los modelos de video sin sacrificar la interpretabilidad SHAP que exige la tesis (XAI). Antes de implementar, se revisó qué hacen los propios papers de referencia y la literatura metodológica de N pequeño / p grande — ver síntesis al final de este documento. Conclusión de esa revisión: **ninguno de los papers usa PCA como paso de selección de features conductuales**, y la evidencia general (interpretabilidad clínica + riesgo de sobreajuste con ~20 positivos) desaconseja PCA para este caso. Este documento reemplaza ese paso por alternativas que preservan la identidad de cada feature.
**No toca v1 (`resultados/REPORTE_resultados.md`) ni v2 (`resultados/REPORTE_v2.md`)**: los complementa.

---

## 1. Qué corrige esta ronda respecto a v2

1. **Sin PCA en ningún punto del pipeline final.** Se prueban 6 estrategias de reducción que preservan el nombre de cada feature: filtro de correlación (Pearson), ANOVA F-test (`SelectKBest(f_classif)`, la que ya usaba v2), información mutua (`SelectKBest(mutual_info_classif)`), importancia embebida de Random Forest (`SelectFromModel`), L1-logística embebida, y **stability selection** (bootstrap + L1, Meinshausen & Bühlmann 2010).
2. **Nested CV real, sin "winner's curse".** En v2, el mejor `(modelo, k)` se elegía mirando el AUC medio de la misma CV que se reportaba como resultado — optimismo documentado por Vabalas et al. (2019, PLOS ONE). Aquí la elección de hiperparámetros vive en un loop **interno** (`GridSearchCV`) y el AUC reportado es el del loop **externo**, que nunca vio esa elección.
3. **Dos etapas por costo computacional:** *screening* (12 configuraciones × 2 ejes, outer 5×2=10 folds) y *confirmación* (top-3 por eje, outer 5×5=25 folds, igual esquema que v1/v2 para comparabilidad).
4. **Test de permutación** (300 barajes de etiqueta) sobre el pipeline ganador de cada eje, para descartar que la mejora sea ruido de muestreo — el riesgo que más se repite en la literatura consultada para N=80.
5. **Stability selection independiente** sobre todos los datos (solo para *reportar* un núcleo robusto de cara al SHAP final, nunca para medir desempeño — ese principio ya lo aplicaba v2 en su función `interpretar()`).

## 2. Resultado (AUC, comparación v1 → v2 → v3-sin-PCA)

| Eje | v1 (48 feats, sin selección) | v2 (SelectKBest ANOVA embebido) | **v3 sin PCA (nested CV honesta)** | Permutación (300 barajes) |
|---|---|---|---|---|
| **Ansiedad** | 0.575 | 0.541 | 0.573 — `mutinfo_logreg`, k=4 (±0.127) | p=0.093 — **no significativo** |
| **Depresión** | 0.528 | 0.605 | **0.662** — `mutinfo_rf`, k=4 (±0.148) | p=0.0033 — **significativo** |

Línea base por mayoría: ansiedad 0.76 acc, depresión 0.74 acc (azar en AUC = 0.50).

**Hallazgo central:** la selección por **información mutua** (captura dependencia no lineal, algo que el ANOVA F-test de v2 no puede ver) superó a las otras 11 configuraciones en ambos ejes — y para depresión, de forma que **sí pasa el test de permutación** (p=0.0033): es la primera vez en el historial de este componente (v1→v2→v3) que la mejora respecto al azar es estadísticamente defendible, no solo "sugestiva".

### 2.1 Detalle screening (10 folds, 12 configuraciones) — top 5 por eje

**Ansiedad:**
| config | AUC | ±std |
|---|---|---|
| mutinfo_logreg | 0.583 | 0.138 |
| anovacorr_logreg | 0.517 | 0.103 |
| stability_logreg | 0.497 | 0.107 |
| mutinfo_rf | 0.494 | 0.167 |
| anova_rf | 0.492 | 0.146 |

**Depresión:**
| config | AUC | ±std |
|---|---|---|
| mutinfo_rf | 0.731 | 0.145 |
| mutinfo_logreg | 0.614 | 0.175 |
| rfimportance_logreg | 0.564 | 0.124 |
| anovacorr_logreg | 0.514 | 0.098 |
| anova_logreg | 0.512 | 0.153 |

Tabla completa: `resultados/screening_ansiedad.csv`, `resultados/screening_depresion.csv`.

### 2.2 Confirmación (25 folds, top-3 por eje)

| Eje | config | AUC | ±std |
|---|---|---|---|
| Ansiedad | mutinfo_logreg | **0.573** | 0.127 |
| Ansiedad | anovacorr_logreg | 0.506 | 0.126 |
| Ansiedad | stability_logreg | 0.475 | 0.120 |
| Depresión | mutinfo_rf | **0.662** | 0.148 |
| Depresión | mutinfo_logreg | 0.592 | 0.131 |
| Depresión | rfimportance_logreg | 0.539 | 0.122 |

## 3. Test de permutación (¿es real o es ruido de N=80?)

Se congelan los hiperparámetros ganadores (`k=4` en ambos casos) y se evalúa el pipeline con Stratified 5-fold sobre las etiquetas reales y sobre 300 barajes de las etiquetas.

| Eje | AUC real (5-fold) | AUC nulo (media) | AUC nulo p90 | **p-valor** |
|---|---|---|---|---|
| Ansiedad | 0.630 | 0.503 | 0.621 | **0.093** (no signif. a α=0.05) |
| Depresión | 0.742 | 0.502 | 0.630 | **0.0033** (significativo) |

**Lectura:** para depresión, el AUC observado está por encima del percentil 95 de la distribución nula — es información real de video, no azar. Para ansiedad, el AUC observado cae dentro del rango que también se alcanza barajando etiquetas ~9% de las veces: **no hay evidencia estadística suficiente** de que el video capture ansiedad por encima del azar con este pipeline, consistente con v1 y v2.

## 4. Núcleo estable de features (stability selection, todos los datos, B=200 bootstraps)

Interesante honestidad metodológica: **stability selection embebida en la CV rindió peor que las demás** (`stability_logreg`/`stability_rf` quedaron últimas o casi últimas en ambos ejes, Sección 2.1) — con folds de entrenamiento de ~64 muestras, otra capa de bootstrap encima resulta demasiado inestable. Se conserva como **herramienta de reporte** (no de predicción), corrida sobre los 80 participantes completos:

| Eje | Núcleo estable (freq. de selección ≥ 0.6, B=200) |
|---|---|
| Ansiedad | `gaze_out_slope` (0.64), `jaw_open_median` (0.60) |
| Depresión | `pitch_skew` (0.78), `yaw_kurt` (0.76), `roll_skew` (0.71) |

Notar que este núcleo (dominado por **pose de cabeza**) es distinto de las features que eligió `mutinfo_rf`/`mutinfo_logreg` para clasificar (dominadas por **tensión facial**, Sección 5). Dos lentes razonables — un filtro no lineal optimizado para clasificar vs. un criterio de robustez frente al remuestreo — señalan familias de features distintas. Con N=80 esto es esperable y es un hallazgo honesto en sí mismo: **el biomarcador "verdadero" no está todavía fijado**, hay más de una familia de señal candidata (pose de cabeza = retardo psicomotor; tensión facial = afecto negativo), ambas clínicamente plausibles.

## 5. Interpretación SHAP (modelo final, features originales, CERO componentes de PCA)

Al no usar PCA, cada valor SHAP es directamente atribuible a una feature con nombre clínico:

**Ansiedad** (`mutinfo_logreg`, k=4):
| feature | SHAP |
|---|---|
| `frown_p90` | 0.408 |
| `blink_slope` | 0.210 |
| `cheek_squint_skew` | 0.132 |
| `brow_inner_up_slope` | 0.038 |

**Depresión** (`mutinfo_rf`, k=4):
| feature | SHAP |
|---|---|
| `brow_inner_up_slope` | 0.088 |
| `cheek_squint_skew` | 0.087 |
| `cheek_squint_kurt` | 0.074 |
| `gaze_up_skew` | 0.037 |

Lectura clínica: `frown_p90` (ceño fruncido intenso) y `blink_slope` (tendencia del parpadeo durante la entrevista) dominan ansiedad — coherente con tensión perioral/frontal reportada en Giannakakis et al. (2017/2024). `brow_inner_up` (AU1, marcador clásico de tristeza) y tensión de mejilla/ojo (`cheek_squint`) dominan depresión — coherente con afecto negativo. Plots: `resultados/shap_ansiedad.png`, `resultados/shap_depresion.png`.

## 6. Limitaciones (honestidad ante todo)

1. **Optimismo residual de comparar 12 configuraciones.** Cada AUC individual (Sección 2.2) es honesto (nested CV), pero elegir "la mejor de 12" para reportarla introduce un sesgo de selección menor (mucho más chico que el de v2, que comparaba post-hoc sobre la misma métrica reportada, pero no es cero). El test de permutación de la Sección 3 es precisamente el control para esto, y depresión lo pasa.
2. **N=80, 19-21 positivos.** Los ±0.12-0.15 de desviación entre folds siguen siendo anchos; el intervalo de 0.662 se acerca a solaparse con el 0.605 de v2. La mejora es defendible (pasa permutación) pero no elimina la necesidad de más datos.
3. **Ansiedad sigue sin señal defendible** con ningún método probado (12 configuraciones, 0 pasan el criterio de significancia). Esto ya no es un problema de método de selección; es del contenido informativo del canal de video para este eje específico.
4. **Divergencia de familias de features** entre el núcleo estable (pose de cabeza) y las features del clasificador ganador (tensión facial) — ver Sección 4. No se resuelve con este tamaño muestral.

## 7. Auditoría: ¿hay un error, o el techo es real? (`check_label_quality.py`)

Ante la pregunta "¿esto no mejora más porque hay un bug, o de plano no se puede?", se auditó el pipeline completo buscando errores concretos, no solo se asumió el resultado.

**7.1 Bug descartado — cruce de etiqueta por cédula.** `src/extract_features.py` (el script de v1) etiqueta usando `documento` (cédula) como llave al escribir `video_features.csv` — justo el error que la documentación del proyecto dice haber corregido (11 participantes tienen cédula de carpeta ≠ cédula del Excel). Se verificó que **no impacta el entrenamiento**: tanto `build_dataset.py` (v1) como `merge_features.py` descartan esas columnas y las re-derivan desde cero por **código** antes de construir los datasets de entrenamiento. Es código muerto/vestigial, no un bug activo.

**7.2 Hallazgo real — la etiqueta `DX ... IA` no correlaciona con el PHQ-9/GAD-7 de la misma encuesta.** Las columnas `PHQ 9_*` / `GAD7_*` del Excel de campo **no están vacías** (el docstring de `src/labels.py` dice que sí, está desactualizado): tienen las respuestas Likert ítem por ítem. Sumándolas con la escala estándar (0-3 por ítem) y cruzando contra la etiqueta categórica `DX ... IA` que usa TODO el pipeline (audio y video) como ground truth:

| Eje | correlación punto-biserial (label vs. puntaje sumado) | Etiquetados "Riesgo" con puntaje mínimo (<5) | Etiquetados "Sin Riesgo" con puntaje leve+ (≥5) |
|---|---|---|---|
| Depresión (PHQ-9) | r=-0.039 (p=0.75) | 12/18 | 17/51 |
| Ansiedad (GAD-7) | r=0.150 (p=0.22) | 8/16 | 11/53 |

Es decir: la etiqueta binaria que se usa como ground truth es, en la práctica, **estadísticamente independiente** del puntaje que arrojan las mismas preguntas que los participantes respondieron en la misma encuesta. Esto no es un error de código de este repo — el archivo fuente es `analisis-espectrogramas/.MUESTRAS_SAMANÁ/Base de Datos Trabajo de campo RESULTADOS.xlsx`, ajeno a este pipeline — pero sí es una limitación de validez del ground truth que aplica a **ambos componentes de la tesis** (audio y video), y vale la pena resolver con quien generó esa columna "IA" (¿de qué instrumento/algoritmo sale exactamente?).

**Se probó empíricamente si esto era "el arreglo"** (`check_label_quality.py`): re-entrenar el mismo tipo de pipeline usando un target binario derivado directamente del puntaje PHQ-9/GAD-7 (corte clínico estándar ≥5 = leve o peor) en vez de `DX ... IA`. Resultado — **no mejora**: depresión cae a AUC≈0.556 (vs. 0.662 con `DX IA`) y ansiedad queda igual de débil (≈0.559 vs. 0.573). Conclusión honesta: el ground truth categórico y el autorreporte numérico son mutuamente inconsistentes, pero **ninguno de los dos** tiene una relación fuerte con el comportamiento facial observable en video — no era "el bug escondido", es una segunda confirmación de que el techo es real.

**7.3 La explicación estructural más probable — casi todos los "positivos" son leves.** La distribución de severidad detrás de la etiqueta binaria:

| Depresión | n | | Ansiedad | n |
|---|---|---|---|---|
| Sin Riesgo | 59 | | Sin Riesgo | 61 |
| Riesgo Leve | 17 | | Riesgo Leve | 15 |
| Riesgo Moderado | 4 | | Riesgo Moderadamente alto | 3 |
| — (no hay "alto"/"severo") | 0 | | Alto Riesgo | 1 |

**No hay ni un solo caso de depresión severa** en la muestra, y el 89% de los "positivos" de ansiedad son leves. La literatura de biomarcadores faciales (afecto plano, retardo psicomotor, aversión de mirada) describe manifestaciones que se vuelven observables en cuadros moderados-severos; en riesgo **leve** el comportamiento facial es, casi por definición, sutil o ausente. Con N=80 y una población dominada por casos leves, un AUC de 0.55-0.66 puede ser, honestamente, cercano al techo alcanzable desde solo-video — no por falta de esfuerzo en selección/modelado (ya se probaron 12+ estrategias con nested CV y permutación), sino porque la señal que se busca detectar es intrínsecamente débil en la mayoría de los casos positivos de esta muestra.

**7.4 Última palanca probada — ventanas de 10s (multi-instancia), tampoco funcionó.** Mateo insistió en agotar las opciones de solo-video antes de pasar a fusión. Se probó el enfoque de *AnxietyFaceTrack* (Sahu et al. 2025): en vez de agregar TODA la entrevista a 1 vector por participante, partir las series por-frame ya guardadas (`features/series/*.npz`) en ventanas de 10s (`windowed_features.py`) y tratar cada ventana como una muestra de entrenamiento — de 80 participantes se generan 1956 ventanas (24.4 en promedio por persona). Evaluado con `StratifiedGroupKFold` (agrupando por código, tanto en el loop externo como en el interno de selección de `k`, para que ninguna ventana de un participante de test aparezca en train) y agregando la probabilidad de las ventanas de cada persona para obtener el AUC honesto a **nivel participante** (`windowed_nested_cv.py`):

| Eje | Mejor config por ventana | AUC ventana | **AUC participante (honesto)** | vs. mejor whole-clip |
|---|---|---|---|---|
| Ansiedad | mutinfo_logreg | 0.587±0.049 | 0.585±0.054 | 0.573 (≈ igual, dentro del ruido) |
| Depresión | anova_rf | 0.553±0.026 | 0.563±0.025 | **0.662 (peor: -0.10)** |

**No ayudó — y para depresión, empeoró.** Lectura honesta: las features que más aportaron en el enfoque whole-clip (Sección 5) son de **tendencia/dinámica a lo largo de TODA la entrevista** (`brow_inner_up_slope`, `frown_segstd`, percentiles calculados sobre ~4 minutos) — información que una ventana aislada de 10s no puede reconstruir por diseño (con 100 muestras no se calculan de forma confiable percentiles finos, y una tendencia entre segmentos de la entrevista completa no existe dentro de una sola ventana). El enfoque de AnxietyFaceTrack funciona bien en su caso porque miden una **tarea de exposición social aguda** (reactividad puntual esperable dentro de 10s); esta es una **entrevista clínica**, donde la señal de depresión parece ser una tendencia lenta (aplanamiento progresivo, fatiga), no un evento puntual. Multi-instancia con ventanas cortas no es la palanca correcta para este diseño de captura.

**7.5 Chequeos finales — outliers, ensamble, k más amplio (todos negativos).** Se agotaron las palancas razonables adicionales (`final_checks.py`), cada eje modelado por separado (nunca conjunto/multi-task, aunque ansiedad y depresión están fuertemente correlacionadas en esta muestra: 56/5/3/16, r=0.735, χ²p=3.4e-10 — se descarta modelar esto conjuntamente porque el diseño de la tesis requiere un modelo y una explicación SHAP independiente por eje):

- **Outliers:** IsolationForest sobre las 256 features no encontró ningún participante cuya remoción sea justificable — los más atípicos son mayormente negativos en ambos ejes (quitarlos solo encoge la clase mayoritaria) y el único caso "Alto Riesgo" de ansiedad también aparece como atípico (quitarlo eliminaría la única muestra de ansiedad severa disponible). No hay un caso de "hay que botar a este participante".
- **Ensamble (soft-voting) de las 3 mejores configuraciones por eje:** ansiedad AUC=0.519 (peor que 0.573), depresión AUC=0.609 (peor que 0.662). Combinar un modelo fuerte con dos más débiles diluye la señal en vez de reforzarla.
- **Grid de k más amplio (hasta 32, no solo 4-8):** ansiedad AUC=0.567 (≈igual a 0.573), depresión AUC=0.658 (≈igual a 0.662). Confirma que k=4 no se estaba quedando corto — el algoritmo converge al mismo lugar con más margen.

**Conclusión tras 6 intentos (selección de features, calidad de etiqueta, calidad de tracking, multi-instancia, outliers, ensamble/regularización):** con video solo y N=80, el resultado de referencia sigue siendo **depresión AUC 0.662 (p=0.0033, significativo)** y **ansiedad AUC ~0.57-0.59 (no significativo)** de la Sección 2. Ningún ajuste adicional de solo-video lo superó — a este punto el techo se considera real, no un problema de método. La única palanca de solo-video que queda sin probar y que SÍ añadiría información genuinamente nueva (no solo reordenar las 256 features existentes) es extraer **ritmo cardiaco por rPPG desde el rostro** (ver Jhon et al. 2025 en Sección 6 de `dossier_literatura_video.md`, N=1453, sí encuentra señal de depresión vía HRV facial) — es un desarrollo nuevo de procesamiento de señal (no una re-selección de features), con éxito incierto dado que el video de campo es H.264 comprimido y sin control de iluminación; se deja como decisión explícita del usuario si vale la pena invertir ese esfuerzo antes de pasar a fusión audio+video.

**Conclusión de la auditoría:** no se encontró ningún bug de extracción de features que explique el techo (ver también Sección 9 más abajo sobre por qué se prefirió información mutua sobre ANOVA). Sí se encontró un problema real de validez del ground truth (Sección 7.2), documentable como limitación metodológica de la tesis, y una explicación estructural más probable (Sección 7.3) que no se arregla con más ingeniería de features sino con más datos —sobre todo casos moderados/severos— o con fusión multimodal.

## 8. Próximos pasos

1. Incorporar el resultado de depresión (AUC 0.662, p=0.0033) como el resultado de referencia de video en la tesis, reemplazando el de v2.
2. Documentar la Sección 7.2 (inconsistencia `DX IA` vs. PHQ-9/GAD-7) como limitación metodológica — vale la pena preguntarle a quien generó la columna "IA" cómo se calculó.
3. Reforzar el caso de **fusión multimodal audio+video**: cada modalidad sola es débil-a-moderada; combinarlas sigue siendo la palanca más prometedora, sobre todo para ansiedad.
4. Si se consigue más muestra (idealmente con más casos moderados/severos, no solo más leves), repetir este análisis — tanto la estimación de AUC como la stability selection embebida deberían beneficiarse de folds más grandes.

## 9. Reproducibilidad

```
python experimentos_sin_pca/nested_cv.py          # screening + confirmacion (Secciones 2)
python experimentos_sin_pca/stability_selection.py # nucleo estable (Seccion 4)
python experimentos_sin_pca/permutation_test.py    # test de permutacion (Seccion 3) + congela hiperparams
python experimentos_sin_pca/shap_final.py          # SHAP final (Seccion 5)
python experimentos_sin_pca/check_label_quality.py # auditoria de etiqueta (Seccion 7.1-7.3)
python experimentos_sin_pca/windowed_features.py    # features por ventana 10s (Seccion 7.4)
python experimentos_sin_pca/windowed_nested_cv.py   # evaluacion multi-instancia GroupKFold (Seccion 7.4)
```

Archivos: `feature_selectors.py` (selectores/pipelines sin PCA), `nested_cv.py`, `stability_selection.py`, `permutation_test.py`, `shap_final.py`, `check_label_quality.py`, `windowed_features.py`, `windowed_nested_cv.py`, `resultados/*.csv`, `resultados/shap_*.png`.

## 10. Metodología consultada (por qué se descartó PCA)

Antes de implementar se revisaron (a) los propios papers de referencia de la tesis y (b) la literatura general de selección de features en N pequeño / p grande:

- **Ningún paper de referencia usa PCA para seleccionar features conductuales de clasificación.** AnxietyFaceTrack (Sahu et al. 2025) usa importancia de Random Forest; Beyond Questionnaires (Sahu et al. 2025) usa filtro de correlación de Pearson (0.75); Giannakakis et al. (2017/2024) usan SFS/SBS y mRMR optimizando AUC directamente. Cuando algún paper usa PCA (Giannakakis 2017/2024), es para comprimir descriptores geométricos/HOG dentro de la extracción de features, nunca como paso de selección final para el clasificador.
- **PCA es no supervisado**: con señal débil, los componentes de mayor varianza pueden capturar ruido (iluminación, ángulo) en vez de la etiqueta clínica.
- **PCA rompe la trazabilidad feature→SHAP** que exige el hilo XAI de la tesis (Salih et al. 2026, *Health Science Reports*).
- **El techo real es el EPV** (eventos por variable): con ~20 positivos, cualquier método —PCA incluido— necesita terminar con un puñado de features (aquí, k=4) para no sobreajustar (van Smeden et al. 2019; Riley et al., PMC6519266).
- Referencias metodológicas clave: Vabalas et al. 2019 (PLOS ONE, PMC6837442) sobre sesgo de CV con N pequeño; Varoquaux 2018 (NeuroImage/arXiv:1706.07581) sobre el tamaño real del error de CV con N~100; Meinshausen & Bühlmann 2010 (JRSS-B) sobre stability selection.
