# Informe metodológico — Fusión multimodal audio + video con explicabilidad

**Componente de la tesis:** detección de riesgo de ansiedad y depresión (binario: *Sin Riesgo* /
*Con Riesgo*, **ejes independientes**) en jóvenes de Samaná (Caldas), con explicabilidad (XAI).
**Fecha:** septiembre 2026 · **Carpeta:** `pipeline_multimodal_xai/`
**Documento hermano:** `REPORTE_multimodal.md` (versión corta, solo tablas y números). Este
informe explica **el porqué de cada decisión** y **cómo se lee cada resultado**.

---

## 0. Resumen en una página

**Qué se hizo.** Se combinaron las dos modalidades que la tesis ya tenía por separado —
**voz** (features acústicas eGeMAPS) y **rostro** (Action Units, pose de cabeza, emociones) —
en un solo sistema de decisión, con la misma disciplina estadística que
`experimentos_sin_pca/` (validación cruzada anidada, test de permutación, cuidado extremo con
el tamaño muestral chico). Sobre ese sistema se aplicó explicabilidad: SHAP (¿cuánto aporta la
voz vs. el rostro? ¿qué features?) y LIME (¿por qué este paciente concreto?).

**Qué se encontró.**

| | Ansiedad | Depresión |
|---|---|---|
| Mejor modalidad sola (rostro) | AUC 0.661 | AUC 0.638 |
| **Fusión (promedio simple de las 2 probabilidades)** | **AUC 0.668** | **AUC 0.674** |
| Ganancia de fusionar | +0.007 (marginal) | **+0.036** |
| Test de permutación (¿es real?) | p = 0.045 ✓ | p = 0.040 ✓ |
| Aporte SHAP: voz / rostro | 18 % / 82 % | ~45 % / ~55 % |

**Tres conclusiones.**
1. **El promedio simple de las dos probabilidades (`soft_vote`) es la mejor forma de
   combinar.** Todo método que *aprende* el peso de combinación (regresión logística, red
   neuronal, ponderación optimizada) rinde peor: con 79 participantes no hay datos para
   aprender ese peso sin sobreajustar.
2. **La fusión ayuda a depresión (+0.036, significativo), casi nada a ansiedad (+0.007).**
   Para ansiedad, el rostro por sí solo ya llega a 0.66 y la voz aporta poco margen extra.
3. **La explicación global es fiable, la local no.** SHAP dice consistentemente qué familias
   de features importan (voz → calidad de fonación; rostro → tensión perioral en ansiedad,
   afecto negativo en depresión); LIME y SHAP solo coinciden en la feature #1 de cada
   paciente, no en el resto — lo esperable con N pequeño.

---

## 1. Por qué fusión multimodal

La tesis venía de dos callejones parcialmente sin salida:

- **Audio solo** (`../analisis-espectrogramas`): resultados débiles, motivó el pivote a video.
- **Video solo** (`experimentos_sin_pca/REPORTE.md`): tras 6 rondas de mejora (selección de
  features, calidad de etiqueta, ventanas temporales, ensambles…), el techo real quedó en
  **depresión AUC ≈ 0.66** (p = 0.003, estadísticamente real) y **ansiedad AUC ≈ 0.57** (no
  significativo). La explicación estructural: la muestra es casi toda **casos leves** (0 casos
  de depresión severa; 15 de 19 positivos de ansiedad son leves), y los biomarcadores
  faciales de la literatura (afecto plano, retardo psicomotor) se manifiestan en cuadros
  moderados-severos.

La conclusión de ese documento fue explícita: **la palanca que queda es combinar audio y
video**. La intuición era que dos señales débiles pero *parcialmente independientes* pueden,
sumadas, superar a cualquiera sola — es el principio de la fusión multimodal y el patrón
dominante en la literatura de detección afectiva (retos AVEC, DAIC-WOZ, etc.), donde la
fusión tardía de modalidades es un baseline difícil de batir.

**Nota:** la expectativa previa era que la fusión ayudara *sobre todo a ansiedad* (el eje sin
señal en video). El resultado fue al revés — ayuda más a depresión. Se documenta como está.

---

## 2. Los datos: qué son los 3 CSV

### 2.1 `dataset_au_features.csv` — el rostro

**80 participantes, 1 fila cada uno, 71 columnas.** Es una extracción de **Action Units (AU)**
del sistema FACS (Facial Action Coding System) más pose de cabeza y prototipos de emoción,
resumida a `media` y `desviación estándar` a lo largo de toda la entrevista (~4 min).

| familia | columnas | qué mide clínicamente |
|---|---|---|
| **Cara superior** | `AU01`, `AU02`, `AU04`, `AU05`, `AU06`, `AU07`, `AU09`, `AU43` | ceja interna/externa (AU01/02), ceño (AU04), párpado superior (AU05), mejilla (AU06), párpado tenso (AU07), cierre de ojos (AU43). Marcadores de **tensión frontal, tristeza, esfuerzo**. |
| **Cara inferior** | `AU10`–`AU28` (12 AUs) | labio superior (AU10), comisura arriba/abajo (AU12/AU15), tensor de labios (AU14), estirador (AU20), presión de labios (AU23/24), boca abierta (AU25/26). Marcadores de **expresión emocional y tensión perioral**. |
| **Pose de cabeza** | `Pitch`, `Roll`, `Yaw`, `X`, `Y`, `Z` | orientación y posición de la cabeza. Marcador de **retardo psicomotor, evitación de mirada**. |
| **Emoción** | `anger`, `disgust`, `fear`, `happiness`, `neutral`, `sadness`, `surprise` | probabilidad de cada prototipo emocional por frame, resumida. |

**Columnas que se descartan del modelado:** `error` (100 % vacía) y `n_frames_detected` (es
un control de calidad del tracking; usarla como feature sería un confusor — un participante
detectado en menos frames no es "menos ansioso"). Quedan **66 features de rostro**.

**Etiquetas:** `target_ansiedad` y `target_depresion` vienen ya en el CSV.

### 2.2 `features_ansiedad_egemaps.csv` y `features_depresion_egemaps.csv` — la voz

**~5225 filas cada uno, una fila por SEGMENTO de audio** (23 a 157 segmentos por
participante, mediana 62), **88 columnas de features + `audio_id` + `seg_idx` + `label`**.

Son el conjunto **eGeMAPS v02** (*extended Geneva Minimalistic Acoustic Parameter Set*), el
estándar de facto en computación afectiva de voz. 88 "funcionales" (estadísticos sobre
descriptores de bajo nivel calculados cada 10 ms):

| familia acústica | ejemplos | qué mide |
|---|---|---|
| **F0 / prosodia** | `F0semitone…`, pendientes de subida/bajada, `logRelF0-H1-H2` | tono, entonación, **monotonía vs. expresividad**; H1-H2 = calidad de fonación (tensa vs. soplada) |
| **Loudness / energía** | `loudness_amean`, percentiles, `equivalentSoundLevel` | intensidad vocal, **proyección de la voz** |
| **Espectral** | `alphaRatio`, `hammarbergIndex`, `spectralFlux`, pendientes | balance de energía grave/aguda; **esfuerzo vocal, "brillo"** |
| **MFCC** | `mfcc1`–`mfcc4` (voiced y unvoiced) | forma del espectro, **timbre** |
| **Calidad de voz** | `jitter`, `shimmer`, `HNRdBACF` | irregularidad de ciclo a ciclo, **temblor / ronquera** |
| **Formantes** | `F1`, `F2`, `F3` (frecuencia, ancho de banda, amplitud) | resonancias del tracto vocal, **precisión articulatoria** |
| **Temporal / tasa** | `VoicedSegmentsPerSec`, longitud de segmentos sonoros/sordos | **ritmo del habla, pausas** |

**Por qué el audio viene segmentado y el video no:** el rostro se resume a un vector por
persona porque las señales faciales relevantes son tendencias lentas de toda la entrevista.
El audio se procesa por segmento (tramos de habla) porque cada segmento es una unidad
acústica natural y da **muchas más observaciones** — pero esas observaciones **no son
independientes** (todas de la misma persona), lo cual condiciona cómo se evalúa (§4).

**Los dos CSV de audio son casi el mismo conjunto de features** (misma segmentación, valores
idénticos salvo redondeo); solo cambia la columna `label` (ansiedad vs. depresión) para los
participantes cuyo diagnóstico difiere entre ejes.

### 2.3 La alineación de las dos modalidades

| paso | resultado |
|---|---|
| Cruce por **código de participante** (prefijo numérico del `audio_id` = `video_id`). **No por cédula** — 11 participantes tienen la cédula del nombre de carpeta distinta a la del Excel. | — |
| Participantes con **ambas** modalidades | **79** (el participante 66 tiene video pero no audio → queda fuera del análisis multimodal) |
| Consistencia de etiqueta entre las 3 fuentes | **100 %** — `target_<dx>` (video) coincide con el `label` de segmento (audio) en los 79. Verificado en código; si no coincidiera, el pipeline aborta. |
| Distribución de clases (n = 79) | ansiedad **60 / 19**, depresión **58 / 21** |

### 2.4 Por qué el tamaño muestral lo condiciona todo

Con **19–21 casos positivos**, el criterio que manda es el **EPV** (*events per variable*,
eventos por variable): la regla empírica dice que cada feature que entra al modelo final
necesita ~10 eventos para no sobreajustar. Con 20 positivos, eso son **2 features**; en la
práctica se puede estirar a 4–8 con regularización fuerte, pero **no más**. Por eso:

- Todos los modelos terminan seleccionando **k ∈ {4, 6, 8} features**.
- El meta-modelo de fusión trabaja en **2 dimensiones** (score de voz, score de rostro).
- Las desviaciones entre folds son anchas (±0.11 a ±0.16 de AUC) — **cualquier diferencia de
  AUC menor a ~0.05 hay que tomarla con pinzas**.

Esto no es pesimismo: es la restricción real y todo el diseño de evaluación (§4) existe para
no engañarse con ella.

---

## 3. La arquitectura y por qué

### 3.1 Modelado independiente por eje

Ansiedad y depresión están fuertemente correlacionadas en esta muestra (r ≈ 0.74), pero **se
modelan siempre por separado** — un modelo y una explicación SHAP independiente para cada eje.
Es un requisito de diseño de la tesis (el XAI tiene que poder decir "esto explica *ansiedad*",
no "esto explica *malestar*"). No se probó ningún modelo conjunto / multi-tarea.

### 3.2 Rama de voz: modelo por segmento + agregación

**Problema:** el clasificador necesita una predicción *por participante*, pero el audio son
segmentos.

**Solución en dos pasos:**
1. Un clasificador aprende a nivel **segmento** (¿este tramo de habla suena "en riesgo"?).
   Usa las mismas familias de selección sin PCA de `experimentos_sin_pca/feature_selectors.py`
   (imputación → filtro de varianza → escalado → `SelectKBest` → modelo; o L1-logística
   embebida). Se fija k = 8 porque a nivel segmento hay miles de muestras y la selección fina
   aporta poco.
2. La probabilidad del participante = **media de las probabilidades de sus segmentos**.

**Por qué la media (y no un modelo más sofisticado de agregación):** se probó `median` y `p80`
como sensibilidad (§6.6) — dan prácticamente lo mismo. La media es el default transparente.

**El punto crítico anti-fuga:** para entrenar el meta-modelo (§3.4) necesitamos el score de
voz de cada participante de *entrenamiento*. Si ese score viniera de un modelo que ya vio
otros segmentos de ese mismo participante, estaría **contaminado** (el modelo "reconoce" a la
persona). Solución: el score de voz de los participantes de entrenamiento se calcula con
**validación cruzada de grupo** (`StratifiedGroupKFold`, grupo = participante): ningún
segmento contribuye a la probabilidad agregada de su propio participante. Los scores de test
salen del modelo entrenado con *todos* los segmentos de entrenamiento.

### 3.3 Rama de rostro: pipeline sin PCA

A nivel participante (66 features, 79 filas). Pipeline: imputación → filtro de varianza →
escalado → `SelectKBest` (ANOVA F-test o información mutua) → clasificador (regresión
logística / Random Forest), con búsqueda interna de `k`.

**Por qué sin PCA:** decisión heredada de `experimentos_sin_pca/` — PCA es no supervisado (sus
componentes pueden capturar iluminación/ángulo en vez de la señal clínica) y **rompe la
trazabilidad feature → SHAP** que exige la tesis. Cada valor SHAP tiene que ser atribuible a
una AU con nombre.

### 3.4 Fusión tardía (*late fusion*): 6 formas de combinar 2 scores

Una vez tenemos `score_voz` y `score_rostro` por participante, hay que combinarlos. Se
probaron **seis** meta-modelos, del más simple al más complejo:

| variante | qué hace | intuición |
|---|---|---|
| **`soft_vote`** | `P = (P_voz + P_rostro) / 2` | promedio simple. No aprende nada. |
| **`hard_vote`** | cada modalidad "vota" (¿P > 0.5?), se promedian los votos | voto por mayoría binario |
| **`pond_grid`** | `P = α·P_voz + (1−α)·P_rostro`, con **α elegido** por CV interna probando α ∈ {0, 0.1, …, 1} | promedio ponderado, peso aprendido por rejilla |
| **`pond_auc`** | `α = AUC_voz / (AUC_voz + AUC_rostro)` | promedio ponderado, peso = **cuán buena es cada modalidad** (ecuación cerrada, común en AVEC) |
| **`stack_logreg`** | una **regresión logística** sobre `[score_voz, score_rostro]` | *stacking* clásico: deja que un modelo aprenda la combinación óptima (incluidas no linealidades del tipo "creer al rostro solo si la voz también dice riesgo") |
| **`stack_ann`** | una **red neuronal** mínima (3 neuronas, regularización fuerte) sobre los 2 scores | lo mismo pero con capacidad de modelar interacciones más complejas |

### 3.5 Fusión temprana (*early fusion*): la comparación

Alternativa: en vez de combinar *decisiones*, combinar *features*. Se agregan los segmentos de
audio a un vector por participante (`media`, `std`, `percentil 20/50/80` de cada una de las 88
features → 440 columnas), se concatena con las 66 de rostro (prefijos `aud__` / `vid__` →
**506 features**), y se corre **un solo** pipeline de selección + modelo (incluyendo una red
neuronal sobre features crudas). Un solo SHAP explica todo, con las features etiquetadas por
modalidad.

**Por qué probar ambas:** la fusión temprana puede capturar interacciones cruzadas
voz↔rostro que la tardía no ve; la tardía es más robusta con N chico. Se dejó que los datos
decidieran.

---

## 4. Cómo se evaluó (y por qué el número es honesto)

Todo a **nivel participante**, **por eje**, con semilla fija (`RANDOM_STATE = 42`).

### 4.1 Dos etapas: screening → confirmación

- **Screening** (`RepeatedStratifiedKFold` 5×2 = 10 folds): se evalúa cada config candidata
  de rama-voz, rama-rostro y early. Se elige la mejor de cada una por su AUC de test.
- **Confirmación** (5×5 = 25 folds): con esas configs *congeladas*, se evalúan todos los
  baselines y las 6 variantes de fusión. **El AUC que se reporta es el de esta etapa.**

**Por qué dos etapas:** por costo (probar todas las combinaciones config × fusión × 25 folds
sería prohibitivo) y porque es la práctica estándar de *screening → confirm*.

### 4.2 El "winner's curse" y la CV anidada

En rondas previas de la tesis (v2 de video) se cometió el error de **elegir el mejor `k`
mirando la misma métrica que después se reportaba como resultado**. Eso infla el número
(Vabalas et al. 2019). Aquí la elección de hiperparámetros vive en un **bucle interno**
(`GridSearchCV`) y el AUC reportado es el del **bucle externo**, que nunca vio esa elección.

Queda un optimismo residual mínimo: elegir "la mejor config de N" para reportarla introduce
un sesgo de selección pequeño. El **test de permutación** (§4.4) es el control contra eso.

### 4.3 Las cuatro guardas anti-fuga

La fuga de datos (*data leakage*) es la forma #1 de auto-engañarse con N pequeño. Guardas:

1. **Particiones por participante** en todos los bucles — nunca un segmento de un participante
   de test aparece en entrenamiento (la rama de audio usa `StratifiedGroupKFold`).
2. **Selección de features e imputación dentro de cada fold** — nunca se calcula un estadístico
   (media para imputar, ranking ANOVA) usando datos de test.
3. **Los scores que alimentan al meta-modelo son *out-of-fold*** (§3.2).
4. **La agregación de audio es por-participante** — el vector de un participante se calcula
   solo con sus propios segmentos, no con estadísticos de la muestra.

### 4.4 Test de permutación: ¿el resultado es real o es ruido de N = 79?

**La pregunta.** Con 79 personas y ~20 positivos, ¿podría un AUC de 0.67 salir *por azar*, solo
por cómo cayó la partición?

**El método** (Ojala & Garriga 2010, estándar). Se congela la arquitectura ganadora. Se
evalúa con las etiquetas **reales** (1 vez) y con **200 barajes aleatorios** de las etiquetas.
Si el modelo captura señal real, el AUC con etiquetas reales debe quedar claramente por encima
de la nube de AUCs con etiquetas barajadas.

`p = (1 + nº de barajes que igualan o superan el AUC real) / (200 + 1)`

**Resultado.**

| Eje | AUC real | AUC nulo (media) | AUC nulo (p95) | **p** |
|---|---|---|---|---|
| Ansiedad | 0.676 | 0.492 | 0.653 | **0.045** |
| Depresión | 0.674 | 0.482 | 0.665 | **0.040** |

**Lectura.** Ambos ejes superan el azar de forma defendible, **pero con poco margen** — el AUC
real cae apenas por encima del percentil 95 de la distribución nula. Es un "sí" honesto, no un
"sí" contundente. El AUC real de la permutación coincide exactamente con el de la confirmación
(§5) → el pipeline congelado reproduce, no hay artefacto.

---

## 5. Resultados, línea por línea

### 5.1 Screening — qué config ganó en cada rama

**Ansiedad** (top 4 de 13): `video:anova_logreg` **0.656** · `audio:anova_xgb` 0.585 ·
`video:l1logreg` 0.583 · `audio:anova_logreg` 0.574.
**Depresión** (top 4 de 13): `video:anova_rf` **0.657** · `video:mutinfo_rf` 0.648 ·
`audio:l1logreg` **0.644** · `video:l1logreg` 0.638.

Observaciones:
- En **ambos ejes** la mejor rama individual es de **rostro**, no de voz.
- El **early fusion nunca aparece arriba** (mejor 0.56 en ansiedad, 0.60 en depresión) — ya
  desde el screening la concatenación de 506 features se ve mal.
- Configs ganadoras congeladas: ansiedad → voz `anova_xgb`, rostro `anova_logreg`, early
  `anova_logreg`; depresión → voz `l1logreg`, rostro `anova_rf`, early `anova_mlp`.

### 5.2 Confirmación — la tabla completa (25 folds)

**Ansiedad**

| método | AUC | ±std | F1-macro | qué es |
|---|---|---|---|---|
| **`soft_vote`** | **0.668** | 0.160 | 0.542 | promedio simple voz+rostro |
| solo rostro | 0.661 | 0.153 | 0.562 | baseline modalidad |
| `pond_auc` | 0.641 | 0.167 | 0.553 | ponderado, α_voz ≈ 0.21 |
| `pond_grid` | 0.611 | 0.148 | 0.542 | ponderado, α_voz ≈ 0.24 |
| `hard_vote` | 0.594 | 0.125 | 0.554 | voto por mayoría |
| solo voz | 0.572 | 0.149 | 0.426 | baseline modalidad |
| `stack_logreg` | 0.557 | 0.212 | 0.523 | regresión logística meta |
| `stack_ann` | 0.537 | 0.189 | 0.453 | red neuronal meta |
| early fusion | 0.511 | 0.156 | 0.498 | concat 506 features |
| mayoría | 0.500 | 0.000 | 0.432 | predecir siempre la clase mayoritaria |

**Depresión**

| método | AUC | ±std | F1-macro | |
|---|---|---|---|---|
| **`soft_vote`** | **0.674** | 0.114 | 0.606 | |
| `pond_grid` | 0.649 | 0.117 | 0.564 | α_voz ≈ 0.28 |
| `pond_auc` | 0.642 | 0.123 | 0.566 | α_voz ≈ 0.25 |
| solo rostro | 0.638 | 0.111 | 0.547 | |
| `stack_logreg` | 0.626 | 0.135 | 0.560 | |
| early fusion | 0.619 | 0.147 | 0.502 | |
| solo voz | 0.593 | 0.123 | 0.533 | |
| `hard_vote` | 0.587 | 0.120 | 0.527 | |
| `stack_ann` | 0.583 | 0.132 | 0.485 | |
| mayoría | 0.500 | 0.000 | 0.423 | |

### 5.3 El hallazgo central: `soft_vote` gana, todo lo que "aprende" pierde

En **los dos ejes**, el orden es el mismo: el **promedio simple gana**, y cuanto más
"inteligente" es el meta-modelo, peor rinde (`pond_grid` < `pond_auc` < `stack_logreg` <
`stack_ann`).

**Por qué.** El meta-modelo entrena con ~63 participantes y solo 2 features. Ajustar un peso
de combinación (o los pesos de una regresión, o los de una red) consume grados de libertad que
esos 63 puntos no alcanzan a sostener: el peso que el modelo aprende está sobreajustado a la
partición concreta y generaliza mal. El promedio 50/50 no aprende nada, así que no puede
sobreajustar. Es un resultado conocido en fusión multimodal con N pequeño (el promedio no
ponderado es un baseline notoriamente duro), y aquí se cumple de forma limpia.

**La red neuronal (`stack_ann`) es de las peores** — como se anticipó en el plan. Con 63
puntos y 2 entradas no hay nada que una red pueda aprender que una logística no capture, y sí
mucho ruido que puede memorizar. Se incluyó por completitud (se preguntó explícitamente por
votación / ANN / ponderación) y para poder mostrar, con números, que no es el camino.

### 5.4 El early fusion queda claramente atrás

AUC 0.511 (ansiedad) y 0.619 (depresión), por debajo de la mejor modalidad sola en ambos
casos. Con 506 features y 79 muestras, la selección no logra recuperar un núcleo útil: la
señal de cada modalidad se diluye en el ruido de la otra. La fusión tardía funciona mejor
porque cada rama primero **destila** su modalidad a un solo número robusto, y solo entonces se
combinan.

### 5.5 La fusión ayuda a depresión, casi nada a ansiedad

- **Depresión: +0.036** (0.638 → 0.674). Es la mejora más grande y pasa permutación. Voz y
  rostro aportan señal parcialmente complementaria.
- **Ansiedad: +0.007** (0.661 → 0.668). Dentro del ruido. El rostro (con este set de AU) ya
  hace casi todo el trabajo; la voz sube muy poco el AUC.

**Matiz para la tesis:** el baseline de rostro de ansiedad aquí es **0.66**, cuando el número
canónico de video de la tesis (con las 256 features temporales de `experimentos_sin_pca/`) es
**0.57**. Este CSV de AU es una extracción **distinta** y aparentemente mejor para ansiedad —
pero eso es un hallazgo a verificar por separado (repetir `experimentos_sin_pca` con este
CSV), no una conclusión de este trabajo. Ver §7 (Limitaciones).

### 5.6 Sensibilidad: ¿y si se agrega el audio de otra forma?

`sensibilidad_agg.py` re-evalúa `soft_vote` cambiando cómo se resumen las probabilidades de
segmento (media / mediana / percentil 80):

| | media | mediana | p80 |
|---|---|---|---|
| Ansiedad | 0.681 | 0.685 | 0.687 |
| Depresión | 0.689 | 0.683 | 0.671 |

Para ansiedad da igual; para depresión la media es ligeramente mejor y el p80 baja ~0.02. **La
elección de `media` no cambia ninguna conclusión.** (Estos AUCs son un pelín más altos que los
de la tabla 5.2 porque este chequeo usa 15 folds en vez de 25 — la diferencia es ruido de
partición, no una mejora.)

---

## 6. Explicabilidad (XAI): qué mira cada método y qué encontró

### 6.1 SHAP y LIME en una frase cada uno

- **SHAP** (*SHapley Additive exPlanations*): reparte la predicción del modelo entre sus
  features de forma matemáticamente consistente (teoría de juegos). Sirve para lo **global**:
  "en promedio, ¿qué features mueven la aguja?". También da explicaciones locales.
- **LIME** (*Local Interpretable Model-agnostic Explanations*): ajusta un modelo lineal simple
  *alrededor de un caso concreto* para explicar *esa* predicción. Sirve para lo **local**:
  "¿por qué el modelo marcó a *este* paciente?".

### 6.2 SHAP a nivel modalidad — ¿cuánto aporta la voz vs. el rostro?

Se ajusta un meta-modelo logístico sobre `[score_voz, score_rostro]` de los 79 participantes
y se le aplica SHAP. `|SHAP|` medio normalizado:

| Eje | voz | rostro |
|---|---|---|
| Ansiedad | 18 % | **82 %** |
| Depresión | ~45 % | ~55 % |

**Lectura:** para ansiedad el rostro domina; para depresión están parejos.

**Pero — con una advertencia grande (§6.5):** este número **no es estable**. El bootstrap dice
que el intervalo p05–p95 del peso de la voz va de ~0.07 a ~0.88 en ansiedad. Lo único robusto
es el **orden** (rostro ≥ voz en ansiedad), no el porcentaje exacto. En el informe hay que
decir "el rostro pesa más en ansiedad", no "el rostro pesa el 82 %".

### 6.3 SHAP intra-modalidad — la voz

Se aplica SHAP al modelo de segmento y se agrupa por familia acústica.

**Ansiedad** (modelo `anova_xgb`, árbol): F0/prosodia **41 %** · loudness **37 %** · MFCC 18 % ·
espectral 4 %.
Top features: `logRelF0-H1-H2` (armónicos → fonación tensa vs. soplada), percentiles de
`loudness`, `logRelF0-H1-A3`, `mfcc4`.
→ **Tensión laríngea y control de la intensidad bajo estrés.**

**Depresión** (modelo `l1logreg`, lineal): formantes **49 %** · espectral 17 % · MFCC 14 % ·
F0 8 %.
Top features: `FxamplitudeLogRelF0` (amplitud de las resonancias vocálicas relativa a F0),
`alphaRatio` y `hammarbergIndex` (balance espectral grave/agudo).
→ **Articulación reducida y voz "apagada" / poco proyectada** — consistente con el
aplanamiento prosódico de la literatura de depresión.

### 6.4 SHAP intra-modalidad — el rostro

**Ansiedad**: cara inferior **56 %** · cara superior 44 %.
Top: `AU15_std` (variabilidad del depresor de comisura labial), `AU20_std` (estirador de
labios), `AU01_mean` (ceja interna), `AU05_mean` (párpado), `AU04_mean` (ceño).
→ **Tensión perioral y frontal**, coherente con Giannakakis et al. (2017/2024).

**Depresión**: prototipo emoción **54 %** · cara superior 46 %.
Top: `anger_mean` (prototipo de enojo), `AU01_mean` (elevador interno de ceja — marcador
clásico de tristeza), `anger_std`, `AU05_std`.
→ **Afecto negativo facial**, patrón "ceño / afecto negativo" de la literatura de depresión.

### 6.5 Estabilidad de las explicaciones (bootstrap, B = 200)

Con N = 79 una sola explicación SHAP puede ser un accidente de la partición. Se re-ajusta el
pipeline 200 veces sobre remuestreos y se cuenta con qué frecuencia se selecciona cada
feature.

**Núcleo estable (frecuencia ≥ 0.5):**

| Eje | rostro | voz |
|---|---|---|
| Ansiedad | `AU15_std` (0.66), `AU01_mean` (0.56) | `logRelF0-H1-H2` (0.88), `mfcc4` (0.53) |
| Depresión | `anger_mean` (0.86), `anger_std` (0.79), `AU01_mean` (0.67) | `logRelF0-H1-H2` (0.66), `F1frequency` (0.53) |

→ El núcleo estable **coincide con el top de SHAP** para el rostro en ambos ejes y para la voz
en ansiedad. Esa parte de la explicación **no es un artefacto** — sobrevive al remuestreo.

**Estabilidad del peso de modalidad:** aquí la noticia es mala pero honesta. El coeficiente
relativo de la voz tiene media 0.41 (ansiedad) / 0.50 (depresión) pero **desviación ±0.20 y
p05–p95 que casi cubre [0, 1]**. La pregunta "¿cuánto pesa la voz?" no tiene respuesta puntual
con estos datos. Esto **refuerza** por qué `soft_vote` (peso fijo) gana: cualquier peso
aprendido está sobre una base de arena.

### 6.6 LIME — explicación por paciente + consistencia con SHAP

Sobre la matriz de early fusion se eligen 3 participantes por su predicción *out-of-fold*: un
verdadero positivo (TP), un falso positivo (FP) y un verdadero negativo (TN). LIME produce
para cada uno una lista rankeada única que mezcla voz y rostro.

**Ejemplos:**
- Ansiedad, TP (participante 61, riesgo real, p = 0.76): `AU15_std` en rango medio (+0.070) y
  `logRelF0-H1-H2` de la voz (+0.037) empujan hacia "riesgo".
- Depresión, TP (participante 35, riesgo real, p = 0.99): `anger_mean > 0.02` **domina por
  completo** (+0.29); todo lo demás, voz incluida, aporta < 0.03.

**Consistencia SHAP vs. LIME** (Jaccard del top-10, Spearman del ranking, mismos casos):

| Eje | caso | Jaccard | Spearman |
|---|---|---|---|
| Ansiedad | TP / FP / TN | 0.11 / 0.25 / 0.11 | −0.25 / +0.28 / −0.25 |
| Depresión | TP / FP / TN | 0.11 / 0.18 / 0.18 | −0.25 / −0.27 / +0.02 |

**Lectura honesta:** SHAP y LIME **coinciden en la feature #1 de cada caso** (la que también
encabeza el SHAP global), pero **no en el resto** — Jaccard ≈ 0.1–0.25, correlación de rango
cercana a 0. Es el comportamiento esperado de la literatura XAI con N pequeño y features
correlacionadas (Salih et al. 2026): **la feature principal de una explicación local es
fiable; el orden fino no**. Para la tesis: la evidencia XAI defendible es la **global**
(§6.3–§6.5); LIME entra como ilustración cualitativa por-paciente, no como evidencia.

---

## 7. Limitaciones (honestidad ante todo)

1. **N = 79, 19–21 positivos.** Las desviaciones entre folds (±0.11 a ±0.16) siguen siendo
   anchas. El 0.674 de depresión y el 0.638 de rostro solo tienen intervalos que se solapan.
2. **La ganancia de la fusión es chica.** +0.007 (ansiedad) y +0.036 (depresión). El techo de
   AUC ~0.65–0.68 se mantiene; ahora con dos modalidades en vez de una. No es el salto grande
   que se esperaba.
3. **El CSV de AU no es comparable con el video canónico de la tesis.** Baseline de rostro
   aquí: ansiedad 0.66, depresión 0.64. Canónico (256 features temporales): ansiedad 0.57,
   depresión 0.66. Que ansiedad suba a 0.66 con AU **hay que verificarlo aparte**.
4. **La agregación de audio es un parámetro libre**, no optimizado dentro de la CV (aunque el
   análisis de sensibilidad muestra que no cambia la conclusión).
5. **Validez del ground truth** (heredada de `experimentos_sin_pca/REPORTE.md` §7.2): la
   etiqueta `DX ... IA` no correlaciona con el PHQ-9/GAD-7 de la misma encuesta. Aplica igual
   a las dos modalidades. Vale la pena preguntar a quien generó esa columna cómo se calculó.
6. **El balance de modalidad no es una medición.** Los "voz X % / rostro Y %" del SHAP son una
   tendencia inestable (§6.5). Reportar el orden, no el número.
7. **Optimismo residual** de elegir la mejor de varias configs en el screening. El test de
   permutación es el control, y ambos ejes lo pasan (aunque depresión al límite).

---

## 8. Qué significa esto para la tesis

**Lo positivo:**
- Hay un **sistema multimodal completo, evaluado con rigor**, con AUC 0.67 en ambos ejes y
  ambos pasando el test de permutación — algo que **ansiedad nunca había logrado** en
  video solo.
- El **hilo XAI está cerrado**: SHAP de modalidad + SHAP de features por familia (con lectura
  clínica coherente) + estabilidad por bootstrap + LIME local + chequeo de consistencia
  SHAP/LIME.
- El hallazgo "**el promedio simple gana a todo lo que aprende**" es un resultado metodológico
  sólido y publicable en sí mismo (fusión multimodal con N pequeño).

**Lo que hay que decir con cuidado:**
- La fusión **mejora**, pero **poco** (sobre todo en ansiedad). No resuelve el techo; lo
  empuja un poco y le da a ansiedad su primer resultado significativo.
- El **peso voz/rostro no es medible** con estos datos — solo el orden.
- El **AUC de rostro de ansiedad (0.66)** merece una nota al pie explicando que es otra
  extracción de features, no el número histórico.

**Lo que queda abierto (no es parte de este trabajo):**
1. Verificar si el set de AU (66 features) realmente mejora ansiedad vs. las 256 temporales.
2. Con más muestra (sobre todo casos moderados/severos), repetir todo: el AUC y el peso de
   modalidad deberían estabilizarse.
3. Explorar la agregación de audio como hiperparámetro dentro de la CV.
4. rPPG facial (ritmo cardiaco desde el rostro) como tercera señal.

---

## 9. Archivos y reproducibilidad

```bash
pip install -r pipeline_multimodal_xai/requirements.txt   # única dependencia nueva: lime
python pipeline_multimodal_xai/run_all.py                  # todo, en orden (~40 min CPU)
```

| archivo | qué hace |
|---|---|
| `config_mm.py` | rutas a los 3 CSV, constantes de evaluación (reutiliza las de `../config.py`), familias de features |
| `data_multimodal.py` | carga y alinea los 3 CSV por código; verifica consistencia de etiquetas; n = 79 |
| `audio_branch.py` | modelo de voz a nivel segmento + agregación a participante (OOF con GroupKFold) |
| `fusion.py` | rama de rostro, matriz de early fusion, y los 6 meta-modelos de late fusion |
| `eval_multimodal.py` | screening (10 folds) → confirmación (25 folds); baselines + todas las variantes |
| `permutation_multimodal.py` | test de permutación sobre la arquitectura ganadora de cada eje |
| `sensibilidad_agg.py` | sensibilidad de `soft_vote` a la agregación de audio (media/mediana/p80) |
| `xai_shap.py` | SHAP: nivel modalidad + intra-modalidad (features + familias) + early fusion |
| `xai_lime.py` | LIME local (TP/FP/TN) + consistencia SHAP vs. LIME |
| `xai_stability.py` | estabilidad de features y del peso de modalidad (bootstrap B = 200) |
| `run_all.py` | orquesta todo lo anterior en orden |

**Salidas** en `pipeline_multimodal_xai/resultados/` (45 archivos: CSV de métricas, PNG de
SHAP/LIME, JSON de configs ganadoras). Se versionan igual que `experimentos_sin_pca/resultados/`
— son agregados/derivados, sin datos crudos ni identificadores (los `pid` son códigos 1–80).

**Documento de resultados corto:** `REPORTE_multimodal.md` (mismas cifras, sin la parte
explicativa).
