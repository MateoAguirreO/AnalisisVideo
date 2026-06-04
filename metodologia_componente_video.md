# Componente multimodal de video: enfoque metodológico (síntesis)

**Tema:** Extensión multimodal (audio + video) del enfoque de Inteligencia Artificial Explicable (XAI) para la detección de ansiedad y depresión en jóvenes de Samaná, Caldas, a partir de biomarcadores conductuales extraídos de video facial.

> Documento de soporte metodológico del componente de video. Se articula con la propuesta base (XAI sobre biomarcadores acústicos) y conserva su hilo central: **capacidad predictiva con explicaciones clínicamente útiles**. El estado del arte verificado que sustenta este documento se encuentra en `dossier_literatura_video.md` (112 trabajos) y `top10_papers.md`.

---

## 1. Introducción y motivación

La línea inicial del proyecto abordó la detección de ansiedad y depresión exclusivamente desde **biomarcadores acústicos** de voz. Si bien la voz es una fuente no invasiva y de bajo costo, los resultados sobre la muestra local fueron limitados, en parte por el tamaño reducido del conjunto (≈80 participantes), la heterogeneidad de las condiciones de captura y la pérdida de información que ocurre al descartar el canal visual de las entrevistas.

La evidencia reciente muestra que la integración de **señales conductuales de video** —expresión facial, comportamiento ocular, movimiento de cabeza y micro-gestos corporales— mejora de forma consistente el desempeño respecto a enfoques unimodales de solo audio, y aporta marcadores con interpretación clínica directa (afecto plano, retardo psicomotor, aversión de la mirada) [1][2][3]. Dado que las muestras de campo de Samaná ya contienen **video frontal del rostro sincronizado con el audio**, la migración a un esquema **multimodal** no requiere recolección adicional y aprovecha información ya disponible.

Este componente conserva el objetivo metodológico de la propuesta original: no basta con predecir una etiqueta de riesgo, sino **justificar la decisión** con evidencia trazable y comprensible para profesionales de salud del territorio.

---

## 2. Planteamiento del componente

### 2.1 Problema

Los modelos de video para salud mental se han construido mayoritariamente con: (i) corpus en inglés y poblaciones no latinoamericanas (DAIC-WOZ, E-DAIC) [4][5]; (ii) arquitecturas profundas de extremo a extremo poco interpretables y exigentes en datos; y (iii) protocolos de captura que no corresponden al video de campo disponible (cámaras de alta velocidad para microexpresiones, eye-trackers infrarrojos para movimiento ocular fino). Trasladar directamente esos modelos al caso de Samaná no es viable ni metodológica ni clínicamente.

### 2.2 Pregunta del componente

¿En qué medida la incorporación de biomarcadores conductuales de **video facial**, fusionados con los biomarcadores acústicos y acompañados de explicaciones (SHAP/LIME), mejora la interpretabilidad y el desempeño de la detección de riesgo de ansiedad y depresión respecto al enfoque de solo audio, bajo las restricciones de captura de la muestra local (video frontal, 1080×1920, 30 fps)?

### 2.3 Objetivo del componente

Diseñar, implementar y evaluar un **pipeline multimodal explicable** que (a) extraiga biomarcadores de video factibles a 30 fps, (b) los combine con el canal acústico, (c) clasifique riesgo binario de ansiedad y de depresión, y (d) genere explicaciones por característica clínicamente plausibles.

---

## 3. Marco teórico: biomarcadores conductuales de video

La detección de afecto desde video se fundamenta en que ciertos patrones conductuales reflejan cambios emocionales y psicofisiológicos observables externamente. Se agrupan en cuatro familias:

- **Faciales / afecto.** Las unidades de acción facial (FACS/AUs) y su dinámica temporal codifican la expresión emocional. En depresión se reporta **afecto aplanado** (menor sonrisa, menor expresividad global) y mayor actividad de acciones de tristeza/contempt; en ansiedad, mayor tensión perioral y de ceño [1][6][7].
- **Oculares (gruesos).** La **tasa de parpadeo**, la apertura ocular y la **aversión de la mirada** (mirar fuera del interlocutor/cámara) son marcadores de ansiedad social y de estrés, observables sin hardware especializado [8][9].
- **Cefálicos.** La reducción de movimiento de cabeza y patrones de pose (yaw/pitch/roll) se asocian a **retardo psicomotor** en depresión [10].
- **Corporales (micro-gestos).** Movimiento de cuerpo y manos, autocontacto (mano-a-cara) y fidgeting aportan señales complementarias de estado emocional [3].

Cada familia es **extraíble a 30 fps**, a diferencia de las microexpresiones faciales puras (requieren 100–200 fps) y del movimiento ocular fino —sacadas, microsacadas, pupilometría— (requiere eye-tracker infrarrojo a 120–250 Hz), que quedan fuera del alcance por restricción de la captura disponible.

---

## 4. Estado del arte que sustenta el enfoque

| Autor (Año) | Enfoque | Aporte que se incorpora | Ref. |
|---|---|---|---|
| Giannakakis et al. (2017) | Cues faciales de estrés/ansiedad | Vocabulario de marcadores (parpadeo, mirada, actividad de boca, movimiento de cabeza) | [8] |
| Mahayossanunt et al. (2023) | AU/gaze/head + LSTM + integrated gradients | Demostración de XAI sobre video de entrevista (el match más cercano al objetivo) | [1] |
| Giannakakis et al. (2024) | AUs + XAI | Identificación, vía XAI, de qué AUs explican el estrés | [9] |
| Guo et al. (2022) | Visual-only AU/pose/gaze + atención | Conjunto de características visuales discriminativas | [6] |
| Sahu et al. (2025) — *Beyond Questionnaires* | AUs + head/body pose + gaze (ML/DL) | Plantilla de características para ansiedad social | [2] |
| Sahu et al. (2025) — *AnxietyFaceTrack* | Features faciales de smartphone + Random Forest | Receta interpretable con hardware equivalente (cámara de celular) | [3] |
| Pampouchidou et al. (2020) | Video facial, cross-corpus | Predicción conjunta de **ansiedad y depresión** y generalización | [7] |
| Gimeno-Gómez et al. (2024) | Cues no verbales in-the-wild (+código) | Validación de landmarks/gaze/blink en video real ruidoso | [11] |

---

## 5. Decisión de diseño: síntesis frente a reimplementación de un único trabajo

El componente **no reproduce un solo artículo**, sino que **sintetiza** los hallazgos más transferibles del estado del arte bajo las restricciones del caso local. La receta global —*video de cámara común → características conductuales interpretables → modelos clásicos → importancia explicable*— corresponde de forma más cercana a **AnxietyFaceTrack** [3]; el conjunto de características proviene de [8][2][6]; y la capa de explicabilidad sigue [9][1].

**Justificación de no usar un modelo profundo de extremo a extremo** (p. ej. LSTM/3D-CNN, como en [1]): con **N≈80** participantes, los modelos secuenciales profundos tienden al **sobreajuste** y comprometen la trazabilidad de las explicaciones. La estrategia de **agregar características por participante + modelos interpretables + SHAP/LIME** es más robusta para muestra pequeña, preserva el hilo XAI de la propuesta y mantiene comparabilidad directa con el componente de audio. Los modelos profundos quedan como **línea de comparación opcional** y trabajo futuro con más datos.

---

## 6. Marco metodológico

### 6.1 Datos

- **Muestras:** ~80 participantes con video frontal de rostro sincronizado con audio (1 clip por participante; ~4.4 min; H.264 1080×1920 vertical, **30 fps**; audio 48 kHz).
- **Etiquetas:** clasificación **binaria de riesgo** derivada de las columnas clínicas `DX DEPRESIÓN IA` y `DX ANSIEDAD IA` (Sin Riesgo = 0; Riesgo leve/moderado/alto = 1). Distribución: depresión ≈59/21, ansiedad ≈61/19 (5 sin etiqueta). Es un problema **desbalanceado y de muestra pequeña**, lo que condiciona las decisiones de modelado.

### 6.2 Viabilidad técnica (qué es realizable a 30 fps)

| Biomarcador | Viable | Justificación |
|---|---|---|
| AU-like faciales + dinámica, sonrisa, ceño, expresividad | Sí | Blendshapes por frame; base de la literatura clínica de video |
| Parpadeo, apertura ocular, aversión de mirada (gruesa) | Sí | Estimables por frame sin hardware especializado |
| Pose y movimiento de cabeza | Sí | Matriz de transformación facial → yaw/pitch/roll |
| Micro-gestos corporales (cuerpo/manos) | Sí | Landmarks de pose y mano a 30 fps |
| Microexpresiones puras (apex 40–200 ms) | No | Requieren 100–200 fps (CASME II, SAMM, SMIC) |
| Movimiento ocular fino (sacadas, pupilometría) | No | Requiere eye-tracker IR 120–250 Hz |
| rPPG (pulso desde rostro) | Parcial | Posible a 30 fps pero ruidoso; exploratorio |

### 6.3 Extracción de características (`src/extract_features.py`)

Se emplea la **MediaPipe Tasks API** con tres modelos:

- **FaceLandmarker** → 52 *blendshapes* (señales tipo AU: `eyeBlink`, `mouthSmile`, `mouthFrown`, `browDown`, `jawOpen`, `eyeLookOut/Down`, etc.) y **matriz de transformación facial** (pose de cabeza).
- **PoseLandmarker** → landmarks de cuerpo (energía de movimiento de hombros/brazos).
- **HandLandmarker** → landmarks de manos (movimiento, eventos mano-a-cara).

El video se submuestrea a 1 de cada 3 frames (~10 fps efectivos), suficiente para los biomarcadores objetivo y mucho más eficiente. Por cada participante se agregan **descriptores estadísticos** (media, desviación, fracciones y tasas) en cinco familias:

- **Ocular:** `blink_rate_min`, `blink_*`, `eye_squint_*`, `eye_wide_*`, `gaze_out_*`, `gaze_down_*`, `gaze_aversion_frac`.
- **Afecto facial:** `smile_*`, `frown_*`, `brow_down_*`, `brow_inner_up_*`, `cheek_squint_*`, `nose_sneer_*`, `expresividad_*` (proxy de **afecto plano**).
- **Boca / habla:** `jaw_open_*`, `mouth_press_*`, `mouth_open_frac`.
- **Cabeza:** `yaw_*`, `pitch_*`, `roll_*`, `head_motion_*`, `head_still_frac` (proxy de **retardo psicomotor**).
- **Corporal:** `body_motion_*`, `hand_motion_*`, `hand_visible_frac`, `hand_to_face_rate`.

Cada característica es **clínicamente nombrable**, condición necesaria para la explicabilidad posterior. Las distancias se normalizan por escala facial (distancia interocular) y corporal (ancho de hombros) para robustez ante la distancia a la cámara.

### 6.4 Construcción del conjunto (`src/build_dataset.py`)

Las características de video se unen con las etiquetas por participante, generando un conjunto por diagnóstico (`dataset_ansiedad.csv`, `dataset_depresion.csv`). La unidad de análisis es el **participante** (no el frame), lo que evita fuga de información entre particiones.

### 6.5 Modelado (`src/train_xai.py`)

Modelos interpretables o de baja complejidad, con imputación por mediana y balanceo de clases:

- **Random Forest** (400 árboles, `class_weight=balanced`).
- **XGBoost** (300 árboles, profundidad 3, `lr=0.05`).
- **SVM-RBF** (con estandarización, `class_weight=balanced`).

### 6.6 Explicabilidad (XAI)

Se aplica **SHAP** (TreeExplainer sobre Random Forest) para obtener importancia **global** y **por característica**, con visualizaciones (summary plot) y ranking exportable. La interpretación se reporta en lenguaje clínico (p. ej. "menor expresividad y menor movimiento de cabeza incrementan el riesgo de depresión"), de forma análoga a SHAP/LIME en el componente acústico, garantizando un **discurso explicativo unificado** entre modalidades.

### 6.7 Fusión multimodal

Se adopta inicialmente **fusión tardía** (*late fusion*): se entrena un clasificador por modalidad (audio, video) y se combinan sus puntuaciones (promedio ponderado o meta-clasificador). Esta estrategia (i) reutiliza el pipeline acústico existente, (ii) es robusta con pocos datos y (iii) preserva explicaciones separadas por modalidad. Como segunda iteración se contempla **fusión por atención** sobre características, cuyos pesos aportan señal interpretable adicional [1][6].

### 6.8 Protocolo de validación

Réplica del protocolo del componente de audio para garantizar comparabilidad:

- **RepeatedStratifiedKFold 5×5** (25 particiones), estratificado y a **nivel de participante**.
- **Métricas:** accuracy, precision/recall/F1 macro y **ROC-AUC**; matriz de confusión y comportamiento por subgrupo.
- **Comparaciones:** (a) solo audio vs. solo video vs. fusión; (b) frente a baselines y al estado del arte reportado.
- **Criterio de éxito orientador:** la fusión supera al mejor unimodal en F1-macro y AUC, con explicaciones clínicamente plausibles.

---

## 7. Trazabilidad: de qué trabajo proviene cada pieza

| Pieza del pipeline | Origen en la literatura |
|---|---|
| Receta global (video común → features → RF → importancia) | AnxietyFaceTrack [3] |
| Conjunto de características (AU/gaze/head/body) | Beyond Questionnaires [2], Guo [6] |
| Vocabulario de cues (parpadeo/mirada/boca/cabeza) | Giannakakis 2017 [8] |
| Explicabilidad por característica (SHAP) | Giannakakis 2024 [9], Mahayossanunt [1] |
| Objetivo conjunto ansiedad + depresión | Pompouchidou [7] |
| Validación de cues en video real | Gimeno-Gómez [11] |

---

## 8. Consideraciones éticas

El **rostro constituye dato biométrico**. Antes del procesamiento de video se debe verificar que los consentimientos informados del trabajo de campo de Samaná cubren explícitamente el **análisis facial/video** (no solo la voz); de lo contrario, se requiere una **adenda al comité de ética**. Las salidas derivadas (características por participante, reportes) se tratan como datos sensibles y se excluyen del control de versiones público.

---

## 9. Productos del componente

| Producto | Archivo |
|---|---|
| Estado del arte verificado (112 trabajos) | `dossier_literatura_video.md` |
| Selección priorizada | `top10_papers.md` |
| Configuración y rutas | `config.py` |
| Carga de etiquetas clínicas | `src/labels.py` |
| Extracción de biomarcadores de video | `src/extract_features.py` |
| Construcción de conjuntos | `src/build_dataset.py` |
| Modelado + XAI | `src/train_xai.py` |
| Métricas y explicaciones | `resultados/` |

---

## 10. Referencias

[1] Y. Mahayossanunt, N. Nupairoj, S. Hemrungrojn, P. Vateekul, "Explainable Depression Detection Based on Facial Expression Using LSTM on Attentional Intermediate Feature Fusion with Label Smoothing," *Sensors*, 23(23):9402, 2023. https://doi.org/10.3390/s23239402

[2] N. K. Sahu, N. S. Harshit, R. Uikey, H. R. Lone, "Beyond Questionnaires: Video Analysis for Social Anxiety Detection," arXiv:2501.05461, 2025. https://arxiv.org/abs/2501.05461

[3] N. K. Sahu, S. Gupta, H. R. Lone, "AnxietyFaceTrack: A Smartphone-Based Non-Intrusive Approach for Detecting Social Anxiety Using Facial Features," arXiv:2502.16106, 2025. https://arxiv.org/abs/2502.16106

[4] J. Gratch et al., "The Distress Analysis Interview Corpus of human and computer interviews," *LREC*, pp. 3123–3128, 2014. https://aclanthology.org/L14-1421/

[5] F. Ringeval et al., "AVEC 2019 Workshop and Challenge: State-of-Mind, Detecting Depression with AI, and Cross-Cultural Affect Recognition," *AVEC '19*, 2019. https://doi.org/10.1145/3347320.3357688

[6] Y. Guo, C. Zhu, S. Hao, R. Hong, "Automatic Depression Detection via Learning and Fusing Features From Visual Cues," *IEEE Trans. Computational Social Systems*, 2022. https://doi.org/10.1109/TCSS.2022.3202316

[7] A. Pampouchidou et al., "Automated facial video-based recognition of depression and anxiety symptom severity: cross-corpus validation," *Machine Vision and Applications*, 31:30, 2020. https://doi.org/10.1007/s00138-020-01080-7

[8] G. Giannakakis et al., "Stress and anxiety detection using facial cues from videos," *Biomedical Signal Processing and Control*, 31:89–101, 2017. https://doi.org/10.1016/j.bspc.2016.06.020

[9] G. Giannakakis, A. Roussos, C. Andreou, S. Borgwardt, A. I. Korda, "Stress recognition identifying relevant facial action units through explainable artificial intelligence and machine learning," *Computer Methods and Programs in Biomedicine*, 257:108507, 2024. https://doi.org/10.1016/j.cmpb.2024.108507

[10] M. Gahalawat, R. Fernandez Rojas, T. Guha, R. Subramanian, R. Goecke, "Explainable Depression Detection via Head Motion Patterns," *ICMI '23*, pp. 261–270, 2023. https://doi.org/10.1145/3577190.3614130

[11] D. Gimeno-Gómez, A.-M. Bucur, A. Cosma, C.-D. Martínez-Hinarejos, P. Rosso, "Reading Between the Frames: Multi-modal Depression Detection in Videos from Non-verbal Cues," *ECIR 2024*. https://doi.org/10.1007/978-3-031-56027-9_12

[12] X. Cao, L. Zhai, P. Zhai, F. Li, T. He, L. He, "Deep learning-based depression recognition through facial expression: A systematic review," *Neurocomputing*, 627:129605, 2025. https://doi.org/10.1016/j.neucom.2025.129605

[13] X. Xu, K. Zhou, Y. Zhang, Y. Wang, F. Wang, X. Zhang, "Faces of the Mind: Unveiling Mental Health States Through Facial Expressions in 11,427 Adolescents," arXiv:2405.20072, 2024. https://arxiv.org/abs/2405.20072
