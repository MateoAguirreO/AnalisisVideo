# Top 10 papers — video para ansiedad/depresión (curado para TU tesis)

Selección priorizada para el pivote multimodal de la tesis de Samaná: **XAI**, **ansiedad + depresión**, **video de cara frontal a 30 fps**, **N pequeño (~80)**, español. Orden = qué leer primero.
Detalle completo de los 112 papers en [`dossier_literatura_video.md`](dossier_literatura_video.md).

---

### 1. Mahayossanunt et al. (2023) — *Explainable Depression Detection Based on Facial Expression Using LSTM on Attentional Intermediate Feature Fusion* — **Sensors** 23(23):9402
**Por qué #1:** es el match más cercano a tu tesis: detecta depresión desde **video de entrevista** con features faciales (AU intensidad, gaze, head pose) + LSTM **y explica** cada predicción con integrated gradients. XAI + AU/gaze/head + entrevista = tu setup exacto.
- DOI: https://doi.org/10.3390/s23239402 · Acceso abierto · PDF descargado en `papers/`

### 2. Giannakakis et al. (2024) — *Stress recognition identifying relevant facial action units through explainable AI* — **Comput. Methods Programs Biomed.** 257:108507
**Por qué:** usa **XAI para identificar qué Action Units** explican el estrés/ansiedad. Blueprint directo de "explicación clínica por AU" para personal de salud no especialista.
- DOI: https://doi.org/10.1016/j.cmpb.2024.108507 · PDF OA del autor (FORTH)

### 3. Xu et al. (2024) — *Faces of the Mind: Unveiling Mental Health States Through Facial Expressions in 11,427 Adolescents* — arXiv:2405.20072
**Por qué:** valida TODO tu pipeline factible: **OpenFace 2.0** (AU/gaze/head/pupil) sobre video estándar + estimación de **depresión Y ansiedad** + análisis de feature importance. Código y dataset públicos.
- https://arxiv.org/abs/2405.20072 · PDF descargado · github.com/xuxiaoooo/FACES

### 4. Gimeno-Gómez et al. (2024) — *Reading Between the Frames: Multi-modal Depression Detection in Videos from Non-verbal Cues* — **ECIR 2024**
**Por qué:** modelo temporal pensado para **video real ruidoso** usando exactamente los cues factibles: landmarks de **cara/cuerpo/mano**, gaze y **parpadeo**. Cubre fusión multimodal + micro-gestos corporales. **Con código.**
- https://arxiv.org/abs/2401.02746 · PDF descargado · github.com/cosmaadrian/multimodal-depression-from-video

### 5. Guo et al. (2022) — *Automatic Depression Detection via Learning and Fusing Features From Visual Cues* — **IEEE Trans. Computational Social Systems**
**Por qué:** método **visual-only** sobre AU/pose/gaze (OpenFace) con **atención por feature** (interpretable). Arquitectura concreta para la parte de video.
- DOI: https://doi.org/10.1109/TCSS.2022.3202316 · PDF descargado (arXiv 2203.00304)

### 6. Sahu et al. (2025) — *Beyond Questionnaires: Video Analysis for Social Anxiety Detection* — arXiv:2501.05461
**Por qué:** plantilla casi idéntica: clasifica **ansiedad social** desde video de tarea de habla usando **AUs + head/body pose + gaze** con ML/DL clásico. Tu mismo set de features y meta de tamizaje.
- https://arxiv.org/abs/2501.05461 · PDF descargado

### 7. Sahu et al. (2025) — *AnxietyFaceTrack: A Smartphone-Based Non-Intrusive Approach for Detecting Social Anxiety Using Facial Features* — arXiv:2502.16106
**Por qué:** ansiedad social con **cámara de smartphone** (tu mismo hardware) + **Random Forest interpretable** con feature importance. Restricciones de hardware y explicabilidad casi calcadas a Samaná.
- https://arxiv.org/abs/2502.16106 · PDF descargado

### 8. Giannakakis et al. (2017) — *Stress and anxiety detection using facial cues from videos* — **Biomed. Signal Process. Control** 31:89-101
**Por qué:** paper fundacional con el **vocabulario de cues faciales** de ansiedad/estrés (tasa de parpadeo, gaze, actividad de boca, movimiento de cabeza) — justo las features que extraerás a 30 fps y explicarás con SHAP.
- DOI: https://doi.org/10.1016/j.bspc.2016.06.020 · PDF OA del autor (FORTH)

### 9. Pampouchidou et al. (2020) — *Automated facial video-based recognition of depression and anxiety symptom severity: cross-corpus validation* — **Machine Vision and Applications** 31:30
**Por qué:** de los pocos que predicen **ansiedad Y depresión** desde video y prueban **generalización cross-corpus** — clave para tu doble objetivo PHQ-9/GAD-7 y para argumentar generalización con muestra pequeña.
- DOI: https://doi.org/10.1007/s00138-020-01080-7 · De pago (usar DOI/biblioteca)

### 10. Cao et al. (2025) — *Deep learning-based depression recognition through facial expression: A systematic review* — **Neurocomputing** 627:129605
**Por qué:** revisión sistemática al día del campo facial-depresión; base para tu capítulo de estado del arte y para justificar elecciones de features/benchmarks.
- DOI: https://doi.org/10.1016/j.neucom.2025.129605 · De pago (usar DOI/biblioteca)

---

**Bonus según el ángulo que elijas:**
- **Fusión audio+video (arquitectura):** Wei et al. 2022, *Sub-attentional Fusion* (ECCV-W) — https://arxiv.org/abs/2207.06180 · Ray et al. 2019, *Multi-level Attention* (AVEC) — https://arxiv.org/abs/1909.01417
- **Movimiento de cabeza explicable (kinemes):** Gahalawat et al. 2023 (ICMI) — https://arxiv.org/abs/2307.12241
- **Benchmark de datos (lo más parecido a Samaná):** DAIC-WOZ / E-DAIC — entrevistas con cara frontal + PHQ, base de casi toda esta literatura.
