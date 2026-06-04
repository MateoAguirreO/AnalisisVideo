# Dossier de literatura: analisis de VIDEO para deteccion de ansiedad/depresion

**Para:** pivote de la tesis (XAI biomarcadores acusticos, Samana) hacia un enfoque **multimodal audio+video**.
**Como se construyo:** busqueda multi-agente (12 sub-temas) con **verificacion adversarial de cada cita** (titulo, autores, venue y DOI confirmados; varios DOIs corregidos). **112 papers verificados**; **64 PDFs de acceso abierto ya descargados** en la carpeta papers/, el resto queda con su enlace.

---

## RESUMEN EJECUTIVO (leer esto primero)

### 1. Reality-check de viabilidad con TUS datos
Tus muestras: **252 clips, cara frontal, 1080x1920 vertical, 30 fps, ~4.4 min, audio 48 kHz**, etiquetas PHQ-9 / GAD-7.

| Lo que pediste / opciones | Viable con tu video? | Por que |
|---|---|---|
| **Micro-gestos corporales** (manos, cuerpo) | **SI** | Landmarks de cuerpo/mano (MediaPipe/OpenPose) funcionan a 30 fps. Angulo novedoso (datasets iMiGUE/SMG/MiGA). |
| **Movimientos de retina / oculares finos** (sacadas, microsacadas, pupilometria) | **NO (en sentido estricto)** | Requieren eye-tracker **infrarrojo a 120-250 Hz**. A 30 fps y con la cara dentro de un frame 1080p no se resuelven. |
| **Microexpresiones faciales puras** (apex 40-200 ms) | **NO** | Los datasets del area (CASME II, SAMM, SMIC) se graban a **100-200 fps**. A 30 fps no capturas el apex. |
| **Mirada / contacto visual grueso** (gaze direction, gaze aversion, % mirada a camara) | **SI** | OpenFace estima vector de mirada por frame; suficiente para aversion de la mirada, un marcador clinico real. |
| **Tasa de parpadeo (blink rate)** | **SI** | Detectable por EAR (eye aspect ratio) u OpenFace; marcador de ansiedad/estres. |
| **Action Units faciales + su dinamica temporal** | **SI (lo mas solido)** | OpenFace 2.0 da intensidad/presencia de ~17 AUs por frame. Base de casi toda la literatura clinica de video. |
| **Pose y movimiento de cabeza** | **SI** | Head pose por frame -> kinemes, energia de movimiento; marca retardo psicomotor en depresion. |
| **rPPG (pulso/estres desde la cara)** | **PARCIAL** | Posible a 30 fps pero ruidoso; tratar como exploratorio. |

> **Traduccion practica:** lo de "movimientos de retina" no es replicable con este video. Reorienta esa idea hacia **mirada/parpadeo (oculares gruesos)** + **micro-gestos corporales**, que SI son viables y siguen siendo novedosos.

### 2. Ruta recomendada (encaja con tu pipeline de audio + XAI)
1. **Extraer features de video con OpenFace 2.0**: AUs (intensidad+presencia), head pose, gaze, landmarks -> por frame.
2. **Agregar por participante**: estadisticos y descriptores de dinamica temporal (media, std, percentiles, tasa de cambio, blink rate, % gaze-aversion, energia de movimiento de cabeza).
3. **Modelos interpretables (RF / XGBoost / SVM)** sobre esas features -> **SHAP / LIME**, igual que tu pipeline acustico. Asi **preservas el hilo XAI** de la tesis.
4. **Fusion multimodal**: empezar por **late fusion** (combinar score de audio + score de video); luego attention-based fusion si hay margen.
5. Anadir **micro-gestos corporales** (MediaPipe) y **oculares gruesos** como features extra.

### 3. Start here -- los 9 papers ancla (los mas alineados a tu caso)
1. **Mahayossanunt et al. 2023, Sensors** -- XAI + AU/gaze/head + LSTM sobre video de entrevista. *El match mas cercano a tu tesis.* (PDF descargado)
2. **Giannakakis et al. 2024, CMPB** -- XAI identifica que AUs explican el estres/ansiedad. (link)
3. **Xu et al. 2024, Faces of the Mind** -- OpenFace en 11.427 adolescentes, depresion+ansiedad, feature importance. (PDF descargado, arXiv)
4. **Gimeno-Gomez et al. 2024, Reading Between the Frames (ECIR)** -- video in-the-wild, cues no verbales (landmarks+gaze+blink), **con codigo**. (PDF descargado, arXiv)
5. **Guo et al. 2022, IEEE TCSS** -- visual-only AU/pose/gaze + attention (XAI). (PDF descargado, arXiv)
6. **Sahu et al. 2025, Beyond Questionnaires** -- ansiedad social desde video (AUs+pose+gaze). (PDF descargado, arXiv)
7. **AnxietyFaceTrack 2025** -- ansiedad social con **camara de smartphone** + RandomForest interpretable. (PDF descargado, arXiv)
8. **Giannakakis et al. 2017, BSPC** -- vocabulario clasico de cues faciales de estres/ansiedad (blink, gaze, head). (link OA del autor)
9. **Pampouchidou et al. 2020, MVA** -- predice **ansiedad Y depresion** desde video con validacion cross-corpus. (link)

### 4. Advertencias honestas (conversar con tu director)
- **Cambia el alcance de la tesis**: tu problema/objetivos hoy dicen "senales de voz". Pasar a multimodal toca pregunta de investigacion, objetivos y estado del arte.
- **Etica/consentimiento**: el **rostro es dato biometrico**. Verificar que los consentimientos de Samana cubren analisis facial/video (no solo voz) antes de procesar rostros. Puede requerir adenda al comite de etica.
- **Tamano muestral**: ~252 clips es chico para deep learning end-to-end (riesgo de overfit). Por eso se recomienda **features OpenFace + modelos clasicos + XAI**, no CNN/3D-CNN crudas.
- **Alcance micro vs macro**: prioriza dinamica de AUs, mirada, parpadeo, cabeza y micro-gestos corporales (todo a 30 fps). Deja microexpresiones y eye-tracking fino fuera del alcance, o como trabajo futuro con hardware dedicado.

---

## Indice de sub-temas
1. Fusion multimodal AUDIO+VIDEO para depresion (el nucleo del pivote) -- 10 papers
2. Action Units faciales y dinamica facial en depresion -- 9 papers
3. Comportamiento facial en ANSIEDAD y ansiedad social -- 8 papers
4. Microexpresiones: metodos y datasets (CASME / SAMM / SMIC...) -- 10 papers
5. Micro-GESTOS corporales para emocion/estres (iMiGUE / SMG / MiGA) -- 10 papers
6. Mirada / contacto visual / gaze aversion en depresion-ansiedad -- 9 papers
7. Parpadeo / pupilometria / marcadores oculares -- 10 papers
8. Pose y movimiento de cabeza (retardo psicomotor) -- 10 papers
9. XAI sobre video/cara (explicabilidad) -- hilo central de la tesis -- 9 papers
10. Webcam / smartphone / 30 fps in-the-wild / rPPG -- 10 papers
11. Datasets y estudios en espanol o Latinoamerica -- 8 papers
12. Surveys y revisiones sistematicas (2021-2026) -- 9 papers

## 1. Fusion multimodal AUDIO+VIDEO para depresion (el nucleo del pivote)

*Sub-tema multimodal-av-depression -- 10 papers verificados*

### 1. The Distress Analysis Interview Corpus of human and computer interviews  `[OA - solo link]`
- **Autores:** Jonathan Gratch, Ron Artstein, Gale Lucas, Giota Stratou, Stefan Scherer, Angela Nazarian, Rachel Wood, Jill Boberg, David DeVault, Stacy Marsella, David Traum, Skip Rizzo, Louis-Philippe Morency
- **Anio / Venue:** 2014 -- LREC 2014 (Proceedings of the 9th International Conference on Language Resources and Evaluation), pp. 3123-3128, ELRA
- **Link:** https://aclanthology.org/L14-1421/
- **PDF OA:** https://aclanthology.org/L14-1421.pdf
- **Dataset:** DAIC / DAIC-WOZ
- **Modalidad:** audio-visual (+ questionnaire/text)
- **Relevancia:** Seminal dataset paper for the DAIC/DAIC-WOZ corpus (audio + frontal-face video + PHQ-8 labels) on which the AV-depression literature builds; gives the student the canonical reference and benchmark protocol for the video pivot.
- **Viabilidad (30fps/cara):** None — DAIC-WOZ provides 30fps OpenFace face/AU/pose/gaze features, the same regime as the student's 30fps phone video; no high-fps or IR hardware involved.

### 2. AVEC 2016 – Depression, Mood, and Emotion Recognition Workshop and Challenge  `[PDF DESCARGADO]`
- **Autores:** Michel Valstar, Jonathan Gratch, Björn Schuller, Fabien Ringeval, Denis Lalanne, Mercedes Torres Torres, Stefan Scherer, Giota Stratou, Roddy Cowie, Maja Pantic
- **Anio / Venue:** 2016 -- AVEC '16 (6th International Workshop on Audio/Visual Emotion Challenge, ACM Multimedia), pp. 3-10
- **DOI:** https://doi.org/10.1145/2988257.2988258
- **Link:** https://arxiv.org/abs/1605.01600
- **PDF OA:** https://arxiv.org/pdf/1605.01600
- **PDF local:** papers/01_002_avec_2016_depression_mood_and_emotion_re.pdf
- **Dataset:** DAIC-WOZ
- **Modalidad:** audio-visual (+ physiological)
- **Relevancia:** Defines the DAIC-WOZ depression sub-challenge and the audio+video late-fusion baseline, giving the student a standard fusion pipeline and benchmark numbers to compare a multimodal model against.
- **Viabilidad (30fps/cara):** None — baseline uses standard-frame-rate AV features; no micro-expression/IR requirements.

### 3. AVEC 2019 Workshop and Challenge: State-of-Mind, Detecting Depression with AI, and Cross-Cultural Affect Recognition  `[PDF DESCARGADO]`
- **Autores:** Fabien Ringeval, Björn Schuller, Michel Valstar, Nicholas Cummins, Roddy Cowie, Leili Tavabi, Maximilian Schmitt, Sina Alisamir, Shahin Amiriparian, Eva-Maria Messner, et al.
- **Anio / Venue:** 2019 -- AVEC '19 (9th International Workshop on Audio/Visual Emotion Challenge and Workshop, ACM Multimedia)
- **DOI:** https://doi.org/10.1145/3347320.3357688
- **Link:** https://arxiv.org/abs/1907.11510
- **PDF OA:** https://arxiv.org/pdf/1907.11510
- **PDF local:** papers/01_003_avec_2019_workshop_and_challenge_state_o.pdf
- **Dataset:** E-DAIC (Extended DAIC)
- **Modalidad:** audio-visual (+ text)
- **Relevancia:** Introduces the E-DAIC benchmark and its audio-visual PHQ-8 regression baseline, and explicitly frames cross-cultural robustness — directly relevant to the student's Spanish-speaking rural-Colombian cohort, which differs from the English DAIC population.
- **Viabilidad (30fps/cara):** None — uses 30fps-derived AV features; cross-cultural transfer caveat is conceptual, not a hardware constraint.

### 4. Multi-level Attention Network using Text, Audio and Video for Depression Prediction  `[PDF DESCARGADO]`
- **Autores:** Anupama Ray, Siddharth Kumar, Rutvik Reddy, Prerana Mukherjee, Ritu Garg
- **Anio / Venue:** 2019 -- AVEC '19 (9th International Workshop on Audio/Visual Emotion Challenge and Workshop, ACM Multimedia)
- **DOI:** https://doi.org/10.1145/3347320.3357697
- **Link:** https://arxiv.org/abs/1909.01417
- **PDF OA:** https://arxiv.org/pdf/1909.01417
- **PDF local:** papers/01_004_multi_level_attention_network_using_text.pdf
- **Dataset:** E-DAIC
- **Modalidad:** audio-visual + text
- **Relevancia:** AVEC 2019 multimodal depression entry whose multi-level (intra- and inter-modality) attention fusion is a concrete, reproducible architecture (code released) the student can adapt for audio+facial fusion, with attention weights doubling as an explainability signal.
- **Viabilidad (30fps/cara):** None — uses provided E-DAIC AV features at standard frame rate.

### 5. Depression Scale Recognition from Audio, Visual and Text Analysis  `[PDF DESCARGADO]`
- **Autores:** Shubham Dham, Anirudh Sharma, Abhinav Dhall
- **Anio / Venue:** 2017 -- arXiv preprint (work for the AVEC 2017 Audio/Visual Emotion Challenge)
- **DOI:** https://doi.org/10.48550/arXiv.1709.05865
- **Link:** https://arxiv.org/abs/1709.05865
- **PDF OA:** https://arxiv.org/pdf/1709.05865
- **PDF local:** papers/01_005_depression_scale_recognition_from_audio_.pdf
- **Dataset:** DAIC-WOZ
- **Modalidad:** audio-visual + text
- **Relevancia:** An early DAIC-WOZ AV+text pipeline (GMM/Fisher-vector visual features, low-level audio, decision-level fusion) giving the student an interpretable hand-crafted-feature fusion baseline before moving to deep models.
- **Viabilidad (30fps/cara):** None — visual features (gaze, pose, AUs) computed from standard 30fps DAIC-WOZ video.

### 6. Automatic Depression Detection via Learning and Fusing Features From Visual Cues  `[PDF DESCARGADO]`
- **Autores:** Yanrong Guo, Chenyang Zhu, Shijie Hao, Richang Hong
- **Anio / Venue:** 2022 -- IEEE Transactions on Computational Social Systems
- **DOI:** https://doi.org/10.1109/TCSS.2022.3202316
- **Link:** https://arxiv.org/abs/2203.00304
- **PDF OA:** https://arxiv.org/pdf/2203.00304
- **PDF local:** papers/01_006_automatic_depression_detection_via_learn.pdf
- **Dataset:** DAIC-WOZ
- **Modalidad:** visual (facial action units, head pose, gaze, landmarks)
- **Relevancia:** Visual-only method using a Temporal Dilated Convolutional Network plus feature-wise attention over AU/pose/gaze cues — maps directly to the feasible visual features (AUs, head pose, coarse gaze) the student can extract from 30fps clips, with feature-wise attention supporting XAI.
- **Viabilidad (30fps/cara):** None — relies on OpenFace AU/pose/gaze sequences at 30fps; no micro-expression spotting or IR eye-tracking required.

### 7. Multi-modal Depression Estimation Based on Sub-attentional Fusion  `[PDF DESCARGADO]`
- **Autores:** Ping-Cheng Wei, Kunyu Peng, Alina Roitberg, Kailun Yang, Jiaming Zhang, Rainer Stiefelhagen
- **Anio / Venue:** 2022 -- Computer Vision – ECCV 2022 Workshops (Springer LNCS 13806), pp. 623-639
- **DOI:** https://doi.org/10.1007/978-3-031-25075-0_42
- **Link:** https://arxiv.org/abs/2207.06180
- **PDF OA:** https://arxiv.org/pdf/2207.06180
- **PDF local:** papers/01_007_multi_modal_depression_estimation_based_.pdf
- **Dataset:** DAIC-WOZ
- **Modalidad:** audio-visual + text
- **Relevancia:** Sub-attentional fusion over Conv-BiLSTM backbones that beats conventional late fusion, giving the student a modern attention-fusion architecture whose per-modality weights are interpretable for clinical reporting.
- **Viabilidad (30fps/cara):** None — uses standard DAIC-WOZ AV/text features at native frame rate.

### 8. Reading Between the Frames: Multi-modal Depression Detection in Videos from Non-verbal Cues  `[PDF DESCARGADO]`
- **Autores:** David Gimeno-Gómez, Ana-Maria Bucur, Adrian Cosma, Carlos-David Martínez-Hinarejos, Paolo Rosso
- **Anio / Venue:** 2024 -- ECIR 2024 (46th European Conference on Information Retrieval), Springer LNCS 14608
- **DOI:** https://doi.org/10.1007/978-3-031-56027-9_12
- **Link:** https://arxiv.org/abs/2401.02746
- **PDF OA:** https://arxiv.org/pdf/2401.02746
- **PDF local:** papers/01_008_reading_between_the_frames_multi_modal_d.pdf
- **Dataset:** DAIC-WOZ, E-DAIC, D-Vlog
- **Modalidad:** audio-visual / non-verbal (speech embeddings, face emotion embeddings, face/body/hand landmarks, gaze and blinking)
- **Relevancia:** A 2024 temporal model built for noisy real-world video using exactly the feasible non-verbal cues — face/body/hand landmarks, coarse gaze, and blink rate — closely matching the student's frontal-face phone clips and validating body micro-gestures + blink as usable signals.
- **Viabilidad (30fps/cara):** None — explicitly designed for in-the-wild standard-frame-rate video; gaze and blink are coarse (not microsaccade/IR), matching the student's data exactly.

### 9. Explainable Depression Detection Based on Facial Expression Using LSTM on Attentional Intermediate Feature Fusion with Label Smoothing  `[PDF DESCARGADO]`
- **Autores:** Yanisa Mahayossanunt, Natawut Nupairoj, Solaphat Hemrungrojn, Peerapon Vateekul
- **Anio / Venue:** 2023 -- Sensors (MDPI), vol. 23, no. 23, art. 9402
- **DOI:** https://doi.org/10.3390/s23239402
- **Link:** https://www.mdpi.com/1424-8220/23/23/9402
- **PDF OA:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10708765/
- **PDF local:** papers/01_009_explainable_depression_detection_based_o.pdf
- **Dataset:** Custom interview-video corpus (Chulalongkorn University); methodology transferable to DAIC-WOZ-style AU/gaze features
- **Modalidad:** visual (facial action unit intensity, gaze angles, head radians) with attention LSTM
- **Relevancia:** The single closest match to the thesis's XAI thread — detects depression from interview-video facial features (AU intensity, gaze, head angles) and uses integrated gradients for per-patient feature explanations, exactly the clinically interpretable output needed for non-specialist rural health workers.
- **Viabilidad (30fps/cara):** None — uses 30fps-derived facial AU/gaze/head-angle features; integrated-gradient explanations require no special hardware.

### 10. Harnessing multimodal approaches for depression detection using large language models and facial expressions  `[PDF DESCARGADO]`
- **Autores:** Misha Sadeghi, Robert Richer, Bernhard Egger, Lena Schindler-Gmelch, Lydia Helene Rupp, Farnaz Rahimi, Matthias Berking, Bjoern M. Eskofier
- **Anio / Venue:** 2024 -- npj Mental Health Research (Nature Portfolio), vol. 3, art. 66
- **DOI:** https://doi.org/10.1038/s44184-024-00112-8
- **Link:** https://www.nature.com/articles/s44184-024-00112-8
- **PDF OA:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11666580/
- **PDF local:** papers/01_010_harnessing_multimodal_approaches_for_dep.pdf
- **Dataset:** E-DAIC (Extended DAIC) — interview transcripts + facial features from video frames
- **Modalidad:** audio-visual (LLM-processed text/audio + facial expressions)
- **Relevancia:** A recent peer-reviewed (Nature-portfolio) study combining LLM-based speech/transcript analysis with facial-expression features on E-DAIC, showing how to fuse the student's existing acoustic pipeline with feasible facial cues toward interpretable, clinically oriented multimodal design.
- **Viabilidad (30fps/cara):** None — facial-expression features are standard frame-rate; no micro-expression or IR eye-tracking.


## 2. Action Units faciales y dinamica facial en depresion

*Sub-tema facial-au-depression -- 9 papers verificados*

### 11. Detecting Depression from Facial Actions and Vocal Prosody  `[PDF DESCARGADO]`
- **Autores:** Jeffrey F. Cohn, Tomas Simon Kruez, Iain Matthews, Ying Yang, Minh Hoai Nguyen, Michael T. (Margara Tejera) Padilla, Feng Zhou, Fernando De la Torre
- **Anio / Venue:** 2009 -- 2009 3rd International Conference on Affective Computing and Intelligent Interaction and Workshops (ACII), IEEE, pp. 1-7
- **DOI:** https://doi.org/10.1109/ACII.2009.5349358
- **Link:** https://www.semanticscholar.org/paper/Detecting-depression-from-facial-actions-and-vocal-Cohn-Kruez/e423265bda7cc1260fe37813facb9b904429aa81
- **PDF OA:** https://www3.cs.stonybrook.edu/~minhhoai/papers/acii-paper_final.pdf
- **PDF local:** papers/02_011_detecting_depression_from_facial_actions.pdf
- **Dataset:** Clinical interview cohort of depressed patients undergoing treatment (University of Pittsburgh); manual FACS coding + active appearance models (AAM) + pitch extraction
- **Modalidad:** facial AUs (manual FACS + AAM) and vocal prosody (audio-visual)
- **Relevancia:** Seminal proof-of-concept that automatically measured facial actions plus vocal prosody map onto clinical depression diagnosis, directly motivating the thesis's pivot from audio-only to adding facial-AU video.
- **Viabilidad (30fps/cara):** None - uses standard-rate clinical interview video and audio prosody, fully compatible with the student's 30fps phone clips.

### 12. Nonverbal Social Withdrawal in Depression: Evidence from manual and automatic analyses  `[OA - solo link]`
- **Autores:** Jeffrey M. Girard, Jeffrey F. Cohn, Mohammad H. Mahoor, S. Mohammad Mavadati, Zakia Hammal, Dean P. Rosenwald
- **Anio / Venue:** 2014 -- Image and Vision Computing, Vol. 32, Issue 10, pp. 641-647 (Elsevier)
- **DOI:** https://doi.org/10.1016/j.imavis.2013.12.007
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC4217695/
- **PDF OA:** https://pmc.ncbi.nlm.nih.gov/articles/PMC4217695/
- **Dataset:** Clinical-trial depression cohort (adults with MDD, HRSD-rated, longitudinal clinical-interview sessions over treatment)
- **Modalidad:** facial AUs (AU12, AU14, AU15, AU24) and head pose/motion dynamics
- **Relevancia:** Foundational FACS-based finding that depression severity tracks specific AUs and reduced head motion; the analyzed clinical-interview video is standard-rate, demonstrating the methodology is feasible at the student's frame rate.
- **Viabilidad (30fps/cara):** None - AU and head-pose features are exactly what 30fps phone video supports; methodology validated on standard-rate clinical interview video.

### 13. Social Risk and Depression: Evidence from Manual and Automatic Facial Expression Analysis  `[OA - solo link]`
- **Autores:** Jeffrey M. Girard, Jeffrey F. Cohn, Mohammad H. Mahoor, Seyed Mohammad Mavadati, Dean P. Rosenwald
- **Anio / Venue:** 2013 -- 2013 10th IEEE International Conference and Workshops on Automatic Face and Gesture Recognition (FG)
- **DOI:** https://doi.org/10.1109/FG.2013.6553748
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC3935843/
- **PDF OA:** https://pmc.ncbi.nlm.nih.gov/articles/PMC3935843/
- **Dataset:** Longitudinal depression treatment cohort, clinical interviews video-recorded over course of treatment
- **Modalidad:** facial AUs / FACS expression dynamics (manual vs automatic), head pose
- **Relevancia:** Shows automatic AU analysis is highly consistent with manual FACS coding and reproduces the same depression-severity effects (more contempt-related actions, fewer smiles), validating automated AU pipelines like the one the thesis will adopt.
- **Viabilidad (30fps/cara):** None - standard clinical interview video; AU-level analysis feasible at 30fps.

### 14. Explainable Depression Detection Based on Facial Expression Using LSTM on Attentional Intermediate Feature Fusion with Label Smoothing  `[PDF DESCARGADO]`
- **Autores:** Yanisa Mahayossanunt, Natawut Nupairoj, Solaphat Hemrungrojn, Peerapon Vateekul
- **Anio / Venue:** 2023 -- Sensors, Vol. 23, Issue 23, Article 9402 (MDPI)
- **DOI:** https://doi.org/10.3390/s23239402
- **Link:** https://www.mdpi.com/1424-8220/23/23/9402
- **PDF OA:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10708765/
- **PDF local:** papers/02_014_explainable_depression_detection_based_o.pdf
- **Dataset:** 474 interview video samples (134 depressed / 340 non-depressed) collected at Chulalongkorn University
- **Modalidad:** facial AU intensity, gaze angles, head pose (OpenFace features) with LSTM + attention + integrated gradients XAI
- **Relevancia:** Directly models the thesis's exact setup - interview-video AU/gaze/head features fed to a sequence model WITH explainability (integrated gradients showing per-patient important features), mirroring the XAI-for-clinicians requirement.
- **Viabilidad (30fps/cara):** None - uses OpenFace AU intensity, gaze, and head pose from standard interview video, all extractable from 30fps; no high-fps or IR eye-tracking needed.

### 15. Explainable Depression Detection via Head Motion Patterns  `[PDF DESCARGADO]`
- **Autores:** Monika Gahalawat, Raul Fernandez Rojas, Tanaya Guha, Ramanathan Subramanian, Roland Goecke
- **Anio / Venue:** 2023 -- Proceedings of the 25th ACM International Conference on Multimodal Interaction (ICMI '23), pp. 261-270
- **DOI:** https://doi.org/10.1145/3577190.3614130
- **Link:** https://arxiv.org/abs/2307.12241
- **PDF OA:** https://arxiv.org/pdf/2307.12241
- **PDF local:** papers/02_015_explainable_depression_detection_via_hea.pdf
- **Dataset:** BlackDog and AVEC2013 depression datasets
- **Modalidad:** head motion / pose dynamics encoded as interpretable 'kinemes' (head motion units)
- **Relevancia:** Provides an inherently interpretable head-motion biomarker (kinemes) that complements facial AUs and preserves the thesis's explainability thread, showing reduced head movement in depressed subjects.
- **Viabilidad (30fps/cara):** None - head pose/motion is reliably estimable at 30fps in a 1080p frame; one of the most robust feasible modalities for this data.

### 16. Exploring Facial Biomarkers for Depression through Temporal Analysis of Action Units  `[PDF DESCARGADO]`
- **Autores:** Aditya Parikh, Misha Sadeghi, Robert Richer, Lydia Helene Rupp, Lena Schindler-Gmelch, Marie Keinert, Malin Hager, Klara Capito, Farnaz Rahimi, Bernhard Egger, Matthias Berking, Bjoern M. Eskofier
- **Anio / Venue:** 2024 -- arXiv preprint (cs.CV), arXiv:2407.13753
- **DOI:** https://doi.org/10.48550/arXiv.2407.13753
- **Link:** https://arxiv.org/abs/2407.13753
- **PDF OA:** https://arxiv.org/pdf/2407.13753
- **PDF local:** papers/02_016_exploring_facial_biomarkers_for_depressi.pdf
- **Dataset:** Video recordings of depressed vs. non-depressed participants (AU time series, feature extraction + time-series classification)
- **Modalidad:** facial AUs and emotion dynamics (temporal analysis, intensity comparisons of key AUs)
- **Relevancia:** Recent work using temporal/dynamic AU patterns (elevated sadness-related AUs; reduced AU6/AU12 happiness) as objective depression biomarkers - a template for the dynamic-AU feature set the thesis will build and explain.
- **Viabilidad (30fps/cara):** None - AU time-series from standard frontal-face video; directly feasible at 30fps.

### 17. Decoding depression with computer vision-assisted analysis of synchronized facial expressions  `[DE PAGO]`
- **Autores:** Seohyeon Lee, Yunsu Kim, Hayoung Ryu, Sunkyung Yoon, M. Justin Kim
- **Anio / Venue:** 2025 -- Journal of Affective Disorders, Vol. 395 (Part A), Article 120663 (Elsevier)
- **DOI:** https://doi.org/10.1016/j.jad.2025.120663
- **Link:** https://doi.org/10.1016/j.jad.2025.120663
- **Dataset:** Naturalistic-viewing paradigm cohort; inter-subject correlation (ISC) of AU time series from facial expressions during emotionally charged videos
- **Modalidad:** facial AUs and AU synchronization/dynamics (inter-subject correlation / AU-ISC vectors)
- **Relevancia:** Recent peer-reviewed study showing AU-based dynamics under naturalistic viewing detect depression at 72-90% accuracy, supporting AU time-series features as discriminative biomarkers relevant to the thesis's naturalistic phone-video data.
- **Viabilidad (30fps/cara):** None - AU extraction from naturalistic facial video; feasible at 30fps. (No OA PDF located; access via DOI/publisher.)

### 18. Demystifying Mental Health by Decoding Facial Action Unit Sequences  `[OA - solo link]`
- **Autores:** Deepika Sharma, Jaiteg Singh, Sukhjit Singh Sehra, Sumeet Kaur Sehra
- **Anio / Venue:** 2024 -- Big Data and Cognitive Computing, Vol. 8, Issue 7, Article 78 (MDPI)
- **DOI:** https://doi.org/10.3390/bdcc8070078
- **Link:** https://www.mdpi.com/2504-2289/8/7/78
- **PDF OA:** https://www.mdpi.com/2504-2289/8/7/78/pdf
- **Dataset:** CASME II and SAMM micro-expression datasets (plus a depression case analysis)
- **Modalidad:** facial AUs and micro-expressions (CNN classification of AU combinations, K-means AU clustering)
- **Relevancia:** Demonstrates AU-combination decoding for mental-health screening and links specific AUs (reduced AU6/AU12) to depression, but is built on high-speed micro-expression corpora; only the coarse AU-presence/AU-combination ideas transfer to the thesis.
- **Viabilidad (30fps/cara):** FEASIBILITY ISSUE: trained on CASME II / SAMM, which are recorded at ~100-200 fps for micro-expression spotting. The micro-expression component is NOT reproducible on the student's 30fps phone video; only the coarse AU-presence/AU-combination ideas transfer.

### 19. Deep learning-based depression recognition through facial expression: A systematic review  `[DE PAGO]`
- **Autores:** Xiaoming Cao, Lingling Zhai, Pengpeng Zhai, Fangfei Li, Tao He, Lang He
- **Anio / Venue:** 2025 -- Neurocomputing, Vol. 627, Article 129605 (Elsevier)
- **DOI:** https://doi.org/10.1016/j.neucom.2025.129605
- **Link:** https://doi.org/10.1016/j.neucom.2025.129605
- **Dataset:** Survey of multiple depression facial-expression datasets (AVEC/DAIC-WOZ, AVEC2013/2014, and others), covering 2017-2024
- **Modalidad:** facial expressions / AUs (review of CNN, 3DCNN, transfer-learning and hybrid spatial / spatial-temporal deep models)
- **Relevancia:** Up-to-date systematic review mapping the landscape of facial-expression deep learning for depression, useful for the thesis's related-work chapter and for justifying AU/expression-dynamics feature choices and benchmarks.
- **Viabilidad (30fps/cara):** N/A (review). Useful caveat: many surveyed methods assume standard-rate interview video, aligning with the student's data; high-fps micro-expression methods are a separate, infeasible branch.


## 3. Comportamiento facial en ANSIEDAD y ansiedad social

*Sub-tema facial-anxiety -- 8 papers verificados*

### 20. Stress and anxiety detection using facial cues from videos  `[PDF DESCARGADO]`
- **Autores:** Giorgos Giannakakis, Matthew Pediaditis, Dimitris Manousos, Eleni Kazantzaki, Franco Chiarugi, Panagiotis G. Simos, Kostas Marias, Manolis Tsiknakis
- **Anio / Venue:** 2017 -- Biomedical Signal Processing and Control (Elsevier), Vol. 31, pp. 89-101
- **DOI:** https://doi.org/10.1016/j.bspc.2016.06.020
- **Link:** https://doi.org/10.1016/j.bspc.2016.06.020
- **PDF OA:** http://users.ics.forth.gr/ggian/publications/journals/2017%20Giannakakis%20Stress%20and%20anxiety%20detection%20using%20facial%20cues%20from%20videos.pdf
- **PDF local:** papers/03_020_stress_and_anxiety_detection_using_facia.pdf
- **Dataset:** Custom FORTH stress/anxiety video dataset (neutral/relaxed/stressed protocol with internal and external stressors)
- **Modalidad:** facial AUs / semi-voluntary facial cues (eye gaze distribution, blink rate, mouth activity, lip deformation, head movement and velocity)
- **Relevancia:** Foundational, directly on-topic paper showing which semi-voluntary facial cues (blink rate, head motion, gaze, mouth activity) discriminate anxiety/stress from neutral video, providing the exact handcrafted-feature vocabulary this thesis can extract at 30fps and explain with SHAP.
- **Viabilidad (30fps/cara):** Fully feasible at 30fps; uses coarse blink/gaze/head-motion features, no high-fps or IR eye-tracking required.

### 21. Beyond Questionnaires: Video Analysis for Social Anxiety Detection  `[PDF DESCARGADO]`
- **Autores:** Nilesh Kumar Sahu, Nandigramam Sai Harshit, Rishabh Uikey, Haroon R. Lone
- **Anio / Venue:** 2025 -- arXiv preprint (cs.CV / cs.HC), arXiv:2501.05461
- **DOI:** https://doi.org/10.48550/arXiv.2501.05461
- **Link:** https://arxiv.org/abs/2501.05461
- **PDF OA:** https://arxiv.org/pdf/2501.05461
- **PDF local:** papers/03_021_beyond_questionnaires_video_analysis_for.pdf
- **Dataset:** Custom dataset of 92 participants giving impromptu speeches in a controlled environment, labeled for Social Anxiety Disorder (SAD)
- **Modalidad:** video: facial action units, head pose, body pose, 3D eye gaze (behavioral features of head, body, eye gaze, AUs)
- **Relevancia:** Very close methodological template: classifies social anxiety from speech-task video using AUs + head/body pose + gaze with standard ML/DL (up to ~74% accuracy), mirroring the feasible feature set and clinical-screening goal of this thesis.
- **Viabilidad (30fps/cara):** Feasible; eye gaze is coarse 3D-vector (OpenFace-style), not IR microsaccade tracking, so it matches this student's 30fps phone video.

### 22. AnxietyFaceTrack: A Smartphone-Based Non-Intrusive Approach for Detecting Social Anxiety Using Facial Features  `[PDF DESCARGADO]`
- **Autores:** Nilesh Kumar Sahu, Snehil Gupta, Haroon R. Lone
- **Anio / Venue:** 2025 -- arXiv preprint (cs.CV), arXiv:2502.16106
- **DOI:** https://doi.org/10.48550/arXiv.2502.16106
- **Link:** https://arxiv.org/abs/2502.16106
- **PDF OA:** https://arxiv.org/pdf/2502.16106
- **PDF local:** papers/03_022_anxietyfacetrack_a_smartphone_based_non_.pdf
- **Dataset:** Custom dataset of 91 participants in unstaged social settings, recorded with a low-cost smartphone camera
- **Modalidad:** facial features from smartphone video: eye movements, head position, facial landmarks, facial action units
- **Relevancia:** Demonstrates that a low-cost smartphone camera plus Random Forest on facial landmarks/AUs/head-position can detect (multiclass) social anxiety at 91% with interpretable feature importance, almost exactly the hardware and explainability constraints of this rural-Colombia thesis.
- **Viabilidad (30fps/cara):** Highly feasible; explicitly built for ordinary smartphone video, no high-fps or IR requirements.

### 23. Faces of the Mind: Unveiling Mental Health States Through Facial Expressions in 11,427 Adolescents  `[PDF DESCARGADO]`
- **Autores:** Xiao Xu, Keyin Zhou, Yan Zhang, Yang Wang, Fei Wang, Xizhe Zhang
- **Anio / Venue:** 2024 -- arXiv preprint arXiv:2405.20072 (May 2024); a related version appears in IEEE Xplore
- **DOI:** https://doi.org/10.48550/arXiv.2405.20072
- **Link:** https://arxiv.org/abs/2405.20072
- **PDF OA:** https://arxiv.org/pdf/2405.20072
- **PDF local:** papers/03_023_faces_of_the_mind_unveiling_mental_healt.pdf
- **Dataset:** FACES / 'Faces of the Mind': 11,427 adolescents (large-scale neutral reading-task facial video collection), DASS-21 labels for depression/anxiety/stress; OpenFace 2.0 features (code at github.com/xuxiaoooo/FACES)
- **Modalidad:** facial expressions from video (OpenFace 2.0 AU/gaze/head-pose/pupil features) with ML (tree-based: RF/XGBoost/FTTransformer) and DL (LI-FPN, MSN)
- **Relevancia:** The strongest data-constraint match: standard-rate face video, a neutral reading task, OpenFace 2.0 AU/gaze/head-pose features, and explicit depression/anxiety/stress estimation with feature-importance analysis, validating the entire feasible-feature + interpretability pipeline this thesis plans.
- **Viabilidad (30fps/cara):** Directly feasible; OpenFace pipeline on standard-rate face video, no high-fps or IR eye-tracking. (Note: candidate's specific '30 fps H.264' framing is plausible but the exact codec/fps spec was not independently confirmed from the abstract; the OpenFace standard-rate-video modality IS confirmed.)

### 24. Automated facial video-based recognition of depression and anxiety symptom severity: cross-corpus validation  `[DE PAGO]`
- **Autores:** Anastasia Pampouchidou, Matthew Pediaditis, Eleni Kazantzaki, Stylianos Sfakianakis, Iro A. Apostolaki, Kalliopi Argyraki, Dimitris Manousos, Fabrice Meriaudeau, Kostas Marias, Fan Yang, Manolis Tsiknakis, Maria Basta, Alexandros N. Vgontzas, Panagiotis Simos
- **Anio / Venue:** 2020 -- Machine Vision and Applications (Springer), Vol. 31, Article 30
- **DOI:** https://doi.org/10.1007/s00138-020-01080-7
- **Link:** https://doi.org/10.1007/s00138-020-01080-7
- **Dataset:** Two clinical corpora (cross-corpus): facial videos with depression and anxiety symptom-severity labels
- **Modalidad:** facial video: dynamic descriptors (motion history image), appearance features (LBP, HOG) and deep VGG features via transfer learning; head pose / facial dynamics
- **Relevancia:** One of the few works that predicts BOTH anxiety and depression symptom severity from facial video and tests cross-corpus generalization, directly relevant to this thesis's dual PHQ-9/GAD-7 targets and to generalization concerns on a small Colombian sample.
- **Viabilidad (30fps/cara):** Feasible; motion/appearance descriptors and head pose are extractable from standard-rate frontal video, no high-fps/IR needed.

### 25. Automatic stress analysis from facial videos based on deep facial action units recognition  `[DE PAGO]`
- **Autores:** Giorgos Giannakakis, Mohammad Rami Koujan, Anastasios Roussos, Kostas Marias
- **Anio / Venue:** 2022 -- Pattern Analysis and Applications (Springer), Vol. 25(3), pp. 521-535
- **DOI:** https://doi.org/10.1007/s10044-021-01012-9
- **Link:** https://doi.org/10.1007/s10044-021-01012-9
- **Dataset:** FORTH stress dataset; deep AU model trained on UNBC-McMaster and Bosphorus AU corpora
- **Modalidad:** facial video: deep-learned facial action unit intensities (geometric 3D-deformation + deep appearance features) as quantitative stress indices
- **Relevancia:** Provides a modern deep AU-recognition pipeline that turns facial video into interpretable AU intensities used as quantitative stress/anxiety indices, supplying the AU-extraction backbone that feeds SHAP/LIME explanations in this thesis.
- **Viabilidad (30fps/cara):** Feasible; AU intensity regression works on standard-rate video, no high-fps or IR eye-tracking.

### 26. Stress recognition identifying relevant facial action units through explainable artificial intelligence and machine learning  `[PDF DESCARGADO]`
- **Autores:** Giorgos Giannakakis, Anastasios Roussos, Christina Andreou, Stefan Borgwardt, Alexandra I. Korda
- **Anio / Venue:** 2024 -- Computer Methods and Programs in Biomedicine (Elsevier), Vol. 257, Article 108507
- **DOI:** https://doi.org/10.1016/j.cmpb.2024.108507
- **Link:** https://doi.org/10.1016/j.cmpb.2024.108507
- **PDF OA:** http://users.ics.forth.gr/ggian/publications/journals/2024%20Giannakakis%20Stress%20recognition%20identifying%20relevant%20facial%20action%20units%20through%20explainable%20artificial%20intelligence%20and%20machine%20learning.pdf
- **PDF local:** papers/03_026_stress_recognition_identifying_relevant_.pdf
- **Dataset:** New FORTH-style acute-stress facial video dataset: 58 participants, 4 experimental phases, 11 stress/non-stress tasks with annotated AUs
- **Modalidad:** facial action units with explainable AI (feature ranking, ML and DL); best stress-recognition accuracy ~95.7%
- **Relevancia:** Directly fuses the two core threads of this thesis: it uses explainable AI to identify which facial action units drive stress/anxiety recognition, giving a concrete blueprint for clinically interpretable AU-level explanations for non-specialist health workers.
- **Viabilidad (30fps/cara):** Feasible; AU + XAI pipeline runs on standard video, no high-fps/IR requirement.

### 27. LI-FPN: Depression and Anxiety Detection from Learning and Imitation  `[DE PAGO]`
- **Autores:** Reported lead author Xingyun Li et al. (per IEEE/Semantic Scholar). NOTE: candidate listed 'Wei Zhang, Kaining Mao, Jie Chen et al.' which appears to be incorrect/conflated; confirm exact author list against IEEE Xplore before citing.
- **Anio / Venue:** 2023 -- 2023 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), pp. (IEEE)
- **DOI:** https://doi.org/10.1109/BIBM58861.2023.10385591
- **Link:** https://ieeexplore.ieee.org/document/10385591/
- **Dataset:** VFEM (Voluntary Facial Expression Mimicry) dataset: 164 subjects (82 case / 82 control) performing facial-expression mimicry, labeled for depression and anxiety
- **Modalidad:** facial expression video: spatiotemporal feature pyramid (Learning-and-Imitation Module LIM + Spatio-Temporal Feature Pyramid Network STFPN)
- **Relevancia:** A state-of-the-art end-to-end deep model that jointly detects depression and anxiety from facial-expression dynamics (used as a DL baseline in the 'Faces of the Mind' study), serving as the high-capacity DL benchmark this thesis can compare against its interpretable handcrafted-feature approach.
- **Viabilidad (30fps/cara):** Likely feasible at 30fps (models expression dynamics, not micro-expressions), but as a deep end-to-end model it is harder to explain with SHAP/LIME and is data-hungry relative to ~252 clips, so it may overfit this small Colombian sample. Also note: VFEM is a mimicry/imitation paradigm, which differs from this thesis's free-speech monologue protocol.


## 4. Microexpresiones: metodos y datasets (CASME / SAMM / SMIC...)

*Sub-tema microexpression-methods -- 10 papers verificados*

### 28. CASME II: An Improved Spontaneous Micro-Expression Database and the Baseline Evaluation  `[PDF DESCARGADO]`
- **Autores:** Wen-Jing Yan, Xiaobai Li, Su-Jing Wang, Guoying Zhao, Yong-Jin Liu, Yu-Hsin Chen, Xiaolan Fu
- **Anio / Venue:** 2014 -- PLOS ONE 9(1): e86041
- **DOI:** https://doi.org/10.1371/journal.pone.0086041
- **Link:** https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0086041
- **PDF OA:** https://journals.plos.org/plosone/article/file?type=printable&id=10.1371/journal.pone.0086041
- **PDF local:** papers/04_028_casme_ii_an_improved_spontaneous_micro_e.pdf
- **Dataset:** CASME II (247 micro-expression clips, 26 subjects, 200 fps, AU + emotion labels)
- **Modalidad:** facial micro-expressions / Action Units
- **Relevancia:** Canonical AU+emotion-labeled micro-expression benchmark; supplies the facial-AU vocabulary the thesis can feasibly adopt for explainable depression/anxiety markers.
- **Viabilidad (30fps/cara):** Recorded at 200 fps for true micro-expression apex capture; the student's 30 fps phone video cannot replicate apex-window spotting, so use only for AU vocabulary/pretraining, not as a target protocol.

### 29. A Spontaneous Micro-expression Database: Inducement, Collection and Baseline  `[PDF DESCARGADO]`
- **Autores:** Xiaobai Li, Tomas Pfister, Xiaohua Huang, Guoying Zhao, Matti Pietikainen
- **Anio / Venue:** 2013 -- IEEE International Conference on Automatic Face & Gesture Recognition (FG 2013), pp. 1-6
- **DOI:** https://doi.org/10.1109/FG.2013.6553717
- **Link:** https://ieeexplore.ieee.org/document/6553717/
- **PDF OA:** https://tomas.pfister.fi/files/li2013microexpressions.pdf
- **PDF local:** papers/04_029_a_spontaneous_micro_expression_database_.pdf
- **Dataset:** SMIC (164 clips, 16 subjects; HS 100 fps, VIS and NIR near-infrared subsets)
- **Modalidad:** facial micro-expressions (high-speed + near-infrared)
- **Relevancia:** Seminal spontaneous micro-expression corpus establishing the positive/negative/surprise valence task; relevant as historical grounding for why the thesis must restrict itself to coarse facial dynamics.
- **Viabilidad (30fps/cara):** Relies on 100 fps high-speed cameras and a near-infrared subset; neither available in the student's 30 fps RGB phone data, so SMIC's acquisition protocol is not reproducible here.

### 30. SAMM: A Spontaneous Micro-Facial Movement Dataset  `[OA - solo link]`
- **Autores:** Adrian K. Davison, Cliff Lansley, Nicholas Costen, Kevin Tan, Moi Hoon Yap
- **Anio / Venue:** 2018 -- IEEE Transactions on Affective Computing, vol. 9, no. 1, pp. 116-129
- **DOI:** https://doi.org/10.1109/TAFFC.2016.2573832
- **Link:** https://ieeexplore.ieee.org/document/7492264/
- **PDF OA:** https://e-space.mmu.ac.uk/617069/
- **Dataset:** SAMM (159 micro-movements, 32 subjects, 200 fps, FACS-coded, diverse ethnicities)
- **Modalidad:** facial micro-expressions / FACS Action Units
- **Relevancia:** Most demographically diverse FACS-coded micro-expression dataset; supports justifying AU-based features for an under-represented rural Colombian cohort.
- **Viabilidad (30fps/cara):** Recorded at 200 fps; cannot be matched by 30 fps data, so it is a source of AU labels/pretraining, not an fps-comparable benchmark.

### 31. Video-based Facial Micro-Expression Analysis: A Survey of Datasets, Features and Algorithms  `[PDF DESCARGADO]`
- **Autores:** Xianye Ben, Yi Ren, Junping Zhang, Su-Jing Wang, Kidiyo Kpalma, Weixiao Meng, Yong-Jin Liu
- **Anio / Venue:** 2021 -- IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
- **DOI:** https://doi.org/10.1109/TPAMI.2021.3067464
- **Link:** https://arxiv.org/abs/2201.12728
- **PDF OA:** https://arxiv.org/pdf/2201.12728
- **PDF local:** papers/04_031_video_based_facial_micro_expression_anal.pdf
- **Dataset:** Introduces MMEW (300 micro-expressions, 900 macro-expressions, 6 basic emotions); surveys CASME, CASME II, SMIC, SAMM, CAS(ME)2
- **Modalidad:** facial micro- and macro-expressions
- **Relevancia:** Authoritative TPAMI survey mapping datasets, handcrafted vs deep features, and protocols; best single orientation reference for the video pivot, and MMEW (macro+micro) is the most 30 fps-compatible dataset it releases.
- **Viabilidad (30fps/cara):** Survey-level; notes most datasets use 100-200 fps, reinforcing that the student should target macro-expression and AU dynamics (feasible at 30 fps) rather than micro-expression spotting.

### 32. CAS(ME)3: A Third Generation Facial Spontaneous Micro-Expression Database With Depth Information and High Ecological Validity  `[OA - solo link]`
- **Autores:** Jingting Li, Zizhao Dong, Shaoyuan Lu, Su-Jing Wang, Wen-Jing Yan, Yinhuan Ma, Ye Liu, Changbing Huang, Xiaolan Fu
- **Anio / Venue:** 2023 -- IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 3, pp. 2782-2800
- **DOI:** https://doi.org/10.1109/TPAMI.2022.3174895
- **Link:** https://ieeexplore.ieee.org/document/9774929/
- **PDF OA:** http://melab.psych.ac.cn/publication/pami22.pdf
- **Dataset:** CAS(ME)3 (~80 h video, 8M+ frames, 1,109 micro- and 3,490 macro-expressions, RGB + depth, high-stakes deception subset)
- **Modalidad:** facial micro- and macro-expressions, RGB + depth (multimodal)
- **Relevancia:** Largest, most ecologically valid micro-expression database and a multimodal exemplar; its macro-expression and AU labels support the move toward feasible facial-dynamic features in naturalistic phone recordings.
- **Viabilidad (30fps/cara):** Micro-expression portion is high-fps and includes depth (RGB-D) the student does not have; the macro-expression and AU annotations are the transferable, 30 fps-feasible part.

### 33. A Survey of Automatic Facial Micro-Expression Analysis: Databases, Methods, and Challenges  `[PDF DESCARGADO]`
- **Autores:** Yee-Hui Oh, John See, Anh Cat Le Ngo, Raphael C.-W. Phan, Vishnu M. Baskaran
- **Anio / Venue:** 2018 -- Frontiers in Psychology, vol. 9, art. 1128
- **DOI:** https://doi.org/10.3389/fpsyg.2018.01128
- **Link:** https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2018.01128/full
- **PDF OA:** https://pmc.ncbi.nlm.nih.gov/articles/PMC6049018/
- **PDF local:** papers/04_033_a_survey_of_automatic_facial_micro_expre.pdf
- **Dataset:** Reviews CASME, CASME II, SMIC, SAMM, CAS(ME)2
- **Modalidad:** facial micro-expressions
- **Relevancia:** Widely cited cross-disciplinary review linking the psychology of micro-expressions to the spotting/recognition pipeline and clinical applications, grounding the thesis's clinical-interpretability framing for non-specialist health workers.
- **Viabilidad (30fps/cara):** Conceptual survey; explicitly notes high-frame-rate acquisition requirements the 30 fps phone data does not meet for true micro-expressions.

### 34. OFF-ApexNet on Micro-expression Recognition System  `[PDF DESCARGADO]`
- **Autores:** Y.S. Gan, Sze-Teng Liong, Wei-Chuen Yau, Yen-Chang Huang, Lit-Ken Tan
- **Anio / Venue:** 2019 -- Signal Processing: Image Communication, vol. 74, pp. 129-139
- **DOI:** https://doi.org/10.1016/j.image.2019.02.005
- **Link:** https://arxiv.org/abs/1805.08699
- **PDF OA:** https://arxiv.org/pdf/1805.08699
- **PDF local:** papers/04_034_off_apexnet_on_micro_expression_recognit.pdf
- **Dataset:** SMIC, CASME II, SAMM (cross-database LOSO)
- **Modalidad:** facial micro-expressions via optical-flow apex-frame CNN
- **Relevancia:** Influential apex-frame + optical-flow CNN baseline showing how onset-to-apex motion can be encoded compactly; the optical-flow-of-AU-region idea transfers to 30 fps expression-dynamics features and pairs naturally with SHAP/LIME over flow channels.
- **Viabilidad (30fps/cara):** Apex detection assumes high-fps capture of a brief micro-expression; at 30 fps the apex window is undersampled, so adapt the method to slower macro/AU dynamics rather than micro-expression apex spotting.

### 35. MESNet: A Convolutional Neural Network for Spotting Multi-Scale Micro-Expression Intervals in Long Videos  `[DE PAGO]`
- **Autores:** Su-Jing Wang, Ying He, Jingting Li, Xiaolan Fu
- **Anio / Venue:** 2021 -- IEEE Transactions on Image Processing, vol. 30, pp. 3956-3969
- **DOI:** https://doi.org/10.1109/TIP.2021.3064258
- **Link:** https://ieeexplore.ieee.org/document/9392303/
- **Dataset:** CAS(ME)2, SAMM Long Videos
- **Modalidad:** facial micro-expression spotting (2+1D spatiotemporal CNN)
- **Relevancia:** State-of-the-art deep spotting architecture for locating expression intervals in long untrimmed videos; directly relevant because the thesis's ~4.4-min clips require interval/segment spotting before per-segment classification.
- **Viabilidad (30fps/cara):** Trained/evaluated on high-fps long-video datasets; the spotting framework is conceptually transferable to 30 fps but will only reliably localize macro-expression-scale intervals, not sub-frame micro-expressions, at the student's frame rate.

### 36. MEGC2025: Micro-Expression Grand Challenge on Spot Then Recognize and Visual Question Answering  `[PDF DESCARGADO]`
- **Autores:** Xinqi Fan, Jingting Li, Moi Hoon Yap, Su-Jing Wang, John See, Adrian K. Davison, et al.
- **Anio / Venue:** 2025 -- Proceedings of the 33rd ACM International Conference on Multimedia (ACM MM 2025, MEGC workshop); preprint arXiv:2506.15298
- **DOI:** https://doi.org/10.1145/3746027.3762065
- **Link:** https://arxiv.org/abs/2506.15298
- **PDF OA:** https://arxiv.org/pdf/2506.15298
- **PDF local:** papers/04_036_megc2025_micro_expression_grand_challeng.pdf
- **Dataset:** CAS(ME)3 and challenge sets for Spot-Then-Recognize (ME-STR) and ME-VQA
- **Modalidad:** facial micro-expressions (spotting + recognition + visual question answering)
- **Relevancia:** Most recent installment of the field-defining benchmark series, formalizing the spot-then-recognize protocol/STRS metric; the new ME-VQA track aligns with the thesis's explainability goal of human-readable clinical justifications.
- **Viabilidad (30fps/cara):** Challenge data are high-fps micro-expression sets; the spot-then-recognize paradigm and STRS metric are reusable, but the 30 fps cohort can only enter a macro/AU-level analogue, not the true micro-expression tracks.

### 37. Catching Elusive Depression via Facial Micro-Expression Recognition  `[PDF DESCARGADO]`
- **Autores:** Xiaohui Chen, Tie Luo
- **Anio / Venue:** 2023 -- IEEE Communications Magazine, vol. 61, no. 10, pp. 30-36
- **DOI:** https://doi.org/10.1109/MCOM.001.2300003
- **Link:** https://arxiv.org/abs/2307.15862
- **PDF OA:** https://arxiv.org/pdf/2307.15862
- **PDF local:** papers/04_037_catching_elusive_depression_via_facial_m.pdf
- **Dataset:** DAIC-WOZ (clinical depression interviews); facial-landmark ROI features
- **Modalidad:** facial micro-expressions / facial-landmark ROIs for depression detection
- **Relevancia:** Closest analogue to the thesis goal: detects concealed depression from facial micro-expressions on interview video via a privacy-preserving, low-cost mobile-device pipeline, mirroring the rural-deployment and depression-screening aims.
- **Viabilidad (30fps/cara):** Targets micro-expressions but uses a landmark-ROI approach on standard interview video, so its ROI/landmark feature design is adaptable to 30 fps phone clips; frame it as expression-dynamics rather than true micro-expression timing.


## 5. Micro-GESTOS corporales para emocion/estres (iMiGUE / SMG / MiGA)

*Sub-tema micro-gesture-emotion -- 10 papers verificados*

### 38. iMiGUE: An Identity-free Video Dataset for Micro-Gesture Understanding and Emotion Analysis  `[PDF DESCARGADO]`
- **Autores:** Xin Liu, Henglin Shi, Haoyu Chen, Zitong Yu, Xiaobai Li, Guoying Zhao
- **Anio / Venue:** 2021 -- CVPR 2021 (IEEE/CVF Conference on Computer Vision and Pattern Recognition)
- **DOI:** https://doi.org/10.1109/CVPR46437.2021.01049
- **Link:** https://arxiv.org/abs/2107.00285
- **PDF OA:** https://arxiv.org/pdf/2107.00285
- **PDF local:** papers/05_038_imigue_an_identity_free_video_dataset_fo.pdf
- **Dataset:** iMiGUE (~18,499 MG samples, 32 micro-gesture categories, 359 long videos, 72 subjects from Grand Slam tennis press conferences)
- **Modalidad:** body micro-gesture; RGB video + 2D skeleton (OpenPose), identity-free (face/voice masked)
- **Relevancia:** The foundational large-scale benchmark for inferring hidden/suppressed emotional states from subtle body micro-gestures using only RGB video and 2D skeleton, directly grounding the thesis's pivot to a video modality with identity-free, clinically interpretable affect cues.
- **Viabilidad (30fps/cara):** Highly feasible: 30fps frontal phone video and 2D pose are exactly the inputs; no high-fps or IR needed.

### 39. SMG: A Micro-gesture Dataset Towards Spontaneous Body Gestures for Emotional Stress State Analysis  `[PDF DESCARGADO]`
- **Autores:** Haoyu Chen, Henglin Shi, Xin Liu, Xiaobai Li, Guoying Zhao
- **Anio / Venue:** 2023 -- International Journal of Computer Vision (IJCV), vol. 131
- **DOI:** https://doi.org/10.1007/s11263-023-01761-6
- **Link:** https://link.springer.com/article/10.1007/s11263-023-01761-6
- **PDF OA:** https://oulurepo.oulu.fi/bitstream/handle/10024/26850/nbnfi-fe202003248953.pdf?sequence=1&isAllowed=y
- **PDF local:** papers/05_039_smg_a_micro_gesture_dataset_towards_spon.pdf
- **Dataset:** SMG (3,692 MG clips of 17 MG classes, 40 subjects, two emotional stress states elicited via real/fake story-telling)
- **Modalidad:** spontaneous body micro-gesture; RGB + skeleton + depth (Kinect capture)
- **Relevancia:** Provides a second core micro-gesture dataset plus benchmarks and statistical evidence that subtle spontaneous body movements correlate with emotional STRESS states, aligning with detecting anxiety/depression-related arousal in the thesis.
- **Viabilidad (30fps/cara):** Mostly feasible; SMG was captured with depth (Kinect) but the MG classification methods run on RGB/2D-skeleton, matching the student's 30fps phone video.

### 40. Analyze Spontaneous Gestures for Emotional Stress State Recognition: A Micro-gesture Dataset and Analysis with Deep Learning  `[PDF DESCARGADO]`
- **Autores:** Haoyu Chen, Xin Liu, Xiaobai Li, Henglin Shi, Guoying Zhao
- **Anio / Venue:** 2019 -- FG 2019 (14th IEEE International Conference on Automatic Face & Gesture Recognition)
- **DOI:** https://doi.org/10.1109/FG.2019.8756513
- **Link:** https://ieeexplore.ieee.org/document/8756513
- **PDF OA:** https://oulurepo.oulu.fi/bitstream/handle/10024/26850/nbnfi-fe202003248953.pdf?sequence=1&isAllowed=y
- **PDF local:** papers/05_040_analyze_spontaneous_gestures_for_emotion.pdf
- **Dataset:** SMG (original conference version of the spontaneous micro-gesture dataset for emotional stress)
- **Modalidad:** spontaneous body micro-gesture; RGB + skeleton
- **Relevancia:** Seminal conference paper introducing the micro-gesture-to-emotional-stress paradigm with deep learning, establishing the conceptual basis the thesis uses to link body cues to hidden anxiety/stress.
- **Viabilidad (30fps/cara):** Feasible on standard video/skeleton; no high-fps or IR eye-tracking required.

### 41. Micro-gesture Recognition: A Comprehensive Survey of Datasets, Methods, and Challenges  `[OA - ver link]`
- **Autores:** Taorui Wang, Xun Lin, Yong Xu, Qilang Ye, Dan Guo, Sergio Escalera, Ghada Khoriba, Zitong Yu
- **Anio / Venue:** 2026 -- Machine Intelligence Research, vol. 23(2), pp. 308-330
- **DOI:** https://doi.org/10.1007/s11633-025-1629-x
- **Link:** https://www.mi-research.net/article/doi/10.1007/s11633-025-1629-x
- **Dataset:** Survey covering iMiGUE, SMG and related MGR datasets
- **Modalidad:** body micro-gesture (skeleton + RGB); reviews supervised, unsupervised, contrastive, multimodal-fusion and MLLM paradigms
- **Relevancia:** A comprehensive survey of micro-gesture recognition; a one-stop reference for the thesis's literature review, taxonomy of methods (incl. multimodal fusion and MLLM paradigms), and dataset/modality choices.
- **Viabilidad (30fps/cara):** N/A (survey); useful for selecting feasible 2D-skeleton/RGB methods over high-fps approaches.

### 42. Joint Skeletal and Semantic Embedding Loss for Micro-gesture Classification  `[PDF DESCARGADO]`
- **Autores:** Kun Li, Dan Guo, Guoliang Chen, Xinge Peng, Meng Wang
- **Anio / Venue:** 2023 -- MiGA Workshop & Challenge @ IJCAI 2023 (1st place, MG classification track); CEUR-WS Vol-3522
- **DOI:** https://doi.org/10.48550/arXiv.2307.10624
- **Link:** https://arxiv.org/abs/2307.10624
- **PDF OA:** https://arxiv.org/pdf/2307.10624
- **PDF local:** papers/05_042_joint_skeletal_and_semantic_embedding_lo.pdf
- **Dataset:** iMiGUE
- **Modalidad:** body micro-gesture; 3D-CNN skeleton-based with semantic label embedding loss
- **Relevancia:** Challenge-winning skeleton-based method showing how semantic label embeddings boost subtle micro-gesture classification; a strong feasible baseline for the thesis's video pipeline.
- **Viabilidad (30fps/cara):** Feasible: skeleton-based method runnable on 30fps phone video; no IR/high-fps needed.

### 43. Prototype Learning for Micro-gesture Classification  `[PDF DESCARGADO]`
- **Autores:** Guoliang Chen, Kun Li, Fei Wang, Zhiliang Wu, Dan Guo, Meng Wang (HFUT-VUT team)
- **Anio / Venue:** 2024 -- MiGA Workshop & Challenge @ IJCAI 2024 (1st place, MG classification track); CEUR-WS Vol-3848
- **DOI:** https://doi.org/10.48550/arXiv.2408.03097
- **Link:** https://arxiv.org/abs/2408.03097
- **PDF OA:** https://arxiv.org/pdf/2408.03097
- **PDF local:** papers/05_043_prototype_learning_for_micro_gesture_cla.pdf
- **Dataset:** iMiGUE
- **Modalidad:** body micro-gesture; cross-modal fusion (RGB + skeleton) with prototypical refinement
- **Relevancia:** Prototype-learning approach with cross-modal fusion and prototypical refinement that improves discrimination of subtle micro-gestures; transferable to the thesis's small-data emotion detection.
- **Viabilidad (30fps/cara):** Feasible: uses RGB + 2D skeleton at standard frame rates; no high-fps or IR sensors required.

### 44. MM-Gesture: Towards Precise Micro-Gesture Recognition through Multimodal Fusion  `[PDF DESCARGADO]`
- **Autores:** Jihao Gu, Fei Wang, Kun Li, Yanyan Wei, Zhiliang Wu, Dan Guo (HFUT-VUT team)
- **Anio / Venue:** 2025 -- MiGA Workshop & Challenge @ IJCAI 2025 (1st place, MG classification track); CEUR-WS Vol-4168
- **DOI:** https://doi.org/10.48550/arXiv.2507.08344
- **Link:** https://arxiv.org/abs/2507.08344
- **PDF OA:** https://arxiv.org/pdf/2507.08344
- **PDF local:** papers/05_044_mm_gesture_towards_precise_micro_gesture.pdf
- **Dataset:** iMiGUE (RGB branch pre-trained on MA-52)
- **Modalidad:** body micro-gesture; multimodal fusion of joint, limb, RGB, Taylor-series, optical-flow and depth video (PoseConv3D + Video Swin Transformer)
- **Relevancia:** Current SOTA (73.2% Top-1 on iMiGUE) demonstrating how multimodal fusion sharpens recognition of subtle gestures, informing the thesis's audio+video multimodal design.
- **Viabilidad (30fps/cara):** Partially feasible: joint/limb/RGB/optical-flow/Taylor branches run on 30fps video, but the depth-video branch needs a depth sensor the student lacks (can be dropped).

### 45. Hybrid-supervised Hypergraph-enhanced Transformer for Micro-gesture Based Emotion Recognition  `[PDF DESCARGADO]`
- **Autores:** Kun Li, Dan Guo, Pengyu Liu, Guoliang Chen, Meng Wang
- **Anio / Venue:** 2025 -- arXiv preprint (H2OFormer)
- **DOI:** https://doi.org/10.48550/arXiv.2507.14867
- **Link:** https://arxiv.org/abs/2507.14867
- **PDF OA:** https://arxiv.org/pdf/2507.14867
- **PDF local:** papers/05_045_hybrid_supervised_hypergraph_enhanced_tr.pdf
- **Dataset:** iMiGUE, SMG
- **Modalidad:** body micro-gesture; skeleton-based hypergraph-enhanced Transformer with self-supervised motion reconstruction
- **Relevancia:** Models higher-order joint relationships with a hypergraph Transformer and self-supervised motion reconstruction for emotion recognition, offering an interpretable, skeleton-only architecture well-suited to the thesis's explainability goals.
- **Viabilidad (30fps/cara):** Feasible: operates purely on 2D skeleton sequences; no high-fps or IR eye-tracking needed.

### 46. Micro-gesture Online Recognition using Learnable Query Points  `[PDF DESCARGADO]`
- **Autores:** Pengyu Liu, Kun Li, Fei Wang, Yanyan Wei, Junhui She, Dan Guo (HFUT-VUT team)
- **Anio / Venue:** 2024 -- MiGA Workshop & Challenge @ IJCAI 2024 (2nd place, MG online recognition track); CEUR-WS Vol-3848
- **DOI:** https://doi.org/10.48550/arXiv.2407.04490
- **Link:** https://arxiv.org/abs/2407.04490
- **PDF OA:** https://arxiv.org/pdf/2407.04490
- **PDF local:** papers/05_046_micro_gesture_online_recognition_using_l.pdf
- **Dataset:** iMiGUE / SMG (online recognition track)
- **Modalidad:** body micro-gesture; skeleton/RGB temporal localization with learnable query points (PointTAD + Mamba-MHSA block)
- **Relevancia:** Addresses ONLINE spotting/temporal localization of micro-gestures in long untrimmed videos, relevant because the thesis's clips are ~4.4 min and require localizing brief gestures within long video.
- **Viabilidad (30fps/cara):** Feasible: temporal detection on standard-frame-rate video using only RGB; no high-fps or IR required.

### 47. CLIP-MG: Guiding Semantic Attention with Skeletal Pose Features and RGB Data for Micro-Gesture Recognition on the iMiGUE Dataset  `[PDF DESCARGADO]`
- **Autores:** Santosh Patapati, Trisanth Srinivasan, Amith Adiraju
- **Anio / Venue:** 2025 -- arXiv preprint
- **DOI:** https://doi.org/10.48550/arXiv.2506.16385
- **Link:** https://arxiv.org/abs/2506.16385
- **PDF OA:** https://arxiv.org/pdf/2506.16385
- **PDF local:** papers/05_047_clip_mg_guiding_semantic_attention_with_.pdf
- **Dataset:** iMiGUE
- **Modalidad:** body micro-gesture; vision-language (frozen CLIP ViT-B/16) fused with OpenPose skeleton via pose-guided semantic attention and gated multimodal fusion
- **Relevancia:** Uses CLIP-style vision-language semantic guidance to make micro-gesture recognition more language-grounded, a promising route toward the clinically interpretable explanations the thesis needs for non-specialist rural health workers.
- **Viabilidad (30fps/cara):** Feasible: RGB + 2D skeleton on standard video; no high-fps or IR sensors needed.


## 6. Mirada / contacto visual / gaze aversion en depresion-ansiedad

*Sub-tema gaze-eye-contact-depression -- 9 papers verificados*

### 48. Visual Attention in Schizophrenia: Eye Contact and Gaze Aversion during Clinical Interactions  `[PDF DESCARGADO]`
- **Autores:** Alexandria K. Vail, Tadas Baltrušaitis, Luciana Pennant, Elizabeth S. Liebson, Justin T. Baker, Louis-Philippe Morency
- **Anio / Venue:** 2017 -- 7th International Conference on Affective Computing and Intelligent Interaction (ACII 2017), IEEE, pp. 490-497
- **DOI:** https://doi.org/10.1109/ACII.2017.8273644
- **Link:** https://akvail.github.io/pubs/vail_acii2017.pdf
- **PDF OA:** https://akvail.github.io/pubs/vail_acii2017.pdf
- **PDF local:** papers/06_048_visual_attention_in_schizophrenia_eye_co.pdf
- **Dataset:** Custom clinical interview corpus (inpatient psychotic-disorder unit), introspective vs. extrospective question segments
- **Modalidad:** Eye-gaze / eye-contact / gaze aversion (OpenFace on interview video)
- **Relevancia:** Foundational template for quantifying gaze aversion during clinical interview questions (introspective vs. extrospective) from OpenFace on standard interview video, directly transferable to building gaze-aversion markers from this thesis's interview-style face videos.
- **Viabilidad (30fps/cara):** Feasible: gaze direction and eye-contact/aversion estimated from standard-rate interview video via OpenFace; no high-fps or IR eye-tracker required.

### 49. Multimodal Depression Detection: Fusion Analysis of Paralinguistic, Head Pose and Eye Gaze Behaviors  `[OA - solo link]`
- **Autores:** Sharifa Alghowinem, Roland Goecke, Michael Wagner, Julien Epps, Matthew Hyett, Gordon Parker, Michael Breakspear
- **Anio / Venue:** 2018 -- IEEE Transactions on Affective Computing, Vol. 9, No. 4, pp. 478-490
- **DOI:** https://doi.org/10.1109/TAFFC.2016.2634527
- **Link:** https://researchprofiles.canberra.edu.au/en/publications/multimodal-depression-detection-fusion-analysis-of-paralinguistic
- **PDF OA:** https://www.researchgate.net/publication/311334815_Multimodal_Depression_DetectionFusion_Analysis_of_Paralinguistic_Head_Pose_and_Eye_Gaze_Behaviors
- **Dataset:** BlackDog (clinician-interview depression corpus)
- **Modalidad:** Audio-visual: paralinguistic (audio) + head pose + eye gaze
- **Relevancia:** Closest blueprint for this thesis's pivot: it fuses acoustic/paralinguistic features (the student's current audio modality) with head pose and eye gaze from interview video, showing which nonverbal cues best discriminate depression and how to fuse them.
- **Viabilidad (30fps/cara):** Feasible: head pose and eye-gaze statistics computed from ordinary interview video; gaze features (look-down duration, gaze angles) do not require IR or high-fps hardware.

### 50. Eye Movement Analysis for Depression Detection  `[OA - solo link]`
- **Autores:** Sharifa Alghowinem, Roland Goecke, Michael Wagner, Gordon Parker, Michael Breakspear
- **Anio / Venue:** 2013 -- 20th IEEE International Conference on Image Processing (ICIP 2013), pp. 4220-4224
- **DOI:** https://doi.org/10.1109/ICIP.2013.6738869
- **Link:** https://www.semanticscholar.org/paper/Eye-movement-analysis-for-depression-detection-Alghowinem-G%C3%B6cke/59100b071f0ac84a8543a50a7055ec5c7aa14817
- **PDF OA:** https://www.researchgate.net/publication/259931796_Eye_movement_analysis_for_depression_detection
- **Dataset:** BlackDog depression interview corpus
- **Modalidad:** Eye movement / gaze (look-down duration) and blink behavior from face video (Active Appearance Models)
- **Relevancia:** Seminal demonstration that simple eye-movement features from face video (longer downward gaze, shorter blink intervals) discriminate depression, providing interpretable, low-cost gaze markers well matched to this thesis's data.
- **Viabilidad (30fps/cara):** Feasible: uses standard-frame-rate face video; blink interval and gaze-direction features computable at 30 fps without IR eye-tracking.

### 51. Reading Between the Frames: Multi-Modal Depression Detection in Videos from Non-Verbal Cues  `[PDF DESCARGADO]`
- **Autores:** David Gimeno-Gómez, Ana-Maria Bucur, Adrian Cosma, Carlos-D. Martínez-Hinarejos, Paolo Rosso
- **Anio / Venue:** 2024 -- Advances in Information Retrieval, ECIR 2024, LNCS vol. 14610 (verify; ECIR'24 spans LNCS 14608-14612), Springer, pp. 191-209
- **DOI:** https://doi.org/10.1007/978-3-031-56027-9_12
- **Link:** https://arxiv.org/abs/2401.02746
- **PDF OA:** https://arxiv.org/pdf/2401.02746
- **PDF local:** papers/06_051_reading_between_the_frames_multi_modal_d.pdf
- **Dataset:** D-Vlog, DAIC-WOZ, and additional video depression benchmarks (in-the-wild / interview videos)
- **Modalidad:** Multimodal temporal: audio speech embeddings + face emotion + face/body/hand landmarks + gaze and blinking
- **Relevancia:** Recent SOTA pipeline showing that adding high-level nonverbal cues including gaze and blinking to noisy real-world phone-style video boosts depression detection, mirroring this thesis's plan to add gaze/blink markers to weak audio-only results; code is open-source.
- **Viabilidad (30fps/cara):** Feasible: explicitly designed for noisy in-the-wild video and extracts gaze/blink from standard-rate frames; no IR or high-fps eye-tracking needed.

### 52. Explainable Depression Detection via Head Motion Patterns  `[PDF DESCARGADO]`
- **Autores:** Monika Gahalawat, Raul Fernandez Rojas, Tanaya Guha, Ramanathan Subramanian, Roland Goecke
- **Anio / Venue:** 2023 -- 25th ACM International Conference on Multimodal Interaction (ICMI 2023), pp. 261-270
- **DOI:** https://doi.org/10.1145/3577190.3614130
- **Link:** https://arxiv.org/abs/2307.12241
- **PDF OA:** https://arxiv.org/pdf/2307.12241
- **PDF local:** papers/06_052_explainable_depression_detection_via_hea.pdf
- **Dataset:** BlackDog and AVEC2013 depression datasets
- **Modalidad:** Head pose / head-motion units (kinemes) from interview video; explainable (XAI) framing
- **Relevancia:** Directly supports the thesis's XAI thread by deriving interpretable head-motion 'kineme' biomarkers for depression that a non-specialist could understand, complementing gaze markers from the same 30 fps frontal-face video.
- **Viabilidad (30fps/cara):** Feasible: head-motion kinemes extracted from standard-rate video head-pose tracking; no specialized eye-tracking hardware required.

### 53. Beyond Questionnaires: Video Analysis for Social Anxiety Detection  `[PDF DESCARGADO]`
- **Autores:** Nilesh Kumar Sahu, Nandigramam Sai Harshit, Rishabh Uikey, Haroon R. Lone
- **Anio / Venue:** 2025 -- arXiv preprint arXiv:2501.05461 (IISER Bhopal)
- **DOI:** https://doi.org/10.48550/arXiv.2501.05461
- **Link:** https://arxiv.org/abs/2501.05461
- **PDF OA:** https://arxiv.org/pdf/2501.05461
- **PDF local:** papers/06_053_beyond_questionnaires_video_analysis_for.pdf
- **Dataset:** Custom: 92 participants, impromptu speech recorded on a low-cost phone camera
- **Modalidad:** Video: eye gaze (pitch/yaw/roll/gaze angle), facial Action Units (OpenFace), head pose (Py-Feat), body pose (MediaPipe)
- **Relevancia:** Almost an exact methodological match to this thesis's setup (low-cost phone video, frontal participants), extracting gaze, AUs, and head/body pose to detect social anxiety, validating the chosen feasible feature set for anxiety markers.
- **Viabilidad (30fps/cara):** Feasible: built entirely on low-cost phone-camera video with OpenFace/Py-Feat/MediaPipe gaze, AU, and pose features; no high-fps or IR tracking.

### 54. Towards Automatic Detection of Social Anxiety Disorder via Gaze Interaction  `[OA - solo link]`
- **Autores:** Sara Shafique, Iftikhar Ahmed Khan, Sajid Shah, Waqas Jadoon, Rab Nawaz Jadoon, Mohammed Elaffendi
- **Anio / Venue:** 2022 -- Applied Sciences (MDPI), Vol. 12, No. 23, Article 12298
- **DOI:** https://doi.org/10.3390/app122312298
- **Link:** https://www.mdpi.com/2076-3417/12/23/12298
- **PDF OA:** https://www.mdpi.com/2076-3417/12/23/12298/pdf
- **Dataset:** Custom: 50 participants, webcam-based gaze interaction/avoidance recordings
- **Modalidad:** Eye gaze interaction/avoidance (Haar Cascade gaze detection) with decision-tree classifiers
- **Relevancia:** Open-access study showing webcam-derived gaze-interaction percentages decline with social-anxiety severity (80% accuracy), giving an interpretable, ordinal gaze marker analogous to the thesis's GAD-7 severity grading; decision-tree approach also fits the XAI thread.
- **Viabilidad (30fps/cara):** Feasible: uses ordinary webcam video and lightweight Haar-cascade gaze detection; interpretable decision-tree approach also fits the XAI thread; no IR/high-fps needed.

### 55. Social Anxiety is Related to Reduced Face Gaze During a Naturalistic Social Interaction  `[DE PAGO]`
- **Autores:** Jiemiao Chen, Esther van den Bos, Julian D. Karch, P. Michiel Westenberg
- **Anio / Venue:** 2023 -- Anxiety, Stress, & Coping (Taylor & Francis), Vol. 36, No. 4, pp. 460-474
- **DOI:** https://doi.org/10.1080/10615806.2022.2125961
- **Link:** https://pubmed.ncbi.nlm.nih.gov/36153759/
- **Dataset:** Custom: face-to-face getting-acquainted conversation, Liebowitz Social Anxiety Scale labels
- **Modalidad:** Eye-gaze to the face (Tobii eye-tracking glasses) during dyadic conversation, moderated by speaking/listening and topic intimacy
- **Relevancia:** Provides the clinical/psychological grounding that higher social anxiety reduces face gaze, especially under intimate or listening conditions, justifying gaze-aversion as a behavioral marker and informing how to segment the thesis's interview by question type.
- **Viabilidad (30fps/cara):** Partial: ground-truth mechanism is sound, but this study used wearable Tobii eye-tracking glasses; for this thesis the equivalent face-gaze must be approximated from 30 fps frontal video (coarse gaze/eye-contact), not precise gaze tracking.

### 56. The Distress Analysis Interview Corpus of Human and Computer Interviews  `[OA - solo link]`
- **Autores:** Jonathan Gratch, Ron Artstein, Gale Lucas, Giota Stratou, Stefan Scherer, Angela Nazarian, Rachel Wood, Jill Boberg, David DeVault, Stacy Marsella, David Traum, Skip Rizzo, Louis-Philippe Morency
- **Anio / Venue:** 2014 -- Proceedings of the 9th International Conference on Language Resources and Evaluation (LREC'14), pp. 3123-3128
- **Link:** https://aclanthology.org/L14-1421.pdf
- **PDF OA:** https://aclanthology.org/L14-1421.pdf
- **Dataset:** DAIC / DAIC-WOZ (audiovisual clinical interviews for depression, anxiety, PTSD with PHQ/PCL labels)
- **Modalidad:** Audio-visual clinical interviews (audio, video, transcripts; downstream gaze/head-pose/AU features available via OpenFace)
- **Relevancia:** The foundational benchmark dataset for audiovisual depression/anxiety detection from clinical interviews, defining the interview-plus-nonverbal-cue paradigm (including gaze and head pose) that this thesis replicates on Spanish-speaking rural youth.
- **Viabilidad (30fps/cara):** Feasible as a reference corpus; its visual features (gaze, head pose, AUs) are derived from standard interview video, matching what is extractable from this thesis's 30 fps clips.


## 7. Parpadeo / pupilometria / marcadores oculares

*Sub-tema blink-pupil-biomarkers -- 10 papers verificados*

### 57. Altered pupil light and darkness reflex and eye-blink responses in late-life depression  `[OA - solo link]`
- **Autores:** Yao-Tung Lee, Yi-Hsuan Chang, Hsu-Jung Tsai, Shu-Ping Chao, David Yen-Ting Chen, Jui-Tai Chen, Yih-Giun Cherng, Chin-An Wang
- **Anio / Venue:** 2024 -- BMC Geriatrics
- **DOI:** https://doi.org/10.1186/s12877-024-05034-w
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11194921/
- **PDF OA:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11194921/
- **Dataset:** 25 late-life depression patients, 29 older healthy controls, 25 younger healthy controls
- **Modalidad:** eye-gaze / blink / pupillometry (video-based eye tracker)
- **Relevancia:** Directly shows elevated blink rate and dampened pupil constriction as biomarkers correlating with depression severity, validating blink as a candidate ocular feature for the thesis.
- **Viabilidad (30fps/cara):** Pupil light/darkness reflex measured with EyeLink-1000 at 500Hz under controlled luminance cannot be reproduced from 30fps phone video; only the blink-rate finding is feasible to replicate.

### 58. Pupillary reactivity to sad stimuli as a biomarker of depression risk: Evidence from a prospective study of children  `[OA - solo link]`
- **Autores:** Katie L. Burkhouse, Greg J. Siegle, Mary L. Woody, Anastacia Y. Kudinova, Brandon E. Gibb
- **Anio / Venue:** 2015 -- Journal of Abnormal Psychology
- **DOI:** https://doi.org/10.1037/abn0000072
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC4573844/
- **PDF OA:** https://pmc.ncbi.nlm.nih.gov/articles/PMC4573844/
- **Dataset:** 47 mother-child dyads (children 8-14), mothers with MDD history; 24-month longitudinal follow-up
- **Modalidad:** pupillometry to emotional face stimuli (task-evoked)
- **Relevancia:** Foundational prospective evidence that pupil dilation to sad stimuli predicts depression onset, motivating pupillary arousal as an interpretable biomarker (as task-evoked reactivity).
- **Viabilidad (30fps/cara):** Requires task-evoked pupillometry with controlled stimuli and an infrared eye-tracker; pupil-diameter measurement is infeasible from 30fps vertical phone video, so relevant as conceptual/prior support only.

### 59. Diagnosing and tracking depression based on eye movement in response to virtual reality  `[OA - solo link]`
- **Autores:** Zhiguo Zheng, Lijuan Liang, Xiong Luo, Jie Chen, Meirong Lin, Guanjun Wang, Chenyang Xue
- **Anio / Venue:** 2024 -- Frontiers in Psychiatry
- **DOI:** https://doi.org/10.3389/fpsyt.2024.1280935
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10875075/
- **PDF OA:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10875075/
- **Dataset:** 69 participants (24 controls, 45 with depressive symptoms), PHQ-9 labeled
- **Modalidad:** eye movement (saccades, fixations) via VR eye tracker; ML (MLP/XGBoost)
- **Relevancia:** PHQ-9-labeled ML pipeline where saccade/fixation features dominate importance, paralleling the thesis's PHQ-9 labels and feature-attribution/XAI angle.
- **Viabilidad (30fps/cara):** Uses an HTC Vive Pro VR headset with embedded eye tracker and structured VR stimuli; precise saccade/fixation metrics need >30fps and the VR paradigm is unavailable, so only coarse gaze/fixation proxies transfer.

### 60. Eye Movement Abnormalities in Major Depressive Disorder  `[PDF DESCARGADO]`
- **Autores:** Jun Takahashi, Yoji Hirano, Kenichiro Miura, Kentaro Morita, Michiko Fujimoto, Hidenaga Yamamori, Yuka Yasuda, et al.
- **Anio / Venue:** 2021 -- Frontiers in Psychiatry
- **DOI:** https://doi.org/10.3389/fpsyt.2021.673443
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC8382962/
- **PDF OA:** https://pmc.ncbi.nlm.nih.gov/articles/PMC8382962/
- **PDF local:** papers/07_060_eye_movement_abnormalities_in_major_depr.pdf
- **Dataset:** Case-control MDD vs healthy controls; free-viewing, smooth pursuit, fixation tasks
- **Modalidad:** eye movement (saccade velocity, scanpath length, fixation)
- **Relevancia:** Clinically validated MDD eye-movement markers (shorter scanpath, reduced peak saccade velocity) giving target ocular features to engineer and explain in the multimodal model.
- **Viabilidad (30fps/cara):** Peak saccade velocity and precise smooth-pursuit gain require high-frequency eye tracking unavailable at 30fps; scanpath/fixation-density proxies and gaze coarseness are the feasible takeaways.

### 61. Eye tracking evidence of threat-related attentional bias in anxiety- and fear-related disorders: A systematic review and meta-analysis  `[DE PAGO]`
- **Autores:** Kate Clauss, Julia Y. Gorday, Joseph R. Bardeen
- **Anio / Venue:** 2022 -- Clinical Psychology Review
- **DOI:** https://doi.org/10.1016/j.cpr.2022.102142
- **Link:** https://doi.org/10.1016/j.cpr.2022.102142
- **Dataset:** Meta-analysis of 40 articles on anxiety/fear disorders
- **Modalidad:** eye-gaze attentional bias (reflexive orienting, attention maintenance)
- **Relevancia:** Meta-analytic evidence linking eye-tracking attentional bias to anxiety/fear severity (GAD-7 side of the thesis), establishing gaze/dwell behavior as a quantifiable anxiety ocular marker.
- **Viabilidad (30fps/cara):** Underlying studies use stimulus-driven dot-probe/free-viewing paradigms with dedicated eye trackers; supports rationale rather than a directly replicable feature from naturalistic 30fps video.

### 62. Systematic Review and Meta-Analysis: Eye-Tracking of Attention to Threat in Child and Adolescent Anxiety  `[DE PAGO]`
- **Autores:** Stephen Lisk, Ayesha Vaswani, Marian Linetzky, Yair Bar-Haim, Jennifer Y. F. Lau
- **Anio / Venue:** 2020 -- Journal of the American Academy of Child & Adolescent Psychiatry
- **DOI:** https://doi.org/10.1016/j.jaac.2019.06.006
- **Link:** https://doi.org/10.1016/j.jaac.2019.06.006
- **Dataset:** Meta-analysis of eye-tracking studies in youth anxiety
- **Modalidad:** eye-gaze attentional bias (vigilance/avoidance of threat)
- **Relevancia:** Youth-focused meta-analysis on anxiety-related gaze biases relevant to the thesis's 18-28 age band, showing threat-avoidance rather than vigilance patterns informing expected gaze behavior.
- **Viabilidad (30fps/cara):** Based on task-based threat-stimulus eye-tracking, not naturalistic video; informs interpretation of coarse gaze/eye-contact features rather than providing a reproducible method.

### 63. Emotional stimulation processing characteristics in depression: Meta-analysis of eye tracking findings  `[OA - solo link]`
- **Autores:** Mengmeng Huang, Bingyan Gong, et al.
- **Anio / Venue:** 2023 -- Frontiers in Psychology
- **DOI:** https://doi.org/10.3389/fpsyg.2022.1089654
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC9880408/
- **PDF OA:** https://pmc.ncbi.nlm.nih.gov/articles/PMC9880408/
- **Dataset:** Meta-analysis of 14 case-control eye-tracking studies in depression (1167 participants)
- **Modalidad:** eye-gaze attentional bias toward emotional stimuli (fixation duration/count)
- **Relevancia:** Consolidates depression-side ocular evidence (negative attentional bias, altered fixation patterns), anchoring which gaze/fixation features are expected discriminative for depression.
- **Viabilidad (30fps/cara):** Pooled studies rely on emotional-stimulus viewing with dedicated eye trackers; conceptual support only, as the bias paradigm and high-precision fixation metrics are not obtainable from 30fps frontal video.

### 64. Dopamine, depressive symptoms, and decision-making: the relationship between spontaneous eye blink rate and depressive symptoms predicts Iowa Gambling Task performance  `[OA - solo link]`
- **Autores:** Kaileigh A. Byrne, Dominique D. Norris, Darrell A. Worthy
- **Anio / Venue:** 2016 -- Cognitive, Affective, & Behavioral Neuroscience
- **DOI:** https://doi.org/10.3758/s13415-015-0377-0
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC5042144/
- **PDF OA:** https://pmc.ncbi.nlm.nih.gov/articles/PMC5042144/
- **Dataset:** 104 undergraduates, depressive symptom self-report
- **Modalidad:** spontaneous eye blink rate (EBR) as dopaminergic marker
- **Relevancia:** Establishes mechanistic rationale that spontaneous blink rate indexes striatal dopamine and interacts with depressive symptoms, supporting blink rate as a theoretically grounded, interpretable biomarker complementing acoustic features.
- **Viabilidad (30fps/cara):** Spontaneous blink rate is fully measurable from the student's 30fps frontal-face video using EAR/landmark methods; no high-fps or IR hardware needed (a strong feasible feature).

### 65. Real-Time Eye Blink Detection using Facial Landmarks  `[PDF DESCARGADO]`
- **Autores:** Tereza Soukupova, Jan Cech
- **Anio / Venue:** 2016 -- 21st Computer Vision Winter Workshop (CVWW)
- **Link:** https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
- **PDF OA:** https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
- **PDF local:** papers/07_065_real_time_eye_blink_detection_using_faci.pdf
- **Dataset:** Eyeblink8, Talking Face (standard blink-detection benchmarks)
- **Modalidad:** blink detection from RGB video via Eye Aspect Ratio (EAR) on facial landmarks
- **Relevancia:** Seminal, widely-used EAR method for detecting blinks from ordinary RGB video and landmarks, providing the practical extraction pipeline to compute blink-rate features from 30fps phone video.
- **Viabilidad (30fps/cara):** Highly feasible: EAR works with standard cameras and head-pose-robust landmarks; directly applicable to 30fps 1080p frontal video with no special hardware.

### 66. Hyper-scanning and hyper-pursuit define eye movement biomarkers of anxiety disorders  `[DE PAGO]`
- **Autores:** Dan Zhang, Yu Li, Lihua Xu, Yangyang Xu, Xu Liu, Wensi Zheng, Yawen Hong, Jinyang Zhao, Yanyan Wei, Huiru Cui, Haichun Liu, Tianhong Zhang, Jijun Wang
- **Anio / Venue:** 2026 -- The British Journal of Psychiatry (Cambridge University Press)
- **DOI:** https://doi.org/10.1192/bjp.2026.10626
- **Link:** https://doi.org/10.1192/bjp.2026.10626
- **Dataset:** 91 patients with anxiety disorders, 118 with depressive disorders, 98 healthy controls (free viewing + smooth pursuit + fixation stability)
- **Modalidad:** eye movement (saccade frequency, scanpath length, smooth pursuit velocity gain)
- **Relevancia:** Anxiety-specific eye-movement biomarkers (increased saccade frequency/path length during free viewing, hyper-pursuit) that dissociate anxiety from depression, valuable for the GAD-7 dimension and multimodal discrimination.
- **Viabilidad (30fps/cara):** Saccade-frequency and smooth-pursuit metrics need higher-frequency tracking than 30fps; only coarse free-viewing scanning behavior is approachable from the student's video, so treat as supporting evidence.


## 8. Pose y movimiento de cabeza (retardo psicomotor)

*Sub-tema head-pose-motion-depression -- 10 papers verificados*

### 67. Head Pose and Movement Analysis as an Indicator of Depression  `[OA - solo link]`
- **Autores:** Sharifa Alghowinem, Roland Goecke, Michael Wagner, Gordon Parker, Michael Breakspear
- **Anio / Venue:** 2013 -- Humaine Association Conference on Affective Computing and Intelligent Interaction (ACII 2013), IEEE, pp. 283-288
- **DOI:** https://doi.org/10.1109/ACII.2013.53
- **Link:** https://dl.acm.org/doi/abs/10.1109/ACII.2013.53
- **PDF OA:** https://www.researchgate.net/publication/259932019_Head_Pose_and_Movement_Analysis_as_an_Indicator_of_Depression
- **Dataset:** BlackDog (Australian English clinical interview videos, depressed vs. controls)
- **Modalidad:** head pose / head movement (3DoF: pitch, yaw, roll velocities and direction changes)
- **Relevancia:** Foundational study showing head pose and movement alone discriminate depressed from non-depressed (71.2% avg), giving the thesis a validated, interpretable head-motion feature set to add to its weak audio-only model.
- **Viabilidad (30fps/cara):** Highly feasible at 30fps; uses an AAM/3D face model on standard frontal-face video with no high-fps or IR requirement, matching the 1080x1920 30fps clips.

### 68. Nonverbal Social Withdrawal in Depression: Evidence from manual and automatic analyses  `[OA - solo link]`
- **Autores:** Jeffrey M. Girard, Jeffrey F. Cohn, Mohammad H. Mahoor, S. Mohammad Mavadati, Zakia Hammal, Dean P. Rosenwald
- **Anio / Venue:** 2014 -- Image and Vision Computing, Vol. 32(10), pp. 641-647
- **DOI:** https://doi.org/10.1016/j.imavis.2013.12.007
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC4217695/
- **PDF OA:** https://pmc.ncbi.nlm.nih.gov/articles/PMC4217695/
- **Dataset:** Clinical depression treatment trial (33 adults; 19 responders, 38 interviews), facial AUs + head motion
- **Modalidad:** head motion (amplitude/velocity) + facial Action Units (AU12, AU14, AU15)
- **Relevancia:** Seminal evidence that high depression severity produces diminished head-motion amplitude/velocity (and AU changes) that increase on recovery, directly motivating the thesis's head-motion + AU multimodal pivot and a clinically explainable 'social withdrawal' interpretation.
- **Viabilidad (30fps/cara):** Feasible; head amplitude/velocity and AUs are extractable from standard-rate frontal video, no high-fps/IR needed.

### 69. Cross-Cultural Detection of Depression from Nonverbal Behaviour  `[OA - solo link]`
- **Autores:** Sharifa Alghowinem, Roland Goecke, Jeffrey F. Cohn, Michael Wagner, Gordon Parker, Michael Breakspear
- **Anio / Venue:** 2015 -- 11th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2015) and Workshops
- **DOI:** https://doi.org/10.1109/FG.2015.7163113
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC4955623/
- **PDF OA:** https://pmc.ncbi.nlm.nih.gov/articles/PMC4955623/
- **Dataset:** Three datasets: BlackDog (Australian English), Pitt (American English), AVEC (German)
- **Modalidad:** head pose (3DoF) + eye activity (gaze, blink)
- **Relevancia:** Directly addresses cross-cultural generalisation of head-pose and eye/gaze depression cues across three languages/cultures, the closest precedent for whether such markers transfer to the thesis's under-studied Spanish-speaking rural Colombian cohort.
- **Viabilidad (30fps/cara):** Feasible; head pose and coarse gaze/blink are 30fps-compatible. Note true microsaccade eye-tracking is not used here either, consistent with the student's constraints.

### 70. Detecting Depression Severity by Interpretable Representations of Motion Dynamics  `[OA - solo link]`
- **Autores:** Anis Kacem, Zakia Hammal, Mohamed Daoudi, Jeffrey Cohn
- **Anio / Venue:** 2018 -- 13th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2018) Workshops (FGAHI), pp. 739-745
- **DOI:** https://doi.org/10.1109/FG.2018.00116
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC6157749/
- **PDF OA:** https://pmc.ncbi.nlm.nih.gov/articles/PMC6157749/
- **Dataset:** 126 clinical interview sessions from 49 participants with chronic depression, severity by Hamilton Rating Scale (HRSD)
- **Modalidad:** head motion (Lie-algebra 3D rotation: yaw/roll/pitch velocity/acceleration) + facial landmark barycentric dynamics
- **Relevancia:** Provides an interpretable kinematic representation (velocity/acceleration of head rotation) for psychomotor retardation that aligns with the thesis's XAI goal of clinically readable motion features for non-specialist health workers.
- **Viabilidad (30fps/cara):** Feasible at 30fps; landmark and 3D head-rotation kinematics do not require high-fps or IR hardware.

### 71. Automated Measurement of Head Movement Synchrony during Dyadic Depression Severity Interviews  `[OA - solo link]`
- **Autores:** Shalini Bhatia, Roland Goecke, Zakia Hammal, Jeffrey F. Cohn
- **Anio / Venue:** 2019 -- 14th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2019)
- **DOI:** https://doi.org/10.1109/FG.2019.8756509
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC6863512/
- **PDF OA:** https://pmc.ncbi.nlm.nih.gov/articles/PMC6863512/
- **Dataset:** 49 MDD patients across 4 visits (weeks 1/7/13/21), 113 sessions, HRSD severity
- **Modalidad:** head movement (pitch/yaw angle synchrony in patient-therapist dyads, windowed cross-correlation)
- **Relevancia:** Cautionary result on interpersonal head-movement synchrony in patient-therapist dyads, warning the thesis to favor individual head-motion energy/kinematics over interpersonal-synchrony features given its single-participant clips.
- **Viabilidad (30fps/cara):** Individual head-pitch/yaw extraction is feasible at 30fps; however the dyadic synchrony method itself is NOT applicable - the thesis has one participant per clip with no interviewer feed.

### 72. Explainable Depression Detection via Head Motion Patterns  `[PDF DESCARGADO]`
- **Autores:** Monika Gahalawat, Raul Fernandez Rojas, Tanaya Guha, Ramanathan Subramanian, Roland Goecke
- **Anio / Venue:** 2023 -- 25th ACM International Conference on Multimodal Interaction (ICMI 2023)
- **DOI:** https://doi.org/10.1145/3577190.3614130
- **Link:** https://arxiv.org/abs/2307.12241
- **PDF OA:** https://arxiv.org/pdf/2307.12241
- **PDF local:** papers/08_072_explainable_depression_detection_via_hea.pdf
- **Dataset:** BlackDog and AVEC2013
- **Modalidad:** head motion (elementary head-motion units called 'kinemes')
- **Relevancia:** The single most on-point paper: an explainable head-motion approach (interpretable 'kineme' patterns, peak F1 0.79/0.82 on BlackDog/AVEC2013) that fuses the thesis's two pillars - head-movement biomarkers plus XAI interpretability for clinicians.
- **Viabilidad (30fps/cara):** Feasible at 30fps; kineme extraction relies on standard head-pose time series, no high-fps/IR needed.

### 73. Measuring Anxiety Levels with Head Motion Patterns in Severe Depression Population  `[PDF DESCARGADO]`
- **Autores:** Fouad Boutaleb, Emery Pierson, Nicolas Doudeau, Clemence Nineuil, Ali Amad, Mohamed Daoudi
- **Anio / Venue:** 2025 -- 19th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2025); also arXiv:2502.08813 (cs.CV)
- **DOI:** https://doi.org/10.48550/arXiv.2502.08813
- **Link:** https://arxiv.org/abs/2502.08813
- **PDF OA:** https://arxiv.org/pdf/2502.08813
- **PDF local:** papers/08_073_measuring_anxiety_levels_with_head_motio.pdf
- **Dataset:** CALYPSO Depression Dataset (32 severe-depression patients, 50/50 gender, predominantly French; informal clinical interview videos)
- **Modalidad:** head motion (speed, acceleration, angular displacement; GMM moving/steady-state segmentation)
- **Relevancia:** Rare paper predicting ANXIETY (not just depression) from head motion in a depressed population (psychological anxiety MAE 0.31, R^2 0.87), directly supporting the thesis's joint GAD-7 anxiety + PHQ-9 depression targets.
- **Viabilidad (30fps/cara):** Feasible at 30fps; speed/acceleration/angular-displacement features derive from ordinary head-pose tracking. Note weaker performance on somatic anxiety (MAE 0.47, R^2 0.53).

### 74. On the Validity of Head Motion Patterns as Generalisable Depression Biomarkers  `[PDF DESCARGADO]`
- **Autores:** Monika Gahalawat, Maneesh Bilalpur, Raul Fernandez Rojas, Jeffrey F. Cohn, Roland Goecke, Ramanathan Subramanian
- **Anio / Venue:** 2025 -- arXiv preprint (cs.CV), arXiv:2505.23427 (posted 29 May 2025)
- **DOI:** https://doi.org/10.48550/arXiv.2505.23427
- **Link:** https://arxiv.org/abs/2505.23427
- **PDF OA:** https://arxiv.org/pdf/2505.23427
- **PDF local:** papers/08_074_on_the_validity_of_head_motion_patterns_.pdf
- **Dataset:** Three cross-cultural depression datasets: AVEC2013 (German), BlackDog (Australian), Pitt (American); cross-dataset evaluation
- **Modalidad:** head motion (kineme-based patterns vs. raw head-motion descriptors, cross-dataset generalisability analysis)
- **Relevancia:** Critically probes whether head-motion depression biomarkers transfer across datasets/cultures, a direct methodological caution for the thesis claiming validity on a new Spanish-speaking rural Colombian cohort.
- **Viabilidad (30fps/cara):** Feasible at 30fps; head-motion-only analysis with no high-fps/IR requirement.

### 75. Digital assessment of nonverbal behaviors forecasts first onset of depression  `[OA - solo link]`
- **Autores:** Sekine Ozturk, Scott Feltman, Daniel N. Klein, Roman Kotov, Aprajita Mohanty
- **Anio / Venue:** 2024 -- Psychological Medicine, Vol. 54(12), pp. 3507-3518, Cambridge University Press
- **DOI:** https://doi.org/10.1017/S0033291724002010
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11496224/
- **PDF OA:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11496224/
- **Dataset:** 359 never-depressed adolescent females (mean age 14.38, range 13-15), 3-year longitudinal follow-up (Suffolk County NY), FaceReader 8.0 facial analysis
- **Modalidad:** head motion (frame-by-frame x/y/z Euclidean distance) + facial Action Units (FaceReader 8.0); key markers: head movement, AU4, AU26, AU43
- **Relevancia:** Clinically validated demonstration that head-movement + AU features forecast depression onset (AUC 0.70 nonverbal-only, 0.78 with risk factors), supporting the thesis's head-motion + AU fusion. CAUTION: cohort is adolescents 13-15, NOT 18-28, so it does NOT support the thesis's specific 18-28 age focus.
- **Viabilidad (30fps/cara):** Feasible; standard recorded interview video and commercial AU/head-motion extraction (FaceReader), no high-fps/IR. FaceReader is proprietary; OpenFace is a free 30fps-compatible substitute. AU43 (eyes closed/blink) at 30fps is coarse but obtainable.

### 76. Explainable Depression Detection via Head Motion Patterns (extended arXiv version)  `[PDF DESCARGADO]`
- **Autores:** Monika Gahalawat, Raul Fernandez Rojas, Tanaya Guha, Ramanathan Subramanian, Roland Goecke
- **Anio / Venue:** 2023 -- arXiv preprint (cs.CV), arXiv:2307.12241 - companion/open-access version of the ICMI 2023 paper
- **DOI:** https://doi.org/10.48550/arXiv.2307.12241
- **Link:** https://arxiv.org/abs/2307.12241
- **PDF OA:** https://arxiv.org/pdf/2307.12241
- **PDF local:** papers/08_076_explainable_depression_detection_via_hea.pdf
- **Dataset:** BlackDog and AVEC2013
- **Modalidad:** head motion (kineme reconstruction-error approach learned from healthy controls)
- **Relevancia:** Open-access full text of the ICMI 2023 kineme study, giving the thesis a reproducible XAI head-motion method (interpretable for rural health workers) beyond the paywalled ACM record.
- **Viabilidad (30fps/cara):** Feasible at 30fps; no high-fps/IR needed. Same underlying study as the ICMI 2023 entry - listed only for the open-access full text.


## 9. XAI sobre video/cara (explicabilidad) -- hilo central de la tesis

*Sub-tema xai-video-affect -- 9 papers verificados*

### 77. Explainable Depression Assessment from Face Videos by Weakly Supervised Learning  `[PDF DESCARGADO]`
- **Autores:** Rongfan Liao, Xiangyu Kong, Shiqing Tang, Lang He, Changzeng Fu, Weicheng Xie, Xiaofeng Liu, Lu Liu, Siyang Song
- **Anio / Venue:** 2026 -- Proceedings of the AAAI Conference on Artificial Intelligence (AAAI-26), Vol. 40
- **DOI:** https://doi.org/10.1609/aaai.v40i3.37173
- **Link:** https://ojs.aaai.org/index.php/AAAI/article/view/37173
- **PDF OA:** https://ojs.aaai.org/index.php/AAAI/article/download/37173/41135
- **PDF local:** papers/09_077_explainable_depression_assessment_from_f.pdf
- **Dataset:** AVEC 2013 (Freeform, Northwind subsets), AVEC 2014
- **Modalidad:** facial expression video (weakly supervised segment selection, 3D-CNN + Transformer)
- **Relevancia:** Directly the target task: explainable, weakly-supervised depression detection from frontal face video that prioritizes the most depression-related facial-behaviour segments, giving the student a state-of-the-art XAI-video blueprint to fuse with her audio pipeline.
- **Viabilidad (30fps/cara):** Feasible on 30fps frontal-face clips; method relies on segment-level facial behaviour rather than high-fps micro-expressions, matching this student's data.

### 78. Explainable Depression Detection Based on Facial Expression Using LSTM on Attentional Intermediate Feature Fusion with Label Smoothing  `[OA - solo link]`
- **Autores:** Yanisa Mahayossanunt, Natawut Nupairoj, Solaphat Hemrungrojn, Peerapon Vateekul
- **Anio / Venue:** 2023 -- Sensors (MDPI), 23(23):9402
- **DOI:** https://doi.org/10.3390/s23239402
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10708765/
- **PDF OA:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10708765/
- **Dataset:** Custom clinical interview video dataset (labeled with HAM-D), OpenFace 2 features (AU intensity, gaze radians, head pose)
- **Modalidad:** facial Action Units + gaze + head pose (LSTM with attention, Integrated Gradients XAI)
- **Relevancia:** Near-identical setup to this thesis: 30fps frontal interview video, OpenFace AU/gaze/head-pose features fed to an attention-LSTM with Integrated Gradients explanations flagging reduced smiling, head turning and gaze, giving a directly transferable feature set and XAI recipe for clinical interpretability.
- **Viabilidad (30fps/cara):** Fully feasible: explicitly uses 30fps video and OpenFace AU/gaze/head-pose features; no high-fps or IR eye-tracking required.

### 79. Predicting Depression, Anxiety, and Stress Levels from Videos Using the Facial Action Coding System  `[OA - solo link]`
- **Autores:** Mihai Gavrilescu, Nicolae Vizireanu
- **Anio / Venue:** 2019 -- Sensors (MDPI), 19(17):3693
- **DOI:** https://doi.org/10.3390/s19173693
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC6749518/
- **PDF OA:** https://pmc.ncbi.nlm.nih.gov/articles/PMC6749518/
- **Dataset:** Own video dataset annotated with DASS (Depression Anxiety Stress Scales); FACS Action Units
- **Modalidad:** facial Action Units (AAM + SVM AU classifier + feed-forward neural network)
- **Relevancia:** One of the few works covering BOTH anxiety and depression (plus stress) from facial AUs, mapping AU-intensity matrices to DASS levels (87.2% depression, 77.9% anxiety), directly supporting the student's GAD-7/PHQ-9 anxiety+depression video labels with inherently interpretable AU-level reasoning.
- **Viabilidad (30fps/cara):** Feasible: standard-frame-rate video and AAM/FACS AU extraction; no high-fps or IR eye-tracking needed.

### 80. Stress Recognition Identifying Relevant Facial Action Units Through Explainable Artificial Intelligence and Machine Learning  `[PDF DESCARGADO]`
- **Autores:** Giorgos Giannakakis, Anastasios Roussos, Christina Andreou, Stefan Borgwardt, Alexandra I. Korda
- **Anio / Venue:** 2024 -- Computer Methods and Programs in Biomedicine (Elsevier), Vol. 257, 108507
- **DOI:** https://doi.org/10.1016/j.cmpb.2024.108507
- **Link:** https://doi.org/10.1016/j.cmpb.2024.108507
- **PDF OA:** http://users.ics.forth.gr/ggian/publications/journals/2024%20Giannakakis%20Stress%20recognition%20identifying%20relevant%20facial%20action%20units%20through%20explainable%20artificial%20intelligence%20and%20machine%20learning.pdf
- **PDF local:** papers/09_080_stress_recognition_identifying_relevant_.pdf
- **Dataset:** Acute-stress facial video dataset (58 participants, 4 phases, 11 stress/non-stress tasks); FACS Action Units
- **Modalidad:** facial Action Units (ML + DL with XAI feature-ranking)
- **Relevancia:** Shows how XAI feature-ranking isolates the most discriminative facial AUs for an affective/clinical state, a methodology the student can mirror to rank AUs for anxiety/depression and explain them to non-specialist rural health workers.
- **Viabilidad (30fps/cara):** Feasible: AU-based from ordinary facial video; no high-fps micro-expression or IR gaze hardware required.

### 81. Towards Trustworthy AI: Evaluating SHAP and LIME for Facial Emotion Recognition  `[OA - solo link]`
- **Autores:** Selina Lorch, Jens Gebele, Philipp Brune
- **Anio / Venue:** 2025 -- Proceedings of the 58th Hawaii International Conference on System Sciences (HICSS-58)
- **DOI:** https://doi.org/10.24251/HICSS.2025.900
- **Link:** https://scholarspace.manoa.hawaii.edu/items/fd5ff3f1-da96-4ea1-8fdf-739e401602d5
- **PDF OA:** https://scholarspace.manoa.hawaii.edu/bitstreams/ceabca96-f1b5-4a37-83e9-de1a27d73c8b/download
- **Dataset:** FER2013, RAF-DB
- **Modalidad:** facial emotion recognition images (SHAP vs LIME comparison)
- **Relevancia:** Uses the EXACT XAI tools in this thesis (SHAP and LIME) on facial emotion models, concluding SHAP is more consistent and that highlighted regions align with FACS AUs but expert interpretation remains essential, a direct methodological and cautionary reference for the student's explainability thread.
- **Viabilidad (30fps/cara):** Feasible: image-level SHAP/LIME on standard face frames; no high-fps or IR requirement.

### 82. Guided Interpretable Facial Expression Recognition via Spatial Action Unit Cues  `[PDF DESCARGADO]`
- **Autores:** Soufiane Belharbi, Marco Pedersoli, Alessandro Lameiras Koerich, Simon Bacon, Eric Granger
- **Anio / Venue:** 2024 -- IEEE International Conference on Automatic Face and Gesture Recognition (FG 2024)
- **DOI:** https://doi.org/10.1109/FG59268.2024.10581986
- **Link:** https://arxiv.org/abs/2402.00281
- **PDF OA:** https://arxiv.org/pdf/2402.00281
- **PDF local:** papers/09_082_guided_interpretable_facial_expression_r.pdf
- **Dataset:** RAF-DB, AffectNet (with AU codebook + facial landmarks)
- **Modalidad:** facial expression recognition with spatial Action Unit heatmaps (interpretable CNN)
- **Relevancia:** Trains a classifier whose spatial attention is constrained to correlate with AU heatmaps, yielding visual explanations that mimic an expert's FACS reasoning, a strong design pattern for building clinically-interpretable face models for non-specialist reviewers.
- **Viabilidad (30fps/cara):** Feasible: uses landmarks + AU heatmaps on standard frames; no high-fps or IR eye-tracking needed.

### 83. Towards End-to-End Explainable Facial Action Unit Recognition via Vision-Language Joint Learning  `[PDF DESCARGADO]`
- **Autores:** Xuri Ge, Junchen Fu, Fuhai Chen, Shan An, Nicu Sebe, Joemon M. Jose
- **Anio / Venue:** 2024 -- Proceedings of the 32nd ACM International Conference on Multimedia (MM '24)
- **DOI:** https://doi.org/10.1145/3664647.3681443
- **Link:** https://arxiv.org/abs/2408.00644
- **PDF OA:** https://arxiv.org/pdf/2408.00644
- **PDF local:** papers/09_083_towards_end_to_end_explainable_facial_ac.pdf
- **Dataset:** BP4D, DISFA (facial AU benchmarks)
- **Modalidad:** facial Action Units + natural-language descriptions (vision-language joint learning, VL-FAU)
- **Relevancia:** Generates natural-language muscle-level descriptions alongside AU predictions, an explainability format well suited to communicating affect cues to non-specialist rural health workers in this thesis.
- **Viabilidad (30fps/cara):** Feasible: AU recognition from standard face images; however natural-language explanations would need Spanish adaptation for the target users.

### 84. Exploring Facial Biomarkers for Depression through Temporal Analysis of Action Units  `[PDF DESCARGADO]`
- **Autores:** Aditya Parikh, Misha Sadeghi, Robert Richer, Lydia Helene Rupp, Lena Schindler-Gmelch, Marie Keinert, Malin Hager, Klara Capito, Farnaz Rahimi, Bernhard Egger, Matthias Berking, Bjoern M. Eskofier
- **Anio / Venue:** 2024 -- arXiv preprint (cs.CV)
- **DOI:** https://doi.org/10.48550/arXiv.2407.13753
- **Link:** https://arxiv.org/abs/2407.13753
- **PDF OA:** https://arxiv.org/pdf/2407.13753
- **PDF local:** papers/09_084_exploring_facial_biomarkers_for_depressi.pdf
- **Dataset:** Clinical video dataset of depressed vs non-depressed participants (AU intensity time series)
- **Modalidad:** facial Action Unit time-series (mean-intensity comparison, PCA, clustering, time-series classification)
- **Relevancia:** Identifies which sadness/happiness-related AUs differ in depressed participants via interpretable temporal AU-intensity analysis, providing concrete candidate biomarkers and a transparent temporal-AU method the student can replicate on her 4.4-min clips.
- **Viabilidad (30fps/cara):** Feasible: temporal AU intensities from standard video; no high-fps micro-expression or IR gaze needed.

### 85. Human-Centered and Quantitative Explainability Evaluation of Facial Emotion Recognition for Trustworthy Mental Health Monitoring  `[OA - solo link]`
- **Autores:** Dina Shehada, Hissam Tawfik, Ahmed Bouridane, Abir Hussain
- **Anio / Venue:** 2026 -- Computers (MDPI), 15(3):139
- **DOI:** https://doi.org/10.3390/computers15030139
- **Link:** https://www.mdpi.com/2073-431X/15/3/139
- **PDF OA:** https://www.mdpi.com/2073-431X/15/3/139/pdf
- **Dataset:** RAF-DB, ExpW, FER2013 (facial emotion datasets, mental-health monitoring context)
- **Modalidad:** facial emotion recognition with XAI (SHAP-guided lightweight CNN; perturbation faithfulness + feature-localization metrics combined into a Global Explanation Quality Score, GEQS)
- **Relevancia:** Provides a framework for human-centered AND quantitative evaluation of facial-emotion XAI specifically for trustworthy mental-health monitoring, directly informing how the student should validate that her SHAP/Grad-CAM explanations are usable and trusted by rural health workers.
- **Viabilidad (30fps/cara):** Feasible: image/standard-video XAI evaluation; no high-fps or IR hardware needed.


## 10. Webcam / smartphone / 30 fps in-the-wild / rPPG

*Sub-tema webcam-inthewild-lowfps -- 10 papers verificados*

### 86. MoodCapture: Depression Detection Using In-the-Wild Smartphone Images  `[OA - solo link]`
- **Autores:** Subigya Nepal, Arvind Pillai, Weichen Wang, Tess Griffin, Amanda C. Collins, Michael Heinz, Damien Lekkas, Shayan Mirjafari, Matthew Nemesure, George Price, Nicholas C. Jacobson, Andrew T. Campbell
- **Anio / Venue:** 2024 -- Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems (CHI '24), ACM
- **DOI:** https://doi.org/10.1145/3613904.3642680
- **Link:** https://arxiv.org/abs/2402.16182
- **PDF OA:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11296678/
- **Dataset:** Custom in-the-wild dataset: ~125,000 front-camera images from N=177 participants (with MDD) over 90 days, labeled with PHQ-8
- **Modalidad:** Smartphone front-camera facial images (in-the-wild); facial landmarks / facial features (random-forest pipeline) plus deep CNN models
- **Relevancia:** Closest analog to this thesis: passive smartphone front-camera face capture with PHQ-based labels and facial features, showing depression can be screened from consumer phone imagery in naturalistic settings.
- **Viabilidad (30fps/cara):** Fully feasible: consumer smartphone front-camera images, no high-fps or IR hardware required; note it uses still photo bursts rather than continuous video, so temporal dynamics differ from the student's 4.4-min clips.

### 87. Reading Between the Frames: Multi-modal Depression Detection in Videos from Non-verbal Cues  `[PDF DESCARGADO]`
- **Autores:** David Gimeno-Gómez, Ana-Maria Bucur, Adrian Cosma, Carlos-David Martínez-Hinarejos, Paolo Rosso
- **Anio / Venue:** 2024 -- Advances in Information Retrieval (ECIR 2024), Springer LNCS vol. 14608
- **DOI:** https://doi.org/10.1007/978-3-031-56027-9_12
- **Link:** https://arxiv.org/abs/2401.02746
- **PDF OA:** https://arxiv.org/pdf/2401.02746
- **PDF local:** papers/10_087_reading_between_the_frames_multi_modal_d.pdf
- **Dataset:** D-Vlog, DAIC-WOZ, and E-DAIC (three video depression benchmarks)
- **Modalidad:** Audio-visual non-verbal cues: speech embeddings, facial emotion embeddings, face/body/hand landmarks, gaze and blinking
- **Relevancia:** Directly models the exact feasible-from-30fps cue set this thesis can extract (face/body landmarks, gaze, blink) in a flexible multimodal temporal model, providing a strong architectural template for the audio+video pivot.
- **Viabilidad (30fps/cara):** Feasible: relies on landmarks, coarse gaze, and blink derived from standard video; no high-fps or IR eye-tracking needed.

### 88. Harnessing multimodal approaches for depression detection using large language models and facial expressions  `[OA - solo link]`
- **Autores:** Misha Sadeghi, Robert Richer, Bernhard Egger, Lena Schindler-Gmelch, Lydia Helene Rupp, Farnaz Rahimi, Matthias Berking, Bjoern M. Eskofier
- **Anio / Venue:** 2024 -- npj Mental Health Research (Nature Portfolio), vol. 3, art. 66
- **DOI:** https://doi.org/10.1038/s44184-024-00112-8
- **Link:** https://www.nature.com/articles/s44184-024-00112-8
- **PDF OA:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11666580/
- **Dataset:** E-DAIC (Extended Distress Analysis Interview Corpus)
- **Modalidad:** Multimodal: LLM over interview transcripts + facial features from video frames; depression severity regression on PHQ-8
- **Relevancia:** Aligned with the thesis's XAI thread: fuses text/audio with facial features and quantifies modality contributions, an interpretable multimodal recipe; honest caveat that the visual modality contributed minimally relative to text.
- **Viabilidad (30fps/cara):** Feasible: OpenFace-style AUs/head pose/gaze run on standard video; honest caveat for the student is that here the visual modality contributed minimally relative to text.

### 89. Explainable Depression Detection via Head Motion Patterns  `[PDF DESCARGADO]`
- **Autores:** Monika Gahalawat, Raul Fernandez Rojas, Tanaya Guha, Ramanathan Subramanian, Roland Goecke
- **Anio / Venue:** 2023 -- Proceedings of the 25th ACM International Conference on Multimodal Interaction (ICMI 2023)
- **DOI:** https://doi.org/10.1145/3577190.3614130
- **Link:** https://arxiv.org/abs/2307.12241
- **PDF OA:** https://arxiv.org/pdf/2307.12241
- **PDF local:** papers/10_089_explainable_depression_detection_via_hea.pdf
- **Dataset:** BlackDog and AVEC2013 depression corpora (clinical interview video)
- **Modalidad:** Head motion / head-pose dynamics ('kinemes') over yaw, pitch, roll; explainable ML
- **Relevancia:** Demonstrates head pose/motion alone as an explainable depression biomarker, validating one of the most robust features the student CAN extract from 30fps vertical-face video.
- **Viabilidad (30fps/cara):** Feasible: head pose from standard video is reliable; no high-fps/IR requirement.

### 90. Explainable Depression Detection Based on Facial Expression Using LSTM on Attentional Intermediate Feature Fusion with Label Smoothing  `[OA - solo link]`
- **Autores:** Yanisa Mahayossanunt, Natawut Nupairoj, Solaphat Hemrungrojn, Peerapon Vateekul
- **Anio / Venue:** 2023 -- Sensors (MDPI), vol. 23, no. 23, art. 9402
- **DOI:** https://doi.org/10.3390/s23239402
- **Link:** https://www.mdpi.com/1424-8220/23/23/9402
- **PDF OA:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10708765/
- **Dataset:** Custom Chulalongkorn University interview-video dataset (474 samples: 134 depressed, 340 non-depressed); features include action unit intensity, gaze angles, head pose
- **Modalidad:** Facial features (action unit intensity, gaze angles, head pose) with attentional LSTM and integrated-gradient explanations
- **Relevancia:** Combines exactly the 30fps-feasible feature set (AUs, coarse gaze, head pose) with an explainable LSTM, mirroring the thesis goal of interpretable video-based depression screening.
- **Viabilidad (30fps/cara):** Feasible: all input features are derivable from standard-rate interview video; uses integrated gradients (a SHAP-adjacent XAI method).

### 91. FacialPulse: An Efficient RNN-based Depression Detection via Temporal Facial Landmarks  `[PDF DESCARGADO]`
- **Autores:** Ruiqi Wang, Jinyang Huang, Jie Zhang, Xin Liu, Xiang Zhang, Zhi Liu, Peng Zhao, Sigui Chen, Xiao Sun
- **Anio / Venue:** 2024 -- Proceedings of the 32nd ACM International Conference on Multimedia (MM '24)
- **DOI:** https://doi.org/10.1145/3664647.3681546
- **Link:** https://arxiv.org/abs/2408.03499
- **PDF OA:** https://arxiv.org/pdf/2408.03499
- **PDF local:** papers/10_091_facialpulse_an_efficient_rnn_based_depre.pdf
- **Dataset:** AVEC2014 and MMDA (Multimodal Dataset for Depression and Anxiety)
- **Modalidad:** 68-point temporal facial landmarks modeled by bidirectional RNN/GRU networks (with a Facial Landmark Calibration Module)
- **Relevancia:** A lightweight landmark-only temporal model that captures expression dynamics from ordinary video frames, matching the thesis's feasible facial-dynamics features and resource-constrained deployment.
- **Viabilidad (30fps/cara):** Feasible: operates on standard-frame-rate 68-point landmarks, no micro-expression high-fps or IR tracking; spotlights expression dynamics rather than true micro-expressions.

### 92. Contactless Depression Screening via Facial Video-derived Heart Rate Variability  `[PDF DESCARGADO]`
- **Autores:** Translational Psychiatry author group (full list on Nature/medRxiv listing)
- **Anio / Venue:** 2026 -- Translational Psychiatry (Nature Portfolio); preprint on medRxiv (2025)
- **DOI:** https://doi.org/10.1038/s41398-026-03831-y
- **Link:** https://www.nature.com/articles/s41398-026-03831-y
- **PDF OA:** https://www.medrxiv.org/content/10.1101/2025.05.01.25326621v1.full.pdf
- **PDF local:** papers/10_092_contactless_depression_screening_via_fac.pdf
- **Dataset:** 1,453 individuals with facial video recordings and PHQ-9 depression labels
- **Modalidad:** Remote photoplethysmography (rPPG): facial-video-derived heart rate variability (HRV) features fed to a stacking ensemble ML model
- **Relevancia:** Demonstrates the rPPG-from-face-video-for-depression pathway on a large sample (best AUROC ~0.64), offering a physiological video modality the thesis could add beyond AUs/pose.
- **Viabilidad (30fps/cara):** Partially feasible: rPPG works from RGB face video, but robust HRV benefits from higher/stable fps and good lighting; 30fps handheld phone video with movement may degrade signal-to-noise, and reported discrimination was only moderate (AUROC ~0.64).

### 93. Enhancing Stress Detection: A Comprehensive Approach through rPPG Analysis and Deep Learning Techniques  `[OA - solo link]`
- **Autores:** Laura Fontes, Pedro Machado, Doratha Vinkemeier, Salisu Yahaya, Jordan J. Bird, Isibor Kennedy Ihianle
- **Anio / Venue:** 2024 -- Sensors (MDPI), vol. 24, no. 4, art. 1096
- **DOI:** https://doi.org/10.3390/s24041096
- **Link:** https://www.mdpi.com/1424-8220/24/4/1096
- **PDF OA:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10892284/
- **Dataset:** UBFC-Phys (stress tasks: rest/speech/arithmetic)
- **Modalidad:** rPPG signals extracted from facial video classified with LSTM, GRU, and 1D-CNN hybrid deep-learning models
- **Relevancia:** Shows an end-to-end pipeline turning face-video rPPG into stress/affect classification (up to ~95.8% accuracy), a concrete recipe for adding a contactless physiological signal to the thesis's anxiety detection.
- **Viabilidad (30fps/cara):** Partially feasible: rPPG extraction is camera-based, but accuracy depends on stable illumination/low motion; vertical handheld 30fps phone clips may not reach the controlled-lab performance reported here.

### 94. OpenFace 2.0: Facial Behavior Analysis Toolkit  `[PDF DESCARGADO]`
- **Autores:** Tadas Baltrušaitis, Amir Zadeh, Yao Chong Lim, Louis-Philippe Morency
- **Anio / Venue:** 2018 -- 13th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2018), pp. 59-66
- **DOI:** https://doi.org/10.1109/FG.2018.00019
- **Link:** https://ieeexplore.ieee.org/document/8373812/
- **PDF OA:** https://par.nsf.gov/servlets/purl/10099458
- **PDF local:** papers/10_094_openface_2_0_facial_behavior_analysis_to.pdf
- **Dataset:** N/A (toolkit; validated on standard AU/landmark benchmarks)
- **Modalidad:** Facial landmark detection, head-pose estimation, facial action unit recognition, and eye-gaze estimation from a plain webcam
- **Relevancia:** Foundational, real-time, webcam-grade toolkit that produces precisely the interpretable features (AUs, head pose, gaze, blink) this thesis will extract from 30fps phone video and feed to SHAP/LIME.
- **Viabilidad (30fps/cara):** Feasible: explicitly designed to run from a simple webcam without specialist hardware; AUs/pose/coarse gaze are exactly the feasible features, while its gaze output is coarse (not microsaccade-grade).

### 95. Context Matters: Vision-Based Depression Detection Comparing Classical and Deep Approaches  `[PDF DESCARGADO]`
- **Autores:** Maneesh Bilalpur, Saurabh Hinduja, Sonish Sivarajkumar, Nicholas Allen, Yanshan Wang, Itir Onal Ertugrul, Jeffrey F. Cohn
- **Anio / Venue:** 2026 -- arXiv preprint (arXiv:2604.10344)
- **DOI:** https://doi.org/10.48550/arXiv.2604.10344
- **Link:** https://arxiv.org/abs/2604.10344
- **PDF OA:** https://arxiv.org/pdf/2604.10344
- **PDF local:** papers/10_095_context_matters_vision_based_depression_.pdf
- **Dataset:** TPOT (mother-child interactions) and Pitt (patient-clinician interviews) video databases
- **Modalidad:** Vision-based facial features: interpretable facial-expression features + SVM (classical) vs. learned VGGNet features (deep)
- **Relevancia:** Directly addresses the thesis's core trade-off, interpretable classical facial features (XAI-friendly) versus deep features, evaluating accuracy and cross-context generalizability for depression detection.
- **Viabilidad (30fps/cara):** Feasible: uses standard-video facial-expression features; the classical/interpretable branch aligns with the thesis XAI requirement, and its cross-context generalizability analysis is relevant to the small rural Colombian sample.


## 11. Datasets y estudios en espanol o Latinoamerica

*Sub-tema spanish-latam-multimodal -- 8 papers verificados*

### 96. Data Collection for Automatic Depression Identification in Spanish Speakers Using Deep Learning Algorithms: Protocol for a Case-Control Study  `[OA - solo link]`
- **Autores:** Luis F. Brenes, Luis A. Trejo, Jose Antonio Cantoral-Ceballos, Daniela Aguilar-De León, Fresia Paloma Hernández-Moreno
- **Anio / Venue:** 2025 -- JMIR Research Protocols (vol. 14, e60439)
- **DOI:** https://doi.org/10.2196/60439
- **Link:** https://www.researchprotocols.org/2025/1/e60439
- **PDF OA:** https://pmc.ncbi.nlm.nih.gov/articles/PMC12355134/
- **Dataset:** D3TEC (TEC de Monterrey Depression Detection Dataset): ~62 Spanish-speaking subjects, dual professional (Shure) + smartphone (iPhone) capture, PHQ-9 labels
- **Modalidad:** Audio only (voice recordings)
- **Relevancia:** Closest direct analogue to this thesis: a Spanish-language, PHQ-9-labeled depression voice dataset built for deep-learning classification, providing a Latin-American acoustic benchmark and a smartphone-vs-professional-mic comparison relevant to the student's phone-recorded cohort.
- **Viabilidad (30fps/cara):** Audio-only (no video/facial data); usable as an acoustic-pipeline benchmark but does not supply the visual modality the student is adding.

### 97. Spanish MEACorpus 2023: A multimodal speech–text corpus for emotion analysis in Spanish from natural environments  `[DE PAGO]`
- **Autores:** Ronghao Pan, José Antonio García-Díaz, Miguel Ángel Rodríguez-García, Rafael Valencia-García
- **Anio / Venue:** 2024 -- Computer Standards & Interfaces (Elsevier)
- **DOI:** https://doi.org/10.1016/j.csi.2024.103856
- **Link:** https://doi.org/10.1016/j.csi.2024.103856
- **Dataset:** Spanish MEACorpus 2023: 13.16 h of speech in 5,129 segments from YouTube videos in natural environments, labeled with Ekman's six basic emotions
- **Modalidad:** Multimodal speech + text (sourced from YouTube video; Ekman six-emotion labels)
- **Relevancia:** One of the only multimodal Spanish emotion corpora drawn from real video, with late-fusion baselines (Macro-F1 ~87.7%) that inform how to fuse acoustic and linguistic streams for a Spanish-language affect model before adding the visual channel.
- **Viabilidad (30fps/cara):** Annotated modalities are speech+text only; no facial/AU features are released despite the video source.

### 98. MuSE: a Multimodal Dataset of Stressed Emotion  `[PDF DESCARGADO]`
- **Autores:** Mimansa Jaiswal, Cristian-Paul Bara, Yuanhang Luo, Mihai Burzo, Rada Mihalcea, Emily Mower Provost
- **Anio / Venue:** 2020 -- LREC 2020 (12th Language Resources and Evaluation Conference), pp. 1499–1510, Marseille
- **Link:** https://aclanthology.org/2020.lrec-1.187/
- **PDF OA:** https://aclanthology.org/2020.lrec-1.187.pdf
- **PDF local:** papers/11_098_muse_a_multimodal_dataset_of_stressed_em.pdf
- **Dataset:** MuSE: video clips annotated for stress and emotion (valence/arousal) with audio, video, text plus physiological signals
- **Modalidad:** Audio-visual + text + physiological (facial video, valence/arousal, ECG/RESP/BPM)
- **Relevancia:** A foundational multimodal stress/affect dataset whose audio-video-text fusion protocol and emotion-under-stress framing is a strong methodological template for combining face video with acoustic biomarkers for anxiety/depression states.
- **Viabilidad (30fps/cara):** Includes physiological (ECG/RESP) sensors the student does not have; the audio-video-text portion is feasible, but the physiological channel and any micro-expression-rate assumptions are not reproducible with 30 fps phone video.

### 99. The Mexican Emotional Speech Database (MESD): elaboration and assessment based on machine learning  `[OA - solo link]`
- **Autores:** Mathilde Marie Duville, Luz María Alonso-Valerdi, David I. Ibarra-Zarate
- **Anio / Venue:** 2021 -- IEEE EMBC 2021 (43rd Annual Intl Conf. of the IEEE EMBS); companion dataset paper in Data (MDPI)
- **DOI:** https://doi.org/10.1109/EMBC46164.2021.9629934
- **Link:** https://pubmed.ncbi.nlm.nih.gov/34891601/
- **PDF OA:** https://www.mdpi.com/2306-5729/6/12/130
- **Dataset:** MESD: 864 single-word emotional utterances (anger, disgust, fear, happiness, neutral, sadness) in Mexican Spanish; adult male/female and child voices; on Mendeley Data
- **Modalidad:** Audio only (Mexican-Spanish emotional speech)
- **Relevancia:** A culturally-shaped Latin-American (Mexican Spanish) emotional-prosody resource useful for pretraining or validating the acoustic-emotion side of the model on regional Spanish prosody before fusing with video.
- **Viabilidad (30fps/cara):** Audio-only and acted single words; no video/facial modality and limited ecological validity for spontaneous rural-youth speech.

### 100. EmoMatchSpanishDB: study of speech emotion recognition machine learning models in a new Spanish elicited database  `[OA - solo link]`
- **Autores:** Esteban García-Cuesta, Antonio Barba Salvador, Diego Gachet Páez
- **Anio / Venue:** 2024 -- Multimedia Tools and Applications (Springer), vol. 83
- **DOI:** https://doi.org/10.1007/s11042-023-15959-w
- **Link:** https://doi.org/10.1007/s11042-023-15959-w
- **PDF OA:** https://oa.upm.es/80921/
- **Dataset:** EmoMatchSpanishDB: 2,005 Spanish speech signals from 50 non-actors over Ekman's six emotions + neutral; crowdsourcing-validated subset of EmoSpanishDB
- **Modalidad:** Audio only (elicited Spanish speech emotion)
- **Relevancia:** A recent crowdsourcing-validated Spanish emotional-speech benchmark with published ML baselines, useful as a comparison corpus for the acoustic-emotion component and for cross-corpus robustness checks of Spanish SER models.
- **Viabilidad (30fps/cara):** Audio-only; no facial or video data, so it supports only the acoustic branch of the planned multimodal system.

### 101. Overview of MentalRiskES at IberLEF 2023: Early Detection of Mental Disorders Risk in Spanish  `[PDF DESCARGADO]`
- **Autores:** Alba M. Mármol-Romero, Adrián Moreno-Muñoz, Flor Miriam Plaza-del-Arco, M. Dolores Molina-González, Arturo Montejo-Ráez
- **Anio / Venue:** 2023 -- Procesamiento del Lenguaje Natural, no. 71 / IberLEF 2023 (SEPLN)
- **Link:** http://journal.sepln.org/sepln/ojs/ojs/index.php/pln/article/view/6564
- **PDF OA:** http://journal.sepln.org/sepln/ojs/ojs/index.php/pln/article/download/6564/3964
- **PDF local:** papers/11_101_overview_of_mentalriskes_at_iberlef_2023.pdf
- **Dataset:** MentalRiskES: Spanish Telegram message threads annotated for eating disorder, depression and anxiety risk (GitHub: sinai-uja/corpusMentalRiskES)
- **Modalidad:** Text only (Spanish Telegram social-media messages)
- **Relevancia:** The leading shared-task corpus for early detection of depression and anxiety risk specifically in Spanish, establishing the Spanish-language clinical-risk evaluation landscape and explainability expectations the thesis sits within.
- **Viabilidad (30fps/cara):** Text-only (audio/images/video excluded); informs task framing and Spanish labels rather than the multimodal signal pipeline.

### 102. LMVD: A large-scale multimodal vlog dataset for depression detection in the wild  `[PDF DESCARGADO]`
- **Autores:** Lang He, Kai Chen, Junnan Zhao, Yimeng Wang, Ercheng Pei, Haifeng Chen, Jiewei Jiang, Shiqing Zhang, Jie Zhang, Zhongmin Wang, Tao He, Prayag Tiwari
- **Anio / Venue:** 2025 -- Information Fusion (Elsevier), vol. 126
- **DOI:** https://doi.org/10.1016/j.inffus.2025.103632
- **Link:** https://arxiv.org/abs/2407.00024
- **PDF OA:** https://arxiv.org/pdf/2407.00024
- **PDF local:** papers/11_102_lmvd_a_large_scale_multimodal_vlog_datas.pdf
- **Dataset:** LMVD: 1,823 video samples (~214 h) from 1,475 participants collected from Sina Weibo, Bilibili, TikTok and YouTube vlogs
- **Modalidad:** Audio-visual (in-the-wild face vlog video for depression)
- **Relevancia:** A large in-the-wild face-vlog depression dataset whose naturalistic single-camera talking-head setup closely mirrors the student's ~252 frontal-face phone clips, providing methods and baselines for depression detection from consumer-grade video.
- **Viabilidad (30fps/cara):** Not Spanish/Latin-American; included as a feasibility-matched in-the-wild video benchmark. Standard-fps video, feasible with the student's 30 fps frontal-face clips (no high-fps/IR requirement).

### 103. Facial action units guided graph representation learning for multimodal depression detection  `[DE PAGO]`
- **Autores:** Changzeng Fu, Fengkui Qian, Yikai Su, Kaifeng Su, Siyang Song, Mingyue Niu, Jiaqi Shi, Zhigang Liu, Chaoran Liu, Carlos Toshinori Ishi, Hiroshi Ishiguro
- **Anio / Venue:** 2025 -- Neurocomputing (Elsevier), vol. 619, art. 129106
- **DOI:** https://doi.org/10.1016/j.neucom.2024.129106
- **Link:** https://doi.org/10.1016/j.neucom.2024.129106
- **Dataset:** Evaluated on standard clinical multimodal depression corpora (DAIC-WOZ / E-DAIC family using facial action units, head motion, landmarks)
- **Modalidad:** Audio-visual; facial Action Units (AUs), head motion, landmarks fused via graph learning
- **Relevancia:** Directly demonstrates AU-guided graph representation learning for depression, validating that facial Action Units and head-motion features (exactly the cues feasible from the student's 30 fps frontal video) can drive multimodal depression detection.
- **Viabilidad (30fps/cara):** AU/head-pose/landmark features are extractable from 30 fps 1080p frontal video, so feasible; does not require micro-expression-rate (100-200 fps) or IR eye-tracking, matching the student's data constraints.


## 12. Surveys y revisiones sistematicas (2021-2026)

*Sub-tema surveys-reviews -- 9 papers verificados*

### 104. Deep learning for depression recognition with audiovisual cues: A review  `[PDF DESCARGADO]`
- **Autores:** Lang He, Mingyue Niu, Prayag Tiwari, Pekka Marttinen, Rui Su, Jiewei Jiang, Chenguang Guo, Hongyu Wang, Songtao Ding, Zhongmin Wang, Xiaoying Pan, Wei Dang
- **Anio / Venue:** 2022 -- Information Fusion (Elsevier), Vol. 80, pp. 56-86
- **DOI:** https://doi.org/10.1016/j.inffus.2021.10.012
- **Link:** https://doi.org/10.1016/j.inffus.2021.10.012
- **PDF OA:** https://arxiv.org/pdf/2106.00610
- **PDF local:** papers/12_104_deep_learning_for_depression_recognition.pdf
- **Dataset:** Reviews benchmark datasets including AVEC 2013/2014, DAIC-WOZ, and other audio-visual depression corpora
- **Modalidad:** audio-visual (facial cues + acoustic/speech)
- **Relevancia:** Foundational, highly-cited review of deep audio-visual depression recognition that frames exactly the student's audio-to-multimodal pivot, cataloguing facial and acoustic biomarkers and the fusion strategies to combine them.

### 105. Deep learning-based depression recognition through facial expression: A systematic review  `[DE PAGO]`
- **Autores:** Xiaoming Cao, Lingling Zhai, Pengpeng Zhai, Fangfei Li, Tao He, Lang He
- **Anio / Venue:** 2025 -- Neurocomputing (Elsevier), Vol. 627, Art. 129605
- **DOI:** https://doi.org/10.1016/j.neucom.2025.129605
- **Link:** https://doi.org/10.1016/j.neucom.2025.129605
- **Dataset:** Reviews AVEC2013, AVEC2014, D-vlog and related facial depression datasets
- **Modalidad:** facial expression (video); spatial and spatial-temporal features
- **Relevancia:** The most directly on-point recent systematic review (2017-2024) of deep facial-expression depression recognition, mapping the exact video-based feature families (spatial vs spatial-temporal) the student can extract from 30fps frontal-face clips.
- **Viabilidad (30fps/cara):** Reviewed methods use ordinary 25-30fps video and standard facial features that match the student's data; no high-fps requirement.

### 106. Automatic Depression Assessment using Machine Learning: A Comprehensive Survey  `[PDF DESCARGADO]`
- **Autores:** Siyang Song, Yupeng Huo, Shiqing Tang, Jiaee Cheong, Rui Gao, Michel Valstar, Hatice Gunes
- **Anio / Venue:** 2025 -- arXiv preprint (cs.CV / q-bio.NC), arXiv:2506.18915
- **DOI:** https://doi.org/10.48550/arXiv.2506.18915
- **Link:** https://arxiv.org/abs/2506.18915
- **PDF OA:** https://arxiv.org/pdf/2506.18915
- **PDF local:** papers/12_106_automatic_depression_assessment_using_ma.pdf
- **Dataset:** Reviews depression assessment datasets across modalities (AVEC series, DAIC-WOZ, body/gait corpora)
- **Modalidad:** multimodal: brain, verbal language, audio, facial, and body/gait behaviors
- **Relevancia:** The newest broad survey of ML-based automatic depression assessment that explicitly organizes non-verbal audio/facial/body cues and fusion, giving the student an up-to-date map for designing the audio+video multimodal pipeline.
- **Viabilidad (30fps/cara):** Covers body-movement/gait methods that require full-body framing; the student's tight frontal-face vertical clips suit only the facial/head-motion subset, not gait analysis.

### 107. Machine Learning for Multimodal Mental Health Detection: A Systematic Review of Passive Sensing Approaches  `[OA - solo link]`
- **Autores:** Lin Sze Khoo, Mei Kuan Lim, Chun Yong Chong, Roisin McNaney
- **Anio / Venue:** 2024 -- Sensors (MDPI), Vol. 24, Issue 2, Art. 348
- **DOI:** https://doi.org/10.3390/s24020348
- **Link:** https://doi.org/10.3390/s24020348
- **PDF OA:** https://www.mdpi.com/1424-8220/24/2/348/pdf
- **Dataset:** Synthesizes 184 studies across passive-sensing corpora (audio/video incl. AVEC-style depression data, social media, smartphone, wearables)
- **Modalidad:** multimodal passive sensing: audio, video/facial, text, smartphone, wearables
- **Relevancia:** A rigorous 184-study systematic review of feature extraction, fusion, and ML for mental-health detection from passively sensed audio/video, giving the student a structured methodological checklist for combining acoustic and facial-behavioral features.

### 108. AI-assisted multi-modal information for the screening of depression: a systematic review and meta-analysis  `[PDF DESCARGADO]`
- **Autores:** Luyao Wang, Chenhan Wang, Chenyang Li, Toshiya Murai, Yicai Bai, et al.
- **Anio / Venue:** 2025 -- npj Digital Medicine (Nature)
- **DOI:** https://doi.org/10.1038/s41746-025-01933-3
- **Link:** https://doi.org/10.1038/s41746-025-01933-3
- **PDF OA:** https://www.nature.com/articles/s41746-025-01933-3.pdf
- **PDF local:** papers/12_108_ai_assisted_multi_modal_information_for_.pdf
- **Dataset:** Meta-analysis pooling studies using EEG, eye-movement, video, audio, and gait modalities
- **Modalidad:** multimodal physiological/behavioral (EEG, eye-movement, video, audio, gait)
- **Relevancia:** Quantitative meta-analytic evidence that multimodal screening (pooled AUC 0.95) outperforms uni-modal (AUC 0.84-0.92), giving the student a citable justification for adding video to the previously weak audio-only model.
- **Viabilidad (30fps/cara):** Pooled estimates include EEG and IR-grade eye-movement studies the student cannot replicate; only the video/audio subset of the evidence transfers to this phone-video setup.

### 109. AI-based recognition of facial and micro-expressions for the diagnosis of mental and neurological disorders: a systematic review  `[PDF DESCARGADO]`
- **Autores:** Sara Ghafarfaraji
- **Anio / Venue:** 2025 -- BMC Psychiatry (Springer Nature), Vol. 26, Art. 78
- **DOI:** https://doi.org/10.1186/s12888-025-07739-7
- **Link:** https://doi.org/10.1186/s12888-025-07739-7
- **PDF OA:** https://bmcpsychiatry.biomedcentral.com/counter/pdf/10.1186/s12888-025-07739-7
- **PDF local:** papers/12_109_ai_based_recognition_of_facial_and_micro.pdf
- **Dataset:** Synthesizes 36 studies (from 1710 screened) on facial/micro-expression AI for autism, depression, anxiety and other disorders
- **Modalidad:** facial expressions and micro-expressions (video), AI/ML methods
- **Relevancia:** A current (2021-2025) systematic review covering both depression and anxiety via facial and micro-expression AI, directly relevant since the student targets both PHQ-9/GAD-7 constructs from facial video.
- **Viabilidad (30fps/cara):** Some included work relies on true micro-expression spotting (needs 100-200fps); the student's 30fps clips support macro facial Action Units and expression dynamics but NOT genuine micro-expression analysis.

### 110. Systematic Review and Meta-Analysis of Explainable Machine Learning Models for Clinical Depression Detection  `[OA - solo link]`
- **Autores:** Ariosto Trelles, Tomas Fontaines Ruiz, Antonio Ponce Rojo
- **Anio / Venue:** 2025 -- Behavioral Sciences (MDPI), Vol. 15, Issue 11, Art. 1476
- **DOI:** https://doi.org/10.3390/bs15111476
- **Link:** https://doi.org/10.3390/bs15111476
- **PDF OA:** https://www.mdpi.com/2076-328X/15/11/1476/pdf
- **Dataset:** Synthesizes 20 studies (2014-2025) using EHR, clinical surveys, EEG, speech/interviews, HRV
- **Modalidad:** multimodal clinical data with explainable ML (SHAP dominant in 70% of studies; SHAP+LIME)
- **Relevancia:** Directly preserves the thesis's XAI thread: a PRISMA review/meta-analysis showing SHAP (and SHAP+LIME) are the predominant interpretability tools in depression ML and that combining them improves F1, validating the student's clinical-interpretability approach for non-specialist health workers.

### 111. Machine learning approaches to anxiety detection: trends, model evaluation, and future directions  `[PDF DESCARGADO]`
- **Autores:** Meruyert Taskynbayeva, Alina Gutoreva
- **Anio / Venue:** 2025 -- Frontiers in Artificial Intelligence, Vol. 8, Art. 1630047
- **DOI:** https://doi.org/10.3389/frai.2025.1630047
- **Link:** https://doi.org/10.3389/frai.2025.1630047
- **PDF OA:** https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1630047/pdf
- **PDF local:** papers/12_111_machine_learning_approaches_to_anxiety_d.pdf
- **Dataset:** Systematic review of 19 studies across heterogeneous anxiety datasets/biosignals (studies 2018-2025)
- **Modalidad:** multimodal ML for anxiety (biosignals, behavioral, facial/video among reviewed studies)
- **Relevancia:** One of the few recent reviews focused on the anxiety (GAD-7) side of the thesis, summarizing model trends and stressing the need for explainable ML plus external validation - a caution directly relevant to the student's small 252-participant rural cohort.

### 112. A systematic review on automated clinical depression diagnosis  `[PDF DESCARGADO]`
- **Autores:** Kaining Mao, Yuqi Wu, Jie Chen, et al.
- **Anio / Venue:** 2023 -- npj Mental Health Research (Nature), Vol. 2, Art. 20
- **DOI:** https://doi.org/10.1038/s44184-023-00040-z
- **Link:** https://doi.org/10.1038/s44184-023-00040-z
- **PDF OA:** https://www.nature.com/articles/s44184-023-00040-z.pdf
- **PDF local:** papers/12_112_a_systematic_review_on_automated_clinica.pdf
- **Dataset:** Reviews datasets across audio, visual, text and physiological modalities (incl. DAIC-WOZ, AVEC)
- **Modalidad:** multimodal (audio, facial/video, text, physiological)
- **Relevancia:** A clinically framed systematic review of automated depression diagnosis that organizes modality-specific biomarkers and evaluation pitfalls, helping the student situate acoustic+facial fusion within validated clinical-screening workflows.


---

*Total: 112 papers verificados | 98 con acceso abierto | 64 PDFs descargados en papers/.*
