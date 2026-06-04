# -*- coding: utf-8 -*-
import json, re, os

SRC = r'C:\Users\USUARIO\AppData\Local\Temp\claude\c--Users-USUARIO-Documents-Maestria-Multimodal\e63508a5-8c14-4c10-bb12-327f93b93495\tasks\whi8nkgza.output'
d = json.load(open(SRC, 'r', encoding='utf-8'))
topics = d['result']['topics']

TOPIC_META = [
    ('multimodal-av-depression',   'Fusion multimodal AUDIO+VIDEO para depresion (el nucleo del pivote)'),
    ('facial-au-depression',       'Action Units faciales y dinamica facial en depresion'),
    ('facial-anxiety',             'Comportamiento facial en ANSIEDAD y ansiedad social'),
    ('microexpression-methods',    'Microexpresiones: metodos y datasets (CASME / SAMM / SMIC...)'),
    ('micro-gesture-emotion',      'Micro-GESTOS corporales para emocion/estres (iMiGUE / SMG / MiGA)'),
    ('gaze-eye-contact-depression','Mirada / contacto visual / gaze aversion en depresion-ansiedad'),
    ('blink-pupil-biomarkers',     'Parpadeo / pupilometria / marcadores oculares'),
    ('head-pose-motion-depression','Pose y movimiento de cabeza (retardo psicomotor)'),
    ('xai-video-affect',           'XAI sobre video/cara (explicabilidad) -- hilo central de la tesis'),
    ('webcam-inthewild-lowfps',    'Webcam / smartphone / 30 fps in-the-wild / rPPG'),
    ('spanish-latam-multimodal',   'Datasets y estudios en espanol o Latinoamerica'),
    ('surveys-reviews',            'Surveys y revisiones sistematicas (2021-2026)'),
]

def slug(s, n=40):
    s = re.sub(r'[^a-zA-Z0-9]+', '_', s).strip('_').lower()
    return s[:n]

INTRO = r"""# Dossier de literatura: analisis de VIDEO para deteccion de ansiedad/depresion

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
"""

lines = []
paper_n = 0
verified_total = 0
oa_count = 0
downloaded = 0
idx_lines = []
for ti, (key, title_es) in enumerate(TOPIC_META):
    n = len([p for p in topics[ti].get('papers', []) if p.get('verified')])
    idx_lines.append(f"{ti+1}. {title_es} -- {n} papers")

for ti, topic in enumerate(topics):
    key, title_es = TOPIC_META[ti]
    papers = [p for p in topic.get('papers', []) if p.get('verified')]
    if not papers:
        continue
    lines.append(f"\n## {ti+1}. {title_es}\n")
    lines.append(f"*Sub-tema {key} -- {len(papers)} papers verificados*\n")
    for p in papers:
        verified_total += 1
        paper_n += 1
        t = p.get('title', '').strip()
        au = p.get('authors', '').strip()
        yr = p.get('year', '').strip()
        ven = p.get('venue', '').strip()
        doi = (p.get('doi') or '').strip()
        url = (p.get('url') or '').strip()
        oa = (p.get('oa_pdf_url') or '').strip()
        best = (p.get('best_link') or url).strip()
        ds = (p.get('dataset') or '').strip()
        mod = (p.get('modality') or '').strip()
        rel = (p.get('relevance') or '').strip()
        feas = (p.get('feasibility_flag') or '').strip()
        oa_av = p.get('oa_available')
        status = ''
        local = ''
        if oa and oa_av:
            oa_count += 1
            fname = f"{ti+1:02d}_{paper_n:03d}_{slug(t)}.pdf"
            fpath = os.path.join('papers', fname)
            if os.path.exists(fpath) and os.path.getsize(fpath) > 20000:
                downloaded += 1
                status = '  `[PDF DESCARGADO]`'
                local = f"- **PDF local:** papers/{fname}"
            else:
                status = '  `[OA - solo link]`'
        elif oa_av:
            status = '  `[OA - ver link]`'
        else:
            status = '  `[DE PAGO]`'
        lines.append(f"### {paper_n}. {t}{status}")
        lines.append(f"- **Autores:** {au}")
        lines.append(f"- **Anio / Venue:** {yr} -- {ven}")
        if doi:
            dd = doi.replace('https://doi.org/', '')
            lines.append(f"- **DOI:** https://doi.org/{dd}")
        if best:
            lines.append(f"- **Link:** {best}")
        if oa and oa_av:
            lines.append(f"- **PDF OA:** {oa}")
        if local:
            lines.append(local)
        if ds:
            lines.append(f"- **Dataset:** {ds}")
        if mod:
            lines.append(f"- **Modalidad:** {mod}")
        if rel:
            lines.append(f"- **Relevancia:** {rel}")
        if feas and feas.lower() not in ('none', 'n/a', ''):
            lines.append(f"- **Viabilidad (30fps/cara):** {feas}")
        lines.append("")

out = INTRO + "\n".join(idx_lines) + "\n" + "\n".join(lines)
out += f"\n\n---\n\n*Total: {verified_total} papers verificados | {oa_count} con acceso abierto | {downloaded} PDFs descargados en papers/.*\n"
open('dossier_literatura_video.md', 'w', encoding='utf-8').write(out)
print('OK -> dossier_literatura_video.md')
print(f'verificados={verified_total} OA={oa_count} descargados={downloaded}')
