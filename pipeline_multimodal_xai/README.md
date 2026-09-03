# pipeline_multimodal_xai — fusión audio + video con explicabilidad (SHAP + LIME)

Componente de **fusión multimodal** de la tesis (riesgo de ansiedad / depresión,
muestra de campo Samaná, hilo XAI). Combina las dos modalidades que por separado
son débiles-a-moderadas:

- **voz** — 88 funcionales eGeMAPS v02, a nivel **segmento** (`features_<dx>_egemaps.csv`)
- **rostro** — 66 features de Action Units / pose de cabeza / prototipos de emoción,
  a nivel participante (`dataset_au_features.csv`)

n = **79 participantes** (audio ∩ video; el 66 no tiene audio). Modelado
**independiente por eje** (nunca conjunto ansiedad/depresión). Evaluación con el
mismo esquema honesto que `experimentos_sin_pca/` (nested CV, permutación, N pequeño).

## Instalación

```bash
pip install -r pipeline_multimodal_xai/requirements.txt   # lime; el resto ya está en ../requirements.txt
```

## Ejecución

```bash
python pipeline_multimodal_xai/run_all.py         # todo, en orden
# o por pasos:
python pipeline_multimodal_xai/data_multimodal.py         # chequeo de carga (n=79, conteos)
python pipeline_multimodal_xai/eval_multimodal.py         # baselines + nested CV -> resultados/
python pipeline_multimodal_xai/permutation_multimodal.py  # test de permutación del ganador
python pipeline_multimodal_xai/sensibilidad_agg.py        # sensibilidad a la agregación de audio
python pipeline_multimodal_xai/xai_shap.py                # SHAP: modalidad + intra-modalidad + familias
python pipeline_multimodal_xai/xai_lime.py                # LIME local (TP/FP/TN) + consistencia SHAP/LIME
python pipeline_multimodal_xai/xai_stability.py           # estabilidad de features y del peso de modalidad
```

Costo aprox. (portátil, CPU): `eval` ~12 min, `permutation` ~15 min, `sensibilidad_agg` ~1 min,
`xai_shap` ~1 min, `xai_lime` ~1 min, `xai_stability` ~10 min.

## Arquitectura

| Etapa | Qué hace |
|---|---|
| **Rama audio** (`audio_branch.py`) | clasificador eGeMAPS a nivel segmento → agrega la probabilidad de los segmentos de cada participante (media). Los scores de entrenamiento son OOF (`StratifiedGroupKFold`, grupo = participante) → sin fuga. |
| **Rama video** (`fusion.py`) | pipeline sin PCA de `experimentos_sin_pca/feature_selectors.py` a nivel participante. |
| **Late fusion** (`fusion.py::MetaFusion`) | meta-modelo sobre `[score_audio, score_video]`: soft/hard voting, ponderado (α por grid y α AUC-proporcional), stacking-logístico, stacking-ANN. |
| **Early fusion** (`fusion.py`) | agrega los segmentos de audio a un vector por participante (media/std/p20/p50/p80), concatena con video (prefijos `aud__`/`vid__`), un solo pipeline de selección + modelo. |
| **Evaluación** (`eval_multimodal.py`) | screening (10 folds) elige config de cada rama; confirmación (25 folds) evalúa todos los baselines y variantes. |
| **XAI** | SHAP a nivel modalidad (voz vs rostro), SHAP intra-modalidad por feature y por familia, SHAP unificado del early fusion, LIME local por paciente, consistencia SHAP/LIME, estabilidad por bootstrap. |

## Salidas (`resultados/`)

Se versionan igual que `experimentos_sin_pca/resultados/` (son agregados/derivados, sin datos
crudos ni identificadores de participante — los `pid` son códigos 1–80).

- `screening_<dx>.csv`, `metricas_multimodal_<dx>.csv`, `best_config_<dx>.json`
- `permutation_multimodal.csv`, `sensibilidad_agg.csv`
- `shap_modalidad_<dx>.{csv,png}`, `shap_audio_<dx>.*`, `shap_video_<dx>.*`, `shap_*_familias_<dx>.csv`, `shap_early_<dx>.csv`
- `lime_<dx>_{TP,FP,TN}.{txt,png}`, `consistencia_shap_lime.csv`
- `estabilidad_{video,audio,modalidad}_<dx>.csv`
- **`../REPORTE_multimodal.md`** — informe estilo tesis con todos los números y la lectura clínica.

## Notas metodológicas

- El CSV de AU (66 features) es una extracción distinta de `video_features_v2.csv` (256
  temporales) usada en `experimentos_sin_pca/`. El baseline solo-video de esta carpeta
  **no** es el número canónico de video de la tesis — ver limitaciones en el REPORTE.
- Las etiquetas salen de `target_<dx>` del CSV de video y se verifica que coincidan con el
  `label` de segmento del audio (coinciden al 100% por participante).
- `error` (100% NaN) y `n_frames_detected` (control de calidad) se excluyen del modelado.
