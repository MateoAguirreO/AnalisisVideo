"""Config del pipeline multimodal audio(eGeMAPS) + video(AU) con explicabilidad.

Aislado del resto del repo: rutas propias, carpeta de resultados propia. Reutiliza
solo las constantes de evaluacion de ../config.py para que el esquema de CV sea el
MISMO que el de los componentes de audio y video por separado (RepeatedStratifiedKFold
5x5, RANDOM_STATE=42).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # pipeline_multimodal_xai/
REPO = ROOT.parent                              # raiz del repo

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experimentos_sin_pca"))

import config as _cfg  # noqa: E402

RANDOM_STATE = _cfg.RANDOM_STATE   # 42
N_SPLITS = _cfg.N_SPLITS           # 5
N_REPEATS = _cfg.N_REPEATS         # 5
DXS = list(_cfg.DXS)               # ["ansiedad", "depresion"]

# --- Datos de entrada (los 3 CSV que definen esta tarea) ---
CSV_VIDEO_AU = REPO / "dataset_au_features.csv"
CSV_AUDIO = {
    "ansiedad": REPO / "features_ansiedad_egemaps.csv",
    "depresion": REPO / "features_depresion_egemaps.csv",
}

# --- Salidas ---
DIR_RESULTADOS = ROOT / "resultados"
DIR_RESULTADOS.mkdir(exist_ok=True)

# --- Columnas del CSV de video que NO son features ---
VIDEO_ID_COL = "video_id"
VIDEO_TARGET_COLS = {"ansiedad": "target_ansiedad", "depresion": "target_depresion"}
# 'error' es 100% NaN; 'n_frames_detected' es control de calidad (posible confusor).
VIDEO_DROP_COLS = ["error", "n_frames_detected"]

# --- Columnas del CSV de audio que NO son features ---
AUDIO_ID_COL = "audio_id"
AUDIO_SEG_COL = "seg_idx"
AUDIO_LABEL_COL = "label"

# --- Agregacion segmento -> participante ---
AUDIO_PROB_AGG = "mean"   # como se resume la prob. de segmento (ver sensibilidad_agg.py)
EARLY_FUSION_AGG = ["mean", "std", "p20", "p50", "p80"]  # agregacion de FEATURES para early fusion

# --- Grid de features finales (EPV: ~20 positivos -> un solo digito) ---
K_GRID = [4, 6, 8]

# --- Familias de features para el SHAP agrupado ---
FAMILIAS_AUDIO = {
    "F0/prosodia": ["F0semitone", "logRelF0"],
    "loudness/energia": ["loudness", "equivalentSoundLevel", "loudnessPeaksPerSec"],
    "espectral": ["spectralFlux", "alphaRatio", "hammarbergIndex", "slopeV", "slopeUV"],
    "MFCC": ["mfcc"],
    "calidad de voz": ["jitter", "shimmer", "HNRdBACF"],
    "formantes": ["F1", "F2", "F3"],
    "temporal/tasa": ["VoicedSegmentsPerSec", "MeanVoicedSegmentLength",
                      "StddevVoicedSegmentLength", "MeanUnvoicedSegmentLength",
                      "StddevUnvoicedSegmentLength"],
}
FAMILIAS_VIDEO = {
    "cara superior": ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU43"],
    "cara inferior": ["AU10", "AU11", "AU12", "AU14", "AU15", "AU17", "AU20",
                      "AU23", "AU24", "AU25", "AU26", "AU28"],
    "pose de cabeza": ["Pitch", "Roll", "Yaw", "X_", "Y_", "Z_"],
    "emocion": ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"],
}


def familia_de(feature: str, familias: dict) -> str:
    """Devuelve el nombre de la familia a la que pertenece `feature` (o 'otra')."""
    base = feature.split("__", 1)[-1]  # quita prefijo aud__/vid__ si lo hay
    for fam, patrones in familias.items():
        if any(p in base for p in patrones):
            return fam
    return "otra"
