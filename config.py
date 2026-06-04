"""Rutas y parametros globales del pipeline multimodal (video)."""
from pathlib import Path

ROOT = Path(__file__).parent

# --- Datos de entrada ---
RUTA_MUESTRAS = Path(
    r"c:/Users/USUARIO/Documents/Maestria/analisis-espectrogramas/.MUESTRAS_SAMANÁ/Muestras"
)
RUTA_LABELS = Path(
    r"c:/Users/USUARIO/Documents/Maestria/analisis-espectrogramas/.MUESTRAS_SAMANÁ/Base de Datos Trabajo de campo RESULTADOS.xlsx"
)

# --- Modelos MediaPipe (Tasks API) ---
DIR_MODELOS = ROOT / "models"
MODELO_FACE = DIR_MODELOS / "face_landmarker.task"
MODELO_POSE = DIR_MODELOS / "pose_landmarker_lite.task"
MODELO_HAND = DIR_MODELOS / "hand_landmarker.task"

# --- Salidas ---
DIR_FEATURES = ROOT / "features"
DIR_RESULTADOS = ROOT / "resultados"
for _d in (DIR_FEATURES, DIR_RESULTADOS):
    _d.mkdir(exist_ok=True)

VIDEO_FEATURES_CSV = DIR_FEATURES / "video_features.csv"

# --- Extraccion de video ---
# El video es 30 fps; procesamos 1 de cada FRAME_STRIDE frames (3 -> ~10 fps),
# suficiente para parpadeo/gaze/cabeza/micro-gestos y mucho mas rapido.
FRAME_STRIDE = 3
BLINK_UMBRAL = 0.5           # blendshape eyeBlink > umbral cuenta como ojo cerrado
GAZE_AVERSION_UMBRAL = 0.30  # blendshape de mirada lateral/abajo > umbral = aversion
MOUTH_OPEN_UMBRAL = 0.25     # jawOpen > umbral = boca abierta (habla/actividad)

# --- Evaluacion (espejo del pipeline de audio: RepeatedStratifiedKFold 5x5) ---
N_SPLITS = 5
N_REPEATS = 5
RANDOM_STATE = 42

# Diagnosticos objetivo
DXS = ["ansiedad", "depresion"]
