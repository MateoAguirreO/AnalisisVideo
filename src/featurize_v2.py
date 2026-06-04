"""v2 - Featurizacion RICA desde las series por-frame (features/series/*.npz).

Por cada senal calcula descriptores distribucionales (percentiles, IQR, asimetria,
curtosis), de DINAMICA (cambio entre frames) y de actividad; ademas features
derivadas (tasa de parpadeo, aversion de mirada) y por-SEGMENTO (cambio a lo largo
de la entrevista). Escribe features/video_features_v2.csv.

No toca nada de la v1.

Uso:  python src/featurize_v2.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

DIR_SERIES = cfg.DIR_FEATURES / "series"
OUT = cfg.DIR_FEATURES / "video_features_v2.csv"

# senales en formato 0-1 (blendshapes) para features de "actividad"
BLEND_SIGNALS = ["blink", "eye_squint", "eye_wide", "gaze_out", "gaze_down", "gaze_up",
                 "smile", "frown", "mouth_press", "jaw_open", "brow_down",
                 "brow_inner_up", "cheek_squint", "nose_sneer", "expresividad"]
SEG_SIGNALS = ["expresividad", "smile", "frown", "blink", "gaze_down", "gaze_out",
               "brow_down", "brow_inner_up", "jaw_open"]
N_SEG = 4


def _skew_kurt(x):
    m, s = np.nanmean(x), np.nanstd(x)
    if s < 1e-9 or np.isnan(s):
        return 0.0, 0.0
    z = (x[~np.isnan(x)] - m) / s
    return float(np.mean(z ** 3)), float(np.mean(z ** 4) - 3.0)


def _stats_senal(x, nombre, blend):
    f = {}
    xv = x[~np.isnan(x)]
    if xv.size == 0:
        return {f"{nombre}_{k}": np.nan for k in
                ["mean", "std", "median", "p10", "p90", "iqr", "skew", "kurt", "dyn", "dynstd", "act"]}
    f[f"{nombre}_mean"] = float(np.mean(xv))
    f[f"{nombre}_std"] = float(np.std(xv))
    f[f"{nombre}_median"] = float(np.median(xv))
    f[f"{nombre}_p10"] = float(np.percentile(xv, 10))
    f[f"{nombre}_p90"] = float(np.percentile(xv, 90))
    f[f"{nombre}_iqr"] = float(np.percentile(xv, 75) - np.percentile(xv, 25))
    sk, ku = _skew_kurt(x)
    f[f"{nombre}_skew"] = sk
    f[f"{nombre}_kurt"] = ku
    d = np.abs(np.diff(xv))                      # dinamica (cuanto se mueve la senal)
    f[f"{nombre}_dyn"] = float(np.mean(d)) if d.size else 0.0
    f[f"{nombre}_dynstd"] = float(np.std(d)) if d.size else 0.0
    # actividad: fraccion de tiempo "expresando" (solo blendshapes 0-1)
    f[f"{nombre}_act"] = float(np.mean(xv > 0.20)) if blend else np.nan
    return f


def _segmento(x, nombre):
    """std del promedio entre N segmentos + pendiente lineal (cambio en la entrevista)."""
    xv = x.copy()
    n = xv.size
    if n < N_SEG:
        return {f"{nombre}_segstd": np.nan, f"{nombre}_slope": np.nan}
    segs = np.array_split(xv, N_SEG)
    medias = [np.nanmean(s) if np.any(~np.isnan(s)) else np.nan for s in segs]
    medias = np.array(medias, dtype=float)
    segstd = float(np.nanstd(medias))
    if np.all(np.isnan(medias)):
        slope = np.nan
    else:
        xs = np.arange(N_SEG)[~np.isnan(medias)]
        ys = medias[~np.isnan(medias)]
        slope = float(np.polyfit(xs, ys, 1)[0]) if xs.size >= 2 else np.nan
    return {f"{nombre}_segstd": segstd, f"{nombre}_slope": slope}


def featurizar(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    sig = d["signals"]                  # [n_frames, K]
    names = list(d["names"])
    fps = float(d["fps"]); n_proc = int(d["n_proc"])
    col = {n: sig[:, i] for i, n in enumerate(names)}
    feat = {"participant": npz_path.stem, "codigo": int(d["codigo"])}

    for n in names:
        feat.update(_stats_senal(col[n], n, blend=n in BLEND_SIGNALS))
    for n in SEG_SIGNALS:
        if n in col:
            feat.update(_segmento(col[n], n))

    # --- derivadas ---
    blink = col["blink"]
    cnt, closed = 0, False
    for b in blink:
        if not np.isnan(b) and b > cfg.BLINK_UMBRAL and not closed:
            cnt += 1; closed = True
        elif np.isnan(b) or b <= cfg.BLINK_UMBRAL:
            closed = False
    dur_min = (n_proc * cfg.FRAME_STRIDE / fps) / 60.0 if fps else np.nan
    feat["blink_rate_min"] = cnt / dur_min if dur_min else np.nan
    go, gd = col["gaze_out"], col["gaze_down"]
    feat["gaze_aversion_frac"] = float(np.nanmean(
        ((go > cfg.GAZE_AVERSION_UMBRAL) | (gd > cfg.GAZE_AVERSION_UMBRAL)).astype(float)))
    feat["mouth_open_frac"] = float(np.nanmean((col["jaw_open"] > cfg.MOUTH_OPEN_UMBRAL).astype(float)))
    feat["hand_visible_frac"] = float(np.nanmean(col["hand_present"]))
    # energia de movimiento de cabeza (combinando yaw/pitch/roll)
    ang = np.vstack([col["yaw"], col["pitch"], col["roll"]]).T
    dmov = np.linalg.norm(np.diff(ang, axis=0), axis=1)
    dmov = dmov[~np.isnan(dmov)]
    feat["head_motion_mean"] = float(np.mean(dmov)) if dmov.size else np.nan
    feat["head_motion_std"] = float(np.std(dmov)) if dmov.size else np.nan
    feat["head_still_frac"] = float(np.mean(dmov < 0.5)) if dmov.size else np.nan
    return feat


def main():
    npzs = sorted(DIR_SERIES.glob("*.npz"))
    if not npzs:
        print(f"No hay series en {DIR_SERIES}. Corre extract_series.py primero."); return
    filas = [featurizar(p) for p in npzs]
    df = pd.DataFrame(filas).sort_values("codigo").reset_index(drop=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"v2 features -> {OUT}  ({len(df)} participantes, {df.shape[1]-2} features)")


if __name__ == "__main__":
    main()
