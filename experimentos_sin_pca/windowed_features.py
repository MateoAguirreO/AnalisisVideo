"""Features por VENTANA de 10s (multi-instancia), no por participante completo.

Motivacion: AnxietyFaceTrack (Sahu et al. 2025) parte los primeros 90s del video
en ventanas de 10s -> de 91 participantes saca 1173 muestras de entrenamiento.
Aqui nunca se probo esto: v1/v2/experimentos_sin_pca siempre agregan TODA la
entrevista a 1 vector por participante (n=80). Reutilizando las series por-frame
ya guardadas (`features/series/*.npz`, ~10 fps efectivos tras el stride de 3),
se generan ~20-40 ventanas de 10s por participante (segun duracion del clip) ->
dataset de ~1600-2000 instancias en vez de 80.

IMPORTANTE: esto NO reemplaza el N real de sujetos (sigue habiendo 80 personas
distintas) -- son ventanas correlacionadas dentro del mismo sujeto. Por eso la
validacion (`windowed_nested_cv.py`) usa GroupKFold por CODIGO, nunca CV plano,
y la metrica que importa es el AUC a nivel PARTICIPANTE (promediando la
probabilidad de sus ventanas), no el AUC a nivel ventana (que seria optimista).

Uso:  python experimentos_sin_pca/windowed_features.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg

DIR_SERIES = cfg.DIR_FEATURES / "series"
OUT = Path(__file__).resolve().parent / "resultados" / "window_features.csv"

FPS_EFECTIVO = 30.0 / cfg.FRAME_STRIDE  # ~10 Hz (stride=3 sobre video a 30fps)
W_SEC = 10
W_ROWS = int(W_SEC * FPS_EFECTIVO)  # ~100 filas por ventana
MIN_VALID_FRAC = 0.5  # descarta la ventana final si tiene menos de la mitad de filas


def _stats_ventana(x, nombre):
    xv = x[~np.isnan(x)]
    if xv.size < 5:
        return {f"{nombre}_mean": np.nan, f"{nombre}_std": np.nan,
                f"{nombre}_p10": np.nan, f"{nombre}_p90": np.nan}
    return {
        f"{nombre}_mean": float(np.mean(xv)),
        f"{nombre}_std": float(np.std(xv)),
        f"{nombre}_p10": float(np.percentile(xv, 10)),
        f"{nombre}_p90": float(np.percentile(xv, 90)),
    }


def featurizar_ventanas(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    sig = d["signals"]
    names = list(d["names"])
    codigo = int(d["codigo"])
    n = sig.shape[0]
    n_win_completas = n // W_ROWS
    resto = n - n_win_completas * W_ROWS
    n_win = n_win_completas + (1 if resto >= MIN_VALID_FRAC * W_ROWS else 0)

    filas = []
    for w in range(n_win):
        ini, fin = w * W_ROWS, min((w + 1) * W_ROWS, n)
        seg = sig[ini:fin]
        col = {nm: seg[:, i] for i, nm in enumerate(names)}
        feat = {"participant": npz_path.stem, "codigo": codigo, "window": w}
        for nm in names:
            feat.update(_stats_ventana(col[nm], nm))

        blink = col["blink"]
        cnt, closed = 0, False
        for b in blink:
            if not np.isnan(b) and b > cfg.BLINK_UMBRAL and not closed:
                cnt += 1; closed = True
            elif np.isnan(b) or b <= cfg.BLINK_UMBRAL:
                closed = False
        dur_min_win = seg.shape[0] / FPS_EFECTIVO / 60.0
        feat["blink_rate_min"] = cnt / dur_min_win if dur_min_win else np.nan

        go, gd = col["gaze_out"], col["gaze_down"]
        feat["gaze_aversion_frac"] = float(np.nanmean(
            ((go > cfg.GAZE_AVERSION_UMBRAL) | (gd > cfg.GAZE_AVERSION_UMBRAL)).astype(float)))
        feat["mouth_open_frac"] = float(np.nanmean((col["jaw_open"] > cfg.MOUTH_OPEN_UMBRAL).astype(float)))

        ang = np.vstack([col["yaw"], col["pitch"], col["roll"]]).T
        dmov = np.linalg.norm(np.diff(ang, axis=0), axis=1)
        dmov = dmov[~np.isnan(dmov)]
        feat["head_motion_mean"] = float(np.mean(dmov)) if dmov.size else np.nan
        feat["hand_visible_frac"] = float(np.nanmean(col["hand_present"]))
        filas.append(feat)
    return filas


def main():
    npzs = sorted(DIR_SERIES.glob("*.npz"))
    if not npzs:
        print(f"No hay series en {DIR_SERIES}. Corre src/extract_series.py primero."); return
    filas = []
    for p in npzs:
        filas.extend(featurizar_ventanas(p))
    df = pd.DataFrame(filas).sort_values(["codigo", "window"]).reset_index(drop=True)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    n_feats = df.shape[1] - 3
    print(f"Ventanas de {W_SEC}s ({W_ROWS} filas c/u) -> {OUT}")
    print(f"  {df['codigo'].nunique()} participantes, {len(df)} ventanas totales "
          f"({len(df)/df['codigo'].nunique():.1f} ventanas/participante en promedio), {n_feats} features/ventana")
    print(df.groupby("codigo").size().describe())


if __name__ == "__main__":
    main()
