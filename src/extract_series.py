"""v2 - Extrae SERIES POR-FRAME (no agregados) y las guarda a disco.

Para cada participante guarda features/series/{participant}.npz con una matriz
[n_frames, K] de senales por frame + sus nombres. Esto permite calcular features
temporales ricas (percentiles, dinamica, por-segmento) sin re-extraer.

Reutiliza la configuracion de MediaPipe de extract_features.py.

Uso:
    python src/extract_series.py                  # todas (resume: salta .npz ya hechos)
    python src/extract_series.py --shard 0/5      # paralelizacion
    python src/extract_series.py --solo 1 --max-frames 300
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
import mediapipe as mp
from extract_features import (_crear_landmarkers, _euler, _video_de_carpeta,
                              BLEND_GROUPS, EXPRESIVIDAD)

DIR_SERIES = cfg.DIR_FEATURES / "series"
DIR_SERIES.mkdir(parents=True, exist_ok=True)

# orden de senales por frame
SIG = list(BLEND_GROUPS.keys()) + ["expresividad", "yaw", "pitch", "roll",
                                    "body_motion", "hand_motion", "hand_present"]


def procesar_series(ruta, frame_stride=3, max_frames=None):
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {ruta}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    face, pose, hand = _crear_landmarkers(do_body=True)

    rows = []
    prev_pose = prev_hand = None
    last_ts = -1
    idx = -1
    n_proc = n_face = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % frame_stride != 0:
            continue
        n_proc += 1
        if max_frames and n_proc > max_frames:
            break
        ts = max(int(idx / fps * 1000), last_ts + 1)
        last_ts = ts
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        row = {s: np.nan for s in SIG}
        fr = face.detect_for_video(mp_img, ts)
        if fr.face_blendshapes:
            n_face += 1
            bs = {c.category_name: c.score for c in fr.face_blendshapes[0]}
            for g, names in BLEND_GROUPS.items():
                row[g] = float(np.mean([bs.get(n, 0.0) for n in names]))
            row["expresividad"] = float(np.mean([bs.get(n, 0.0) for n in EXPRESIVIDAD]))
            if fr.facial_transformation_matrixes:
                yaw, pit, rol = _euler(fr.facial_transformation_matrixes[0])
                row["yaw"], row["pitch"], row["roll"] = yaw, pit, rol

        pr = pose.detect_for_video(mp_img, ts)
        if pr.pose_landmarks:
            pl = pr.pose_landmarks[0]
            sw = abs(pl[11].x - pl[12].x) * w + 1e-6
            key = np.array([[pl[i].x * w, pl[i].y * h] for i in (11, 12, 13, 14, 15, 16)])
            if prev_pose is not None:
                row["body_motion"] = float(np.mean(np.linalg.norm(key - prev_pose, axis=1)) / sw)
            prev_pose = key
        hr = hand.detect_for_video(mp_img, ts)
        row["hand_present"] = 1.0 if hr.hand_landmarks else 0.0
        if hr.hand_landmarks:
            wr = hr.hand_landmarks[0][0]
            cur_h = np.array([wr.x * w, wr.y * h])
            if prev_hand is not None:
                row["hand_motion"] = float(np.linalg.norm(cur_h - prev_hand) / (w + 1e-6))
            prev_hand = cur_h
        rows.append([row[s] for s in SIG])

    cap.release(); face.close(); pose.close(); hand.close()
    return np.array(rows, dtype=np.float32), fps, n_proc, n_face


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--solo", type=int, default=None)
    ap.add_argument("--shard", type=str, default=None, help="i/N")
    args = ap.parse_args()
    shard_i = shard_n = None
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split("/"))

    carpetas = sorted([p for p in cfg.RUTA_MUESTRAS.iterdir()
                       if p.is_dir() and not p.name.startswith(".")])
    n_ok = 0
    for idx, carp in enumerate(carpetas):
        if shard_n is not None and idx % shard_n != shard_i:
            continue
        try:
            codigo = int(carp.name.split("_")[0])
        except (ValueError, IndexError):
            continue
        if args.solo is not None and codigo != args.solo:
            continue
        dest = DIR_SERIES / f"{carp.name}.npz"
        if dest.exists():
            continue
        vid = _video_de_carpeta(carp)
        if vid is None:
            print(f"  [skip] sin video: {carp.name}"); continue
        t0 = time.time()
        try:
            arr, fps, n_proc, n_face = procesar_series(vid, cfg.FRAME_STRIDE, args.max_frames)
        except Exception as e:
            print(f"  [ERROR] {carp.name}: {e}"); continue
        np.savez_compressed(dest, signals=arr, names=np.array(SIG),
                            fps=fps, n_proc=n_proc, n_face=n_face, codigo=codigo)
        n_ok += 1
        print(f"  [ok] {carp.name}  frames={n_proc} face={n_face/max(n_proc,1):.2f} ({time.time()-t0:.1f}s)")
    print(f"\nSeries guardadas: {n_ok}  en {DIR_SERIES}")


if __name__ == "__main__":
    main()
