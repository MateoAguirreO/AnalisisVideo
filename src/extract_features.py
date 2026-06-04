"""Extraccion de biomarcadores de VIDEO factibles a 30 fps (cara frontal).

Usa la MediaPipe **Tasks API**:
  - FaceLandmarker -> 52 blendshapes (señales tipo Action Unit: parpadeo, sonrisa,
    ceño, mirada, mandibula...) + matriz de transformacion facial (pose de cabeza).
  - PoseLandmarker / HandLandmarker -> micro-gestos corporales (energia de movimiento).

Features agregadas por participante (todas justificadas en el dossier; ninguna
requiere alta-fps ni eye-tracker IR):
  OCULAR:   tasa de parpadeo, eyeBlink, eyeSquint, eyeWide, gaze lateral/abajo, aversion
  AFECTO:   sonrisa, ceño, cejas (browDown/InnerUp), expresividad global (afecto plano)
  BOCA:     jawOpen, mouthPress, fraccion boca abierta (proxy de habla)
  CABEZA:   yaw/pitch/roll (media+std) + energia de movimiento (retardo psicomotor)
  CUERPO:   movimiento de cuerpo y manos, mano-a-cara

Uso:
    python src/extract_features.py                      # todas las muestras
    python src/extract_features.py --solo 1 --max-frames 300   # smoke test
    python src/extract_features.py --no-body            # sin Pose/Hands (mas rapido)
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from labels import cargar_labels

import mediapipe as mp
from mediapipe.tasks import python as mpy
from mediapipe.tasks.python import vision

# Grupos de blendshapes -> señal derivada interpretable (se promedian L/R)
BLEND_GROUPS = {
    "blink":       ["eyeBlinkLeft", "eyeBlinkRight"],
    "eye_squint":  ["eyeSquintLeft", "eyeSquintRight"],
    "eye_wide":    ["eyeWideLeft", "eyeWideRight"],
    "gaze_out":    ["eyeLookOutLeft", "eyeLookOutRight"],
    "gaze_down":   ["eyeLookDownLeft", "eyeLookDownRight"],
    "gaze_up":     ["eyeLookUpLeft", "eyeLookUpRight"],
    "smile":       ["mouthSmileLeft", "mouthSmileRight"],
    "frown":       ["mouthFrownLeft", "mouthFrownRight"],
    "mouth_press": ["mouthPressLeft", "mouthPressRight"],
    "jaw_open":    ["jawOpen"],
    "brow_down":   ["browDownLeft", "browDownRight"],
    "brow_inner_up": ["browInnerUp"],
    "cheek_squint": ["cheekSquintLeft", "cheekSquintRight"],
    "nose_sneer":  ["noseSneerLeft", "noseSneerRight"],
}
# blendshapes que cuentan como "expresividad" (afecto). Bajo promedio ~ afecto plano.
EXPRESIVIDAD = ["mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
                "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
                "browOuterUpRight", "cheekSquintLeft", "cheekSquintRight", "jawOpen",
                "noseSneerLeft", "noseSneerRight", "mouthPucker", "mouthStretchLeft",
                "mouthStretchRight"]


def _euler(M):
    R = np.asarray(M)[:3, :3]
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    pitch = np.degrees(np.arctan2(-R[2, 0], sy))
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return yaw, pitch, roll


def _stats(arr, pref):
    a = np.asarray([x for x in arr if x is not None and not np.isnan(x)], dtype=float)
    if a.size == 0:
        return {f"{pref}_mean": np.nan, f"{pref}_std": np.nan}
    return {f"{pref}_mean": float(np.mean(a)), f"{pref}_std": float(np.std(a))}


def _crear_landmarkers(do_body):
    fo = vision.FaceLandmarkerOptions(
        base_options=mpy.BaseOptions(model_asset_path=str(cfg.MODELO_FACE)),
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True)
    face = vision.FaceLandmarker.create_from_options(fo)
    pose = hand = None
    if do_body:
        po = vision.PoseLandmarkerOptions(
            base_options=mpy.BaseOptions(model_asset_path=str(cfg.MODELO_POSE)),
            running_mode=vision.RunningMode.VIDEO, num_poses=1)
        pose = vision.PoseLandmarker.create_from_options(po)
        ho = vision.HandLandmarkerOptions(
            base_options=mpy.BaseOptions(model_asset_path=str(cfg.MODELO_HAND)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2)
        hand = vision.HandLandmarker.create_from_options(ho)
    return face, pose, hand


def procesar_video(ruta, frame_stride=3, max_frames=None, do_body=True):
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {ruta}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    face, pose, hand = _crear_landmarkers(do_body)

    series = {g: [] for g in BLEND_GROUPS}
    expr_s, yaw_s, pit_s, rol_s, headmov_s, bodymov_s, handmov_s = [], [], [], [], [], [], []
    n_proc = n_face = n_body = n_hand = n_handface = 0
    prev_angles = prev_pose = prev_hand = None
    last_ts = -1
    idx = -1

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

        fr = face.detect_for_video(mp_img, ts)
        if fr.face_blendshapes:
            n_face += 1
            bs = {c.category_name: c.score for c in fr.face_blendshapes[0]}
            for g, names in BLEND_GROUPS.items():
                series[g].append(float(np.mean([bs.get(n, 0.0) for n in names])))
            expr_s.append(float(np.mean([bs.get(n, 0.0) for n in EXPRESIVIDAD])))
            if fr.facial_transformation_matrixes:
                yaw, pit, rol = _euler(fr.facial_transformation_matrixes[0])
                yaw_s.append(yaw); pit_s.append(pit); rol_s.append(rol)
                if prev_angles is not None:
                    headmov_s.append(float(np.linalg.norm(
                        np.array([yaw, pit, rol]) - np.array(prev_angles))))
                prev_angles = (yaw, pit, rol)

        if do_body:
            pr = pose.detect_for_video(mp_img, ts)
            if pr.pose_landmarks:
                n_body += 1
                pl = pr.pose_landmarks[0]
                sw = abs(pl[11].x - pl[12].x) * w + 1e-6
                key = np.array([[pl[i].x * w, pl[i].y * h] for i in (11, 12, 13, 14, 15, 16)])
                if prev_pose is not None:
                    bodymov_s.append(float(np.mean(np.linalg.norm(key - prev_pose, axis=1)) / sw))
                prev_pose = key
            hr = hand.detect_for_video(mp_img, ts)
            if hr.hand_landmarks:
                n_hand += 1
                wr = hr.hand_landmarks[0][0]
                cur_h = np.array([wr.x * w, wr.y * h])
                if prev_hand is not None:
                    handmov_s.append(float(np.linalg.norm(cur_h - prev_hand) / (w + 1e-6)))
                prev_hand = cur_h
                if fr.face_landmarks:
                    nose = np.array([fr.face_landmarks[0][1].x * w, fr.face_landmarks[0][1].y * h])
                    if np.linalg.norm(cur_h - nose) < 0.25 * h:
                        n_handface += 1

    cap.release()
    face.close()
    if pose: pose.close()
    if hand: hand.close()

    # parpadeos: cruces hacia arriba del umbral
    blink_count, closed = 0, False
    for b in series["blink"]:
        if b > cfg.BLINK_UMBRAL and not closed:
            blink_count += 1; closed = True
        elif b <= cfg.BLINK_UMBRAL:
            closed = False
    dur_min = (n_proc * frame_stride / fps) / 60.0 if fps else np.nan

    feat = {
        "n_frames_proc": n_proc,
        "face_detect_rate": n_face / n_proc if n_proc else 0.0,
        "dur_min": round(dur_min, 3) if dur_min else np.nan,
        "blink_count": blink_count,
        "blink_rate_min": blink_count / dur_min if dur_min else np.nan,
        "gaze_aversion_frac": float(np.mean([
            1.0 if (o > cfg.GAZE_AVERSION_UMBRAL or d > cfg.GAZE_AVERSION_UMBRAL) else 0.0
            for o, d in zip(series["gaze_out"], series["gaze_down"])])) if series["gaze_out"] else np.nan,
        "mouth_open_frac": float(np.mean([1.0 if j > cfg.MOUTH_OPEN_UMBRAL else 0.0
                                          for j in series["jaw_open"]])) if series["jaw_open"] else np.nan,
        "head_still_frac": float(np.mean([1.0 if m < 0.5 else 0.0 for m in headmov_s])) if headmov_s else np.nan,
    }
    for g in BLEND_GROUPS:
        feat.update(_stats(series[g], g))
    feat.update(_stats(expr_s, "expresividad"))
    feat.update(_stats(yaw_s, "yaw"))
    feat.update(_stats(pit_s, "pitch"))
    feat.update(_stats(rol_s, "roll"))
    feat.update(_stats(headmov_s, "head_motion"))
    if do_body:
        feat["body_detect_rate"] = n_body / n_proc if n_proc else 0.0
        feat["hand_visible_frac"] = n_hand / n_proc if n_proc else 0.0
        feat["hand_to_face_rate"] = n_handface / n_proc if n_proc else 0.0
        feat.update(_stats(bodymov_s, "body_motion"))
        feat.update(_stats(handmov_s, "hand_motion"))
    return feat


def _video_de_carpeta(carpeta: Path):
    mp4s = [p for p in carpeta.glob("*.mp4") if not p.name.startswith("._")]
    return max(mp4s, key=lambda p: p.stat().st_size) if mp4s else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=None, help="cap de frames (smoke test)")
    ap.add_argument("--solo", type=int, default=None, help="procesar solo el codigo N")
    ap.add_argument("--no-body", action="store_true", help="omitir Pose/Hands (mas rapido)")
    ap.add_argument("--out", type=str, default=str(cfg.VIDEO_FEATURES_CSV))
    ap.add_argument("--shard", type=str, default=None,
                    help="i/N -> procesa solo carpetas con indice %% N == i (paralelizacion)")
    ap.add_argument("--skip-from", type=str, default=None,
                    help="CSV(s) separados por coma cuyos 'participant' ya hechos se saltan")
    args = ap.parse_args()

    shard_i = shard_n = None
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split("/"))

    lab = cargar_labels().set_index("documento")
    carpetas = sorted([p for p in cfg.RUTA_MUESTRAS.iterdir()
                       if p.is_dir() and not p.name.startswith(".")])
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)

    # reanudacion: saltar participantes ya presentes en el CSV de salida...
    hechos = set()
    if out_path.exists():
        try:
            hechos = set(pd.read_csv(out_path)["participant"].astype(str))
            print(f"Reanudando: {len(hechos)} participantes ya en {out_path.name}.")
        except Exception:
            hechos = set()
    # ...y los presentes en otros CSV (p. ej. la corrida previa)
    for extra in (args.skip_from or "").split(","):
        extra = extra.strip()
        if extra and Path(extra).exists():
            try:
                hechos |= set(pd.read_csv(extra)["participant"].astype(str))
            except Exception:
                pass

    filas, t_ini = [], time.time()
    for idx, carp in enumerate(carpetas):
        if shard_n is not None and idx % shard_n != shard_i:
            continue
        partes = carp.name.split("_")
        try:
            codigo = int(partes[0]); doc = int(partes[1])
        except (ValueError, IndexError):
            continue
        if args.solo is not None and codigo != args.solo:
            continue
        if carp.name in hechos:
            continue
        vid = _video_de_carpeta(carp)
        if vid is None:
            print(f"  [skip] sin video: {carp.name}"); continue
        t0 = time.time()
        try:
            feat = procesar_video(vid, cfg.FRAME_STRIDE, args.max_frames, not args.no_body)
        except Exception as e:
            print(f"  [ERROR] {carp.name}: {e}"); continue
        fila = {"participant": carp.name, "codigo": codigo, "documento": doc,
                "ansiedad": np.nan, "depresion": np.nan}
        if doc in lab.index:
            fila["ansiedad"] = lab.loc[doc, "ansiedad"]
            fila["depresion"] = lab.loc[doc, "depresion"]
        fila.update(feat)
        filas.append(fila)
        # escritura incremental (append): el progreso sobrevive a fallos
        df_fila = pd.DataFrame([fila])
        df_fila.to_csv(out_path, mode="a", header=not out_path.exists(),
                       index=False, encoding="utf-8")
        print(f"  [ok] {carp.name}  frames={feat['n_frames_proc']} "
              f"face={feat['face_detect_rate']:.2f} blink/min={feat['blink_rate_min']:.1f} "
              f"smile={feat.get('smile_mean', float('nan')):.3f} ({time.time()-t0:.1f}s)")

    n_total = len(hechos) + len(filas)
    print(f"\nListo. Procesados ahora: {len(filas)}  |  total en {out_path.name}: {n_total}  "
          f"|  tiempo: {(time.time()-t_ini)/60:.1f} min")


if __name__ == "__main__":
    main()
