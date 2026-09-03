"""Carga y alineacion de los 3 CSV de entrada por CODIGO de participante.

- Video (`dataset_au_features.csv`): 1 fila / participante, 68 features utiles.
- Audio (`features_<dx>_egemaps.csv`): 1 fila / segmento, 88 features eGeMAPS, label por segmento.
- Cruce por codigo (prefijo numerico del audio_id == video_id). NO por cedula.
- Se conservan solo los participantes presentes en AMBAS modalidades (n=79; el 66 no tiene audio).
- Las etiquetas se toman de `target_<dx>` del CSV de video; se verifica que coincidan
  con el `label` de segmento del CSV de audio (coinciden al 100% por participante).

Uso:  python pipeline_multimodal_xai/data_multimodal.py
"""
import re
import numpy as np
import pandas as pd

import config_mm as cfg


def _pid_de_audio(audio_id: str) -> int:
    return int(re.match(r"(\d+)", str(audio_id)).group(1))


def feature_cols_video(df: pd.DataFrame) -> list[str]:
    excl = {cfg.VIDEO_ID_COL, "pid", "label",
            *cfg.VIDEO_DROP_COLS, *cfg.VIDEO_TARGET_COLS.values()}
    return [c for c in df.columns if c not in excl]


def feature_cols_audio(df: pd.DataFrame) -> list[str]:
    excl = {cfg.AUDIO_ID_COL, cfg.AUDIO_SEG_COL, cfg.AUDIO_LABEL_COL, "pid"}
    return [c for c in df.columns if c not in excl]


def cargar(dx: str):
    """Devuelve (video_df, audio_seg_df) alineados a los participantes comunes.

    video_df:      index=pid, columnas = features de video + 'label'
    audio_seg_df:  columnas = 'pid', 'seg_idx', features eGeMAPS, 'label' (nivel segmento)
    """
    if dx not in cfg.DXS:
        raise ValueError(dx)

    vid = pd.read_csv(cfg.CSV_VIDEO_AU)
    vid["pid"] = vid[cfg.VIDEO_ID_COL].astype(int)
    tgt = cfg.VIDEO_TARGET_COLS[dx]
    vid = vid.dropna(subset=[tgt]).copy()
    vid["label"] = vid[tgt].astype(int)
    vid = vid.drop(columns=[c for c in cfg.VIDEO_DROP_COLS if c in vid.columns])
    vid = vid.drop(columns=[c for c in cfg.VIDEO_TARGET_COLS.values() if c in vid.columns])

    aud = pd.read_csv(cfg.CSV_AUDIO[dx])
    aud["pid"] = aud[cfg.AUDIO_ID_COL].map(_pid_de_audio)
    aud["label"] = aud[cfg.AUDIO_LABEL_COL].astype(int)

    comunes = sorted(set(vid["pid"]) & set(aud["pid"]))
    vid = vid[vid["pid"].isin(comunes)].set_index("pid").sort_index()
    aud = aud[aud["pid"].isin(comunes)].sort_values(["pid", cfg.AUDIO_SEG_COL]).reset_index(drop=True)

    # verificacion de consistencia de etiqueta entre modalidades
    lab_aud = aud.groupby("pid")["label"].first()
    desalineados = (vid["label"] != lab_aud.reindex(vid.index)).sum()
    if desalineados:
        raise RuntimeError(f"[{dx}] {desalineados} participantes con label video != label audio")

    fc_v = feature_cols_video(vid)
    vid = vid[fc_v + ["label"]]
    fc_a = feature_cols_audio(aud)
    aud = aud[["pid", cfg.AUDIO_SEG_COL] + fc_a + ["label"]]
    return vid, aud


def resumen():
    print(f"{'eje':10s} {'n':>4s} {'neg/pos':>10s} {'segmentos':>10s} {'seg/part (med)':>14s}")
    for dx in cfg.DXS:
        vid, aud = cargar(dx)
        vc = vid["label"].value_counts().to_dict()
        seg_por = aud.groupby("pid").size()
        print(f"{dx:10s} {len(vid):4d} {vc.get(0,0):5d}/{vc.get(1,0):<4d} "
              f"{len(aud):10d} {int(np.median(seg_por)):14d}")
        print(f"           features video={len(feature_cols_video(vid))}  "
              f"features audio={len(feature_cols_audio(aud))}")


if __name__ == "__main__":
    resumen()
