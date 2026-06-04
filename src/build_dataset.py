"""Une features de video + etiquetas -> dataset por diagnostico (ansiedad/depresion).

Genera features/dataset_ansiedad.csv y features/dataset_depresion.csv,
cada uno con: participant + columnas de features + columna 'label' (0/1).
Reporta balance de clases y valores faltantes.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from labels import cargar_labels

ID_COLS = ["participant", "codigo", "documento"]
LABEL_COLS = ["ansiedad", "depresion"]
# Confounds de duracion/calidad de grabacion (NO biomarcadores): se excluyen para
# evitar que el modelo aprenda la longitud del clip en vez de comportamiento.
# (dur_min correlaciona con la etiqueta; blink_count/n_frames dependen de la longitud.)
META_EXCLUDE = ["n_frames_proc", "dur_min", "blink_count",
                "face_detect_rate", "body_detect_rate"]


def feature_cols(df):
    return [c for c in df.columns if c not in ID_COLS + LABEL_COLS + META_EXCLUDE]


def _features_con_labels_limpias():
    """Lee el CSV de features y RE-DERIVA las etiquetas desde el Excel por
    CODIGO (llave fiable; la cedula no cruza para 11 participantes)."""
    df = pd.read_csv(cfg.VIDEO_FEATURES_CSV)
    df = df.drop(columns=[c for c in LABEL_COLS if c in df.columns])
    lab = cargar_labels()[["codigo", "ansiedad", "depresion"]]
    df["codigo"] = pd.to_numeric(df["codigo"], errors="coerce").astype("Int64")
    return df.merge(lab, on="codigo", how="left")


def build(dx):
    df = _features_con_labels_limpias()
    if dx not in df.columns:
        raise ValueError(f"No existe columna de etiqueta '{dx}'")
    feats = feature_cols(df)
    out = df[["participant"] + feats].copy()
    out["label"] = pd.to_numeric(df[dx], errors="coerce")
    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    dest = cfg.DIR_FEATURES / f"dataset_{dx}.csv"
    out.to_csv(dest, index=False, encoding="utf-8")
    vc = out["label"].value_counts().to_dict()
    miss = int(out[feats].isna().sum().sum())
    print(f"[{dx}] {len(out)} participantes  clases={vc}  features={len(feats)}  NaN_celdas={miss}")
    print(f"       -> {dest}")
    return out


def main():
    for dx in cfg.DXS:
        try:
            build(dx)
        except Exception as e:
            print(f"[{dx}] ERROR: {e}")


if __name__ == "__main__":
    main()
