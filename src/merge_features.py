"""Combina la corrida seed + los CSV de los workers en un unico video_features.csv.

- Une features/video_features.csv (seed) + features/_worker_*.csv
- Deduplica por 'participant'
- Re-deriva etiquetas limpias (ansiedad/depresion) desde el Excel por documento
- Reescribe features/video_features.csv ordenado
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from labels import cargar_labels

LABEL_COLS = ["ansiedad", "depresion"]


def main():
    partes = []
    seed = cfg.VIDEO_FEATURES_CSV
    if seed.exists():
        partes.append(pd.read_csv(seed))
    for wf in sorted(cfg.DIR_FEATURES.glob("_worker_*.csv")):
        try:
            partes.append(pd.read_csv(wf))
        except Exception as e:
            print(f"  (omito {wf.name}: {e})")
    if not partes:
        print("No hay nada que combinar."); return

    df = pd.concat(partes, ignore_index=True)
    df = df.drop_duplicates(subset="participant", keep="first")

    # re-derivar etiquetas limpias por CODIGO (la cedula no cruza para 11 participantes)
    df = df.drop(columns=[c for c in LABEL_COLS if c in df.columns])
    df["codigo"] = pd.to_numeric(df["codigo"], errors="coerce").astype("Int64")
    lab = cargar_labels()[["codigo", "ansiedad", "depresion"]]
    df = df.merge(lab, on="codigo", how="left")

    # reordenar columnas: ids + labels primero
    front = [c for c in ["participant", "codigo", "documento", "ansiedad", "depresion"] if c in df.columns]
    df = df[front + [c for c in df.columns if c not in front]]
    df = df.sort_values("participant").reset_index(drop=True)

    df.to_csv(seed, index=False, encoding="utf-8")
    print(f"Combinado -> {seed}  ({len(df)} participantes, {df.shape[1]} columnas)")
    for dx in LABEL_COLS:
        print(f"  {dx}: {df[dx].value_counts(dropna=False).to_dict()}")


if __name__ == "__main__":
    main()
