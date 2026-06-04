"""Carga de etiquetas clinicas desde el Excel de RESULTADOS.

Las columnas reales utiles son:
  - col 0: CODIGO        (1..80, == prefijo de carpeta de muestra)
  - col 1: DOCUMENTO     (cedula, == sufijo de carpeta de muestra)
  - 'DX DEPRESION IA'    (texto: 'Sin Riesgo...' / 'Riesgo Leve...' / 'Moderado...')
  - 'DX ANSIEDAD IA'     (idem para ansiedad)

Los items PHQ-9 / GAD-7 vienen vacios (0) en este archivo, por eso usamos las
columnas DX como verdad de terreno. Tarea = clasificacion BINARIA de riesgo:
  'sin riesgo' -> 0 ;  cualquier nivel de riesgo (leve/moderado/alto) -> 1
"""
import unicodedata
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def _norm(s) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()


def _col_por_patron(df, patrones):
    for c in df.columns:
        n = _norm(str(c)).replace("�", "")
        if all(p in n for p in patrones):
            return c
    return None


def _a_binario(texto) -> int | None:
    n = _norm(texto)
    if not n:
        return None
    if "sin riesgo" in n:
        return 0
    # 'riesgo leve/moderado/alto' o el typo 'riego leve'
    if "riesgo" in n or "riego" in n:
        return 1
    return None


def cargar_labels(ruta=None) -> pd.DataFrame:
    """Devuelve DataFrame: codigo, documento, ansiedad, depresion (0/1, NaN si falta)."""
    ruta = ruta or cfg.RUTA_LABELS
    df = pd.read_excel(ruta)
    col_cod = df.columns[0]
    col_doc = df.columns[1]
    col_dep = _col_por_patron(df, ["dx", "depresi"])
    col_anx = _col_por_patron(df, ["dx", "ansiedad"]) or _col_por_patron(df, ["dx", "dansiedad"])
    if col_dep is None or col_anx is None:
        raise ValueError(f"No se hallaron columnas DX. Columnas: {list(df.columns)}")

    out = pd.DataFrame({
        "codigo": pd.to_numeric(df[col_cod], errors="coerce").astype("Int64"),
        "documento": pd.to_numeric(df[col_doc], errors="coerce").astype("Int64"),
        "depresion": df[col_dep].map(_a_binario).astype("Int64"),
        "ansiedad": df[col_anx].map(_a_binario).astype("Int64"),
    })
    # La llave fiable es CODIGO (numero de participante 1..80), presente tanto en el
    # nombre de la carpeta como en el Excel. NO se usa la cedula/documento para cruzar:
    # 11 participantes tienen la cedula del nombre de carpeta distinta a la del Excel
    # (errores de digitacion), y cruzar por cedula perdia esas etiquetas.
    out = out.dropna(subset=["codigo"]).drop_duplicates(subset="codigo", keep="first")
    return out


if __name__ == "__main__":
    lab = cargar_labels()
    print(f"Filas: {len(lab)}")
    for dx in ("ansiedad", "depresion"):
        vc = lab[dx].value_counts(dropna=False).to_dict()
        print(f"  {dx}: {vc}")
    print(lab.head(8).to_string(index=False))
