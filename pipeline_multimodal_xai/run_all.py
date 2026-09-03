"""Orquesta el pipeline multimodal completo en orden.

    python pipeline_multimodal_xai/run_all.py

Equivale a correr, en secuencia:
    data_multimodal.py          (chequeo de carga)
    eval_multimodal.py          (baselines + nested CV -> best_config_<dx>.json, metricas_*)
    permutation_multimodal.py   (test de permutacion sobre el ganador)
    sensibilidad_agg.py         (sensibilidad a la agregacion de audio: mean/median/p80)
    xai_shap.py                 (SHAP: modalidad + intra-modalidad + familias + early)
    xai_lime.py                 (LIME local + consistencia SHAP/LIME)
    xai_stability.py            (estabilidad de features y del peso de modalidad)
"""
import runpy
import sys
import time

PASOS = [
    "data_multimodal",
    "eval_multimodal",
    "permutation_multimodal",
    "sensibilidad_agg",
    "xai_shap",
    "xai_lime",
    "xai_stability",
]


def main():
    solo = sys.argv[1:] or PASOS
    for mod in solo:
        print("\n" + "=" * 70 + f"\n>>> {mod}\n" + "=" * 70)
        t = time.time()
        argv0 = sys.argv
        sys.argv = [f"{mod}.py"]  # cada modulo hace su propio argparse
        try:
            runpy.run_module(mod, run_name="__main__")
        except SystemExit:
            pass
        finally:
            sys.argv = argv0
        print(f"<<< {mod}  ({time.time()-t:.0f}s)")


if __name__ == "__main__":
    main()
