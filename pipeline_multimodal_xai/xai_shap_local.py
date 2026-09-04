"""SHAP local (waterfall) para 2 muestras especificas por eje + export JSON.

Genera, para UNA muestra positiva (TP: y=1, acertada) y UNA negativa (TN: y=0,
acertada) por eje, un waterfall plot y un JSON con los SHAP values ordenados.
Reusa EXACTAMENTE la seleccion de casos de xai_lime.py (mismo pipeline de early
fusion, mismo RANDOM_STATE) para que los pid coincidan con los ya reportados en
REPORTE_multimodal.md SS8 (ansiedad TP=61/TN=54, depresion TP=35/TN=9).

El JSON (`shap_local_<dx>_<caso>.json`) es el insumo del loop de verificacion
jr/senior (xai_reflexion.py): values + base_value + top features con nombre,
modalidad y familia, listos para que un LLM los audite sin tener que parsear CSVs.

Uso:  python pipeline_multimodal_xai/xai_shap_local.py   (requiere best_config_<dx>.json)
"""
import json
import warnings

import numpy as np

import config_mm as cfg
from data_multimodal import cargar
from fusion import matriz_early_fusion
from xai_lime import _pipe_early, _casos

warnings.filterwarnings("ignore")
RS = cfg.RANDOM_STATE
CASOS_A_EXPORTAR = ("TP", "TN")   # muestra positiva y negativa (ver xai_lime._casos)
TOP_K_JSON = 20                   # features con mayor |SHAP| que se listan en el JSON


def _familia(feature):
    fams = cfg.FAMILIAS_AUDIO if feature.startswith("aud__") else cfg.FAMILIAS_VIDEO
    return cfg.familia_de(feature, fams)


def shap_local_caso(dx, nombre, idx, pipe, X, y, oof, feats, pids, bg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    shap_expl = shap.KernelExplainer(lambda z: pipe.predict_proba(z)[:, 1], bg)
    sv = np.asarray(shap_expl.shap_values(X[idx:idx + 1], nsamples=200)).ravel()
    ev = shap_expl.expected_value
    base_value = float(ev if np.isscalar(ev) else np.asarray(ev).ravel()[0])
    p_full = float(pipe.predict_proba(X[idx:idx + 1])[0, 1])

    exp = shap.Explanation(values=sv, base_values=base_value,
                            data=X[idx], feature_names=feats)
    shap.plots.waterfall(exp, max_display=15, show=False)
    plt.title(f"{dx} — SHAP local caso {nombre} (pid {pids[idx]}, y={y[idx]})", fontsize=9)
    plt.tight_layout()
    png = cfg.DIR_RESULTADOS / f"shap_local_{dx}_{nombre}.png"
    plt.savefig(png, dpi=130, bbox_inches="tight")
    plt.close()

    orden = np.argsort(-np.abs(sv))[:TOP_K_JSON]
    top = [{
        "feature": feats[i],
        "modalidad": "audio" if feats[i].startswith("aud__") else "video",
        "familia": _familia(feats[i]),
        "valor_feature": float(X[idx, i]),
        "shap_value": round(float(sv[i]), 5),
    } for i in orden]

    payload = {
        "eje": dx,
        "caso": nombre,
        "pid": int(pids[idx]),
        "y_true": int(y[idx]),
        "prediccion_clase": "riesgo" if p_full >= 0.5 else "sin riesgo",
        "p_oof": round(float(oof[idx]), 4),
        "p_pred_full": round(p_full, 4),
        "base_value": round(base_value, 4),
        # chequeo de consistencia numerica: base + suma(shap de TODAS las features,
        # no solo el top-K) debe aproximar p_pred_full (aditividad de SHAP).
        "suma_shap_total_mas_base": round(base_value + float(sv.sum()), 4),
        "n_features_total": len(feats),
        "top_features": top,
    }
    out = cfg.DIR_RESULTADOS / f"shap_local_{dx}_{nombre}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {nombre} (pid {pids[idx]}, y={y[idx]}): p={p_full:.3f}  base={base_value:.3f}  "
          f"top1={top[0]['feature']} ({top[0]['shap_value']:+.3f})  -> {png.name}, {out.name}")


def main():
    import shap
    for dx in cfg.DXS:
        pj = cfg.DIR_RESULTADOS / f"best_config_{dx}.json"
        if not pj.exists():
            print(f"[{dx}] falta {pj.name}; corre eval_multimodal.py primero"); continue
        best = json.loads(pj.read_text(encoding="utf-8"))
        vid, aud = cargar(dx)
        Me = matriz_early_fusion(vid, aud)
        pipe, X, y, feats = _pipe_early(Me, best["early_config"])
        casos, oof = _casos(pipe, Me, best["early_config"], X, y)
        pids = list(vid.index)
        bg = shap.sample(X, 40, random_state=RS)
        print(f"\n=== {dx.upper()} === early_config={best['early_config']}")
        for nombre in CASOS_A_EXPORTAR:
            shap_local_caso(dx, nombre, casos[nombre], pipe, X, y, oof, feats, pids, bg)


if __name__ == "__main__":
    main()
