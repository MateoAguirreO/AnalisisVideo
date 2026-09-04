"""Entrena y congela los modelos finales (voz + rostro) por eje, sobre los 79
participantes completos, y los guarda en disco para que el microservicio
(servicio/app.py) los sirva sin reentrenar en cada request.

Usa la config ganadora de resultados/best_config_<dx>.json (hoy: soft_vote en
ambos ejes -- ver REPORTE_multimodal.md SS3). Si algun dia el ganador deja de
ser soft_vote (por ejemplo al adaptar los modelos de 0.8 AUC), este script
debe extenderse para tambien persistir el meta-modelo entrenado.

Guarda, por eje:
  modelos/<dx>_audio.joblib     -- pipeline sklearn de segmento (voz)
  modelos/<dx>_video.joblib     -- pipeline sklearn de participante (rostro)
  modelos/<dx>_audio_bg.joblib  -- muestra de background (features CRUDAS,
                                    antes del pipeline) para SHAP de una
                                    muestra nueva en el microservicio de
                                    PlataformaMultimodal
  modelos/<dx>_video_bg.joblib  -- idem, rostro (los 79 participantes)
  modelos/<dx>_meta.json        -- orden EXACTO de features que espera cada
                                    pipeline (el orden importa: los arrays de
                                    sklearn son posicionales) + config + AUC
                                    esperado, para validar en produccion.

Uso:  python pipeline_multimodal_xai/train_final.py
"""
import json

import joblib
import numpy as np

import config_mm as cfg
from audio_branch import RamaAudio
from data_multimodal import cargar
from fusion import construir_video_config

DIR_MODELOS = cfg.ROOT / "modelos"
N_BG_AUDIO = 60  # tamano del background de segmentos para SHAP (KernelExplainer/LinearExplainer)


def _entrenar_video(vid, video_cfg):
    from sklearn.model_selection import GridSearchCV
    feats = [c for c in vid.columns if c != "label"]
    X, y = vid[feats].values, vid["label"].values
    pipe, grid = construir_video_config(video_cfg)
    if grid:
        pipe = GridSearchCV(pipe, grid, cv=5, scoring="roc_auc", n_jobs=-1).fit(X, y).best_estimator_
    else:
        pipe.fit(X, y)
    return pipe, feats


def main():
    DIR_MODELOS.mkdir(exist_ok=True)
    for dx in cfg.DXS:
        pj = cfg.DIR_RESULTADOS / f"best_config_{dx}.json"
        if not pj.exists():
            print(f"[{dx}] falta {pj.name}; corre eval_multimodal.py primero"); continue
        best = json.loads(pj.read_text(encoding="utf-8"))
        if best["ganador"] != "soft_vote":
            raise NotImplementedError(
                f"[{dx}] ganador='{best['ganador']}' -- train_final.py solo sabe "
                f"congelar soft_vote por ahora (es el ganador actual en ambos ejes). "
                f"Si esto cambio, hay que persistir tambien el meta-modelo entrenado.")

        vid, aud = cargar(dx)
        rama_audio = RamaAudio(config=best["audio_config"], agg=cfg.AUDIO_PROB_AGG).fit(aud)
        pipe_video, feats_video = _entrenar_video(vid, best["video_config"])

        joblib.dump(rama_audio.estimator_, DIR_MODELOS / f"{dx}_audio.joblib")
        joblib.dump(pipe_video, DIR_MODELOS / f"{dx}_video.joblib")

        rng = np.random.RandomState(cfg.RANDOM_STATE)
        idx_bg = rng.choice(len(aud), size=min(N_BG_AUDIO, len(aud)), replace=False)
        bg_audio = aud[rama_audio.feats_].values[idx_bg]
        bg_video = vid[feats_video].values
        joblib.dump(bg_audio, DIR_MODELOS / f"{dx}_audio_bg.joblib")
        joblib.dump(bg_video, DIR_MODELOS / f"{dx}_video_bg.joblib")

        meta = {
            "dx": dx,
            "audio_config": best["audio_config"],
            "video_config": best["video_config"],
            "fusion": "soft_vote",
            "audio_agg": cfg.AUDIO_PROB_AGG,
            "audio_features": rama_audio.feats_,   # orden exacto esperado por el pipeline
            "video_features": feats_video,          # idem
            "auc_esperado": best["auc_ganador"],
            "n_train": {"audio_segmentos": len(aud), "video_participantes": len(vid)},
        }
        (DIR_MODELOS / f"{dx}_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{dx}] audio={best['audio_config']} ({len(rama_audio.feats_)} feats)  "
              f"video={best['video_config']} ({len(feats_video)} feats)  "
              f"fusion=soft_vote  auc_esperado={best['auc_ganador']}  bg_audio={len(bg_audio)}  "
              f"bg_video={len(bg_video)}  -> modelos/{dx}_{{audio,video,audio_bg,video_bg,meta}}")


if __name__ == "__main__":
    main()
