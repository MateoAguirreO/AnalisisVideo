"""Loop agentico de reflexion jr/senior sobre un caso de SHAP local.

Implementa el patron "Reflection" / variante Evaluator-Optimizer (generador y
critico son MODELOS DISTINTOS, no el mismo modelo autocriticandose), tal como
lo describe Shinn et al. 2023 (Reflexion: Language Agents with Verbal
Reinforcement Learning, NeurIPS 2023 -- PDF citable en
papers/shinn2023_reflexion.pdf, ver papers/_manifest_descargas.csv):

  1) "jr" (modelo mas barato/rapido, el Actor) redacta una lectura inicial del
     caso: chequea consistencia numerica (aditividad de SHAP) y da una lectura
     clinica de las top features, anclada a un prior fijo (no inventa).
  2) "senior" (modelo mas capaz, el Evaluator) audita el borrador del jr
     contra el mismo JSON: marca numeros mal citados y lecturas clinicas no
     sostenidas por el prior, y aprueba o devuelve feedback.
  3) si el senior no aprueba, el jr reescribe con TODO el historial de
     feedback acumulado hasta ahora (memoria episodica, no solo la ultima
     ronda -- asi el jr no repite un error ya senalado en una ronda previa),
     hasta N_RONDAS_MAX rondas.

Terminacion (nunca un bucle sin tope -- es el anti-patron, no el objetivo):
aprobacion del senior, tope duro de N_RONDAS_MAX, o deteccion de estancamiento
(si dos rondas seguidas del senior senalan exactamente los mismos problemas,
se corta -- no esta convergiendo, seguir gastando llamadas no ayuda).

Requiere `pip install google-genai` y `GEMINI_API_KEY` en el entorno (consola de
Google AI Studio: https://aistudio.google.com/apikey).

Uso:
  python pipeline_multimodal_xai/xai_reflexion.py --dx ansiedad --caso TP
  python pipeline_multimodal_xai/xai_reflexion.py --dx depresion --caso TN

Nota sobre los modelos: la key usada para probar esto NO tenia acceso a
modelos Pro (429 RESOURCE_EXHAUSTED por plan/billing, no por rate limit --
confirmado con gemini-pro-latest y gemini-3.1-pro-preview). Mientras esa key
no tenga billing habilitado, jr/senior son dos modelos Flash (mismo tier,
Flash-Lite vs Flash normal) en vez de Flash vs Pro. Si mas adelante activas
billing en el proyecto de Google AI Studio, cambia MODEL_SENIOR a
"gemini-pro-latest" para recuperar la asimetria real jr/senior. Verifica
disponibilidad en https://ai.google.dev/gemini-api/docs/models -- son la
unica linea que hay que tocar si un modelo deja de existir.
"""
import argparse
import json
import os
import re
import time

import config_mm as cfg

MODEL_JR = "gemini-3.1-flash-lite"  # borrador: el mas barato/rapido disponible
MODEL_SENIOR = "gemini-3.6-flash"   # auditoria: Flash normal (sin acceso a Pro con esta key)
MAX_OUTPUT_TOKENS = 8000            # gemini-3.x razona antes de responder (thinking
                                     # tokens cuentan contra este limite); con 4000 se
                                     # trunca a veces antes de emitir el JSON completo
N_RONDAS_MAX = 4  # ~Self-Refine/Reflexion usan 3-4; con estancamiento cortamos antes igual

# Prior clinico ya establecido en REPORTE_multimodal.md SS6 (SHAP intra-modalidad).
# El LLM audita CONTRA esto, no inventa relaciones nuevas.
PRIOR_CLINICO = """
Prior clinico de la tesis (REPORTE_multimodal.md SS6). Usalo para chequear
plausibilidad; cualquier lectura fuera de esto es especulativa y debe marcarse:
- Voz, ansiedad: logRelF0-H1-H2 / logRelF0-H1-A3 (fonacion tensa vs soplada) +
  percentiles de loudness (energia vocal) -> tension laringea / control de intensidad.
- Voz, depresion: amplitudes de formantes relativas a F0 (FxamplitudeLogRelF0) +
  balance espectral (alphaRatio, hammarbergIndex) -> articulacion reducida, voz apagada.
- Rostro, ansiedad: variabilidad de AU15 (depresor comisura labial) y AU20
  (estirador labios) + AU01/AU04/AU05 (tension frontal/parpado) -> tension perioral y frontal.
- Rostro, depresion: prototipo 'anger' + AU01 (elevador interno de ceja) -> afecto
  negativo facial / ceno, patron clasico de tristeza en la literatura.
""".strip()

SYSTEM_JR = f"""Eres un analista JUNIOR de explicabilidad (XAI) para un modelo de
riesgo de ansiedad/depresion (fusion audio+video, tesis de maestria). Se te da el
JSON de un caso (SHAP local: base_value, prediccion, top features con su valor
SHAP). Tu tarea:

1. CONSISTENCIA NUMERICA: revisa que 'suma_shap_total_mas_base' se aproxime a
   'p_pred_full' (aditividad de SHAP sobre las 506 features, no solo el top-K
   listado). Senala cualquier discrepancia grande entre 'p_oof' y 'p_pred_full'
   (indica que el caso no es un acierto limpio fuera de muestra).
2. LECTURA CLINICA: para las top features, da una lectura breve usando SOLO el
   prior clinico de abajo. Si una feature no esta en ese prior, dilo
   explicitamente como 'sin prior establecido' en vez de inventar una
   explicacion.

{PRIOR_CLINICO}

Responde EXCLUSIVAMENTE con un JSON (sin texto alrededor, sin markdown):
{{
  "chequeo_numerico": {{"consistente": bool, "nota": "..."}},
  "lectura_clinica": [{{"feature": "...", "lectura": "...", "con_prior": bool}}],
  "resumen": "..."
}}"""

SYSTEM_SENIOR = f"""Eres un analista SENIOR de explicabilidad (XAI), auditando el
borrador de un analista junior sobre un caso de SHAP local (modelo de riesgo de
ansiedad/depresion, fusion audio+video). Se te da el JSON original del caso y el
borrador del junior. Tu tarea:

1. Verifica cada afirmacion numerica del junior contra el JSON original (no le
   des el beneficio de la duda a su aritmetica).
2. Verifica que cada lectura clinica este realmente respaldada por el prior de
   abajo; marca como no sostenida cualquier lectura que el junior presente como
   firme sin estarlo (incluye las que el junior ya marco 'con_prior: true' pero
   en realidad no aplican).
3. Si se te da un historial de rondas previas, confirma que el junior
   REALMENTE corrigio lo que se le senalo -- no repitas un error de la lista
   como nuevo si ya fue corregido, pero tampoco apruebes si sigue presente.
4. Aprueba SOLO si no hay errores numericos y ninguna lectura clinica se
   presenta como mas firme de lo que el prior sostiene.

{PRIOR_CLINICO}

Responde EXCLUSIVAMENTE con un JSON (sin texto alrededor, sin markdown):
{{
  "aprobado": bool,
  "errores_numericos": ["..."],
  "lecturas_no_sostenidas": ["..."],
  "feedback_para_junior": "...",
  "version_final": {{
    "chequeo_numerico": {{"consistente": bool, "nota": "..."}},
    "lectura_clinica": [{{"feature": "...", "lectura": "...", "con_prior": bool}}],
    "resumen": "..."
  }}
}}"""


def _cargar_caso(dx, caso):
    p = cfg.DIR_RESULTADOS / f"shap_local_{dx}_{caso}.json"
    if not p.exists():
        raise FileNotFoundError(f"falta {p}; corre xai_shap_local.py primero")
    return json.loads(p.read_text(encoding="utf-8"))


def _extraer_json(texto):
    """El LLM puede envolver el JSON en ```json ... ``` o con texto alrededor."""
    m = re.search(r"\{.*\}", texto, re.DOTALL)
    if not m:
        raise ValueError(f"no se encontro JSON en la respuesta:\n{texto[:500]}")
    return json.loads(m.group(0))


def _llamar(client, model, system, user, reintentos=4):
    from google.genai import errors, types
    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.2,  # bajo: queremos lectura ceñida al prior, no creativa
    )
    for intento in range(1, reintentos + 1):
        try:
            resp = client.models.generate_content(model=model, contents=user, config=config)
            return _extraer_json(resp.text)
        except errors.ServerError as e:
            # 503/500: "alta demanda", transitorio -- reintentar con backoff.
            # El SDK ya reintenta internamente unas pocas veces; esto cubre
            # rachas mas largas sin abortar todo el loop jr/senior.
            if intento == reintentos:
                raise
            espera = 2 ** intento
            print(f"  ({model} 503, reintento {intento}/{reintentos} en {espera}s: {e})")
            time.sleep(espera)


def _resumen_historial(historial):
    """Memoria episodica (Shinn et al. 2023 SS3): el jr/senior no ven solo la
    ultima critica, ven el log completo de lo ya senalado en rondas previas."""
    if not historial:
        return "(sin rondas previas)"
    bloques = []
    for h in historial:
        s = h["auditoria_senior"]
        bloques.append(
            f"-- Ronda {h['ronda']} --\n"
            f"  errores_numericos: {s.get('errores_numericos', [])}\n"
            f"  lecturas_no_sostenidas: {s.get('lecturas_no_sostenidas', [])}\n"
            f"  feedback: {s.get('feedback_para_junior', '')}"
        )
    return "\n".join(bloques)


def _sin_progreso(historial):
    """True si las ultimas 2 auditorias del senior senalan EXACTAMENTE los
    mismos problemas -> el loop no esta convergiendo, cortar en vez de seguir
    gastando llamadas (guardrail recomendado junto al tope de rondas)."""
    if len(historial) < 2:
        return False
    a, b = historial[-2]["auditoria_senior"], historial[-1]["auditoria_senior"]
    return (set(a.get("errores_numericos", [])) == set(b.get("errores_numericos", []))
            and set(a.get("lecturas_no_sostenidas", [])) == set(b.get("lecturas_no_sostenidas", [])))


def reflexion(dx, caso):
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("falta GEMINI_API_KEY en el entorno")
    client = genai.Client(api_key=api_key)
    payload = _cargar_caso(dx, caso)
    caso_json = json.dumps(payload, ensure_ascii=False, indent=2)

    jr_out = _llamar(client, MODEL_JR, SYSTEM_JR, f"JSON del caso:\n{caso_json}")
    print(f"[jr] resumen: {jr_out.get('resumen', '')[:200]}")

    historial, senior_out, ronda, estancado = [], {}, 0, False
    for ronda in range(1, N_RONDAS_MAX + 1):
        senior_in = (f"JSON del caso:\n{caso_json}\n\n"
                     f"Historial de rondas previas (verifica que lo ya senalado se haya "
                     f"corregido):\n{_resumen_historial(historial)}\n\n"
                     f"Borrador del junior (ronda {ronda}):\n"
                     f"{json.dumps(jr_out, ensure_ascii=False, indent=2)}")
        senior_out = _llamar(client, MODEL_SENIOR, SYSTEM_SENIOR, senior_in)
        print(f"[senior] ronda {ronda}: aprobado={senior_out.get('aprobado')}  "
              f"errores={len(senior_out.get('errores_numericos', []))}  "
              f"no_sostenidas={len(senior_out.get('lecturas_no_sostenidas', []))}")
        historial.append({"ronda": ronda, "borrador_jr": jr_out, "auditoria_senior": senior_out})

        if senior_out.get("aprobado") or ronda == N_RONDAS_MAX:
            break
        if _sin_progreso(historial):
            estancado = True
            print(f"  (sin progreso entre rondas {ronda - 1} y {ronda}, corto el bucle)")
            break

        jr_in = (f"JSON del caso:\n{caso_json}\n\n"
                 f"Historial COMPLETO de feedback recibido hasta ahora (no repitas ningun "
                 f"error ya senalado, no solo el de la ultima ronda):\n"
                 f"{_resumen_historial(historial)}\n\n"
                 f"Tu ultimo borrador fue RECHAZADO en la ronda {ronda}. Reescribe tu "
                 f"analisis corrigiendo TODOS los puntos senalados arriba.")
        jr_out = _llamar(client, MODEL_JR, SYSTEM_JR, jr_in)

    resultado = {
        "eje": dx, "caso": caso, "pid": payload["pid"],
        "rondas": ronda, "aprobado_final": senior_out.get("aprobado", False),
        "estancado": estancado,
        "version_final": senior_out.get("version_final", jr_out),
        "historial": historial,
    }
    out = cfg.DIR_RESULTADOS / f"reflexion_{dx}_{caso}.json"
    out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {out}")
    return resultado


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", required=True, choices=cfg.DXS)
    ap.add_argument("--caso", required=True, choices=["TP", "TN"])
    args = ap.parse_args()
    reflexion(args.dx, args.caso)


if __name__ == "__main__":
    main()
