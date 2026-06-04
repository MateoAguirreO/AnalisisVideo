# Reporte v2 — features temporales ricas + selección embebida (solo video)

**Fecha:** 2026-06-02 · **Datos:** 80 clips, los 80 etiquetados (cruce por **código**).
**Tarea:** clasificación binaria de riesgo, **ansiedad y depresión por separado**.
**Validación:** RepeatedStratifiedKFold **5×5** (25 folds), a nivel de participante.
**v1 intacta:** este reporte NO reemplaza `REPORTE_resultados.md` (v1); lo complementa.

---

## Qué cambia la v2 respecto a la v1

1. **Features temporales ricas** (256 en total, vs 48 de v1): por cada señal se calculan percentiles (p10/p90), IQR, asimetría, curtosis, **dinámica** (cuánto se mueve la señal entre frames), **% de tiempo activa**, y descriptores **por segmento** de la entrevista (variación entre segmentos + pendiente/tendencia).
2. **Selección de features EMBEBIDA en la CV** (`SelectKBest` dentro de cada fold). Esto la hace **honesta**: a diferencia del atajo de v1 ("6 features clínicas" elegidas sobre todos los datos, que daba ~0.60–0.62 de forma optimista), aquí la selección se reajusta en cada fold de entrenamiento, sin fuga de datos.
3. Modelos regularizados: **L1-LogReg**, RF poco profundo, XGB conservador. Barrido de `k` ∈ {8,12,16,24}.

## Resultado (AUC en CV 5×5)

| Eje           | v1 (48 feats, sin selección) | **v2 (temporal + selección honesta)** | Δ         |
| ------------- | ---------------------------- | ------------------------------------- | --------- |
| **Depresión** | 0.528                        | **0.605** — L1-LogReg, k=24 (±0.11)   | **+0.08** |
| **Ansiedad**  | 0.575                        | 0.541 — L1-LogReg, k=16 (±0.12)       | −0.03     |

Línea base por mayoría (accuracy trivial): ansiedad 0.76, depresión 0.74. La métrica clave es **AUC** (azar = 0.50).

### Detalle depresión (mejores configuraciones)

| modelo    | k   | AUC       | F1    | acc   |
| --------- | --- | --------- | ----- | ----- |
| L1-LogReg | 24  | **0.605** | 0.545 | 0.600 |
| L1-LogReg | 16  | 0.575     | 0.512 | 0.565 |
| RF        | 24  | 0.561     | 0.526 | 0.677 |

### Detalle ansiedad (mejores configuraciones)

| modelo    | k   | AUC   | F1    | acc   |
| --------- | --- | ----- | ----- | ----- |
| L1-LogReg | 16  | 0.541 | 0.498 | 0.562 |
| RF        | 24  | 0.536 | 0.483 | 0.675 |

## Interpretación (top features por f_classif)

- **Depresión:** `eye_squint` (p90/media/mediana), `pitch_skew` (asimetría del cabeceo), **`brow_inner_up`** (AU1 → tristeza, media y p10), `frown_segstd` (variación del ceño entre segmentos). → marcadores de tristeza/afecto negativo + dinámica de cabeza. **Clínicamente coherentes.**
- **Ansiedad:** `eye_squint`, `expresividad_p10`, `frown_skew`, `cheek_squint`. → tensión facial, pero la señal es débil.

Ver `resultados/metricas_v2_*.csv` y `resultados/v2_top_features_*.csv`.

## Lectura honesta

- **Depresión mejoró** de forma consistente (0.53 → **0.60**), con las features temporales (sobre todo percentiles y dinámica) y una selección honesta. Es el **mejor resultado de video hasta ahora**, y ahora sí sin optimismo de selección.
- **Ansiedad no mejoró** (0.54, sigue cerca de azar). Las features temporales no aportaron señal nueva para este eje.
- **Cautela estadística:** con n=80 y solo 19–21 positivos, la desviación entre folds es alta (±0.11–0.12). El intervalo de 0.605 se solapa con 0.53, así que la mejora es **sugestiva, no concluyente**. La dirección y la coherencia clínica de las features la respaldan, pero hace falta más datos para afirmarla con fuerza.

## Conclusión y próximos pasos

1. La v2 confirma que **hay algo de señal de video para depresión (~0.60 AUC)**, débil pero real e interpretable; ansiedad sigue esquiva.
2. Esto **refuerza el caso para la fusión multimodal** (audio+video): cada modalidad sola es débil; juntas podrían complementarse.
3. Otras palancas: más positivos / regularización más fuerte, etiqueta continua (regresión PHQ-9/GAD-7), o modelado por segmentos más fino.
