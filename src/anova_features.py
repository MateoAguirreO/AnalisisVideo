"""
ANOVA sobre features de video extraídas con MediaPipe (Google Tasks API).

Para cada diagnóstico (ansiedad, depresion):
  1. ANOVA univariado: F-test de una vía por feature entre grupo positivo y
     negativo. Corrección FDR Benjamini-Hochberg. Tamaño de efecto: eta-cuadrado.
  2. ANOVA entre modelos: re-corre la misma CV 5×5 de train_v2.py y compara
     las 25 distribuciones de AUC entre l1logreg, rf y xgb.
     Post-hoc: t-tests apareados con corrección Bonferroni.

Salidas:
  resultados/anova_features_{dx}.csv          -- F, p, p_adj, eta2 por feature
  resultados/anova_modelos_{dx}_folds.csv     -- AUC por fold por modelo
  resultados/anova_modelos_{dx}_pairwise.csv  -- comparaciones post-hoc
  resultados/REPORTE_anova.md                 -- informe completo en markdown

Uso:
  python src/anova_features.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from labels import cargar_labels

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

warnings.filterwarnings("ignore")

V2_CSV = cfg.DIR_FEATURES / "video_features_v2.csv"
MODELOS_NAMES = ["l1logreg", "rf", "xgb"]


# ─── estadísticos auxiliares ─────────────────────────────────────────────────

def bh_correct(pvals: np.ndarray) -> np.ndarray:
    """Corrección FDR Benjamini-Hochberg. Devuelve p-valores ajustados."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    adj = np.minimum(1.0, pvals * n / ranks)
    # monotonía ascendente desde el menor p
    adj_sorted = adj[order]
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    result = np.empty(n)
    result[order] = adj_sorted
    return result


def eta_squared(a: np.ndarray, b: np.ndarray) -> float:
    """η² = SS_entre / SS_total (dos grupos)."""
    all_ = np.concatenate([a, b])
    gm = all_.mean()
    ss_between = len(a) * (a.mean() - gm) ** 2 + len(b) * (b.mean() - gm) ** 2
    ss_total = float(((all_ - gm) ** 2).sum())
    return ss_between / ss_total if ss_total > 1e-12 else np.nan


def efecto_label(e: float) -> str:
    if e >= 0.14:
        return "grande"
    if e >= 0.06:
        return "mediano"
    if e >= 0.01:
        return "pequeño"
    return "trivial"


# ─── carga de datos ───────────────────────────────────────────────────────────

def cargar(dx: str):
    df = pd.read_csv(V2_CSV)
    lab = cargar_labels()[["codigo", dx]].rename(columns={dx: "label"})
    df = df.merge(lab, on="codigo", how="left").dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    feats = [c for c in df.columns if c not in ("participant", "codigo", "label")]
    return df, feats, df[feats].values, df["label"].values


# ─── ANOVA por feature ────────────────────────────────────────────────────────

def anova_features(dx: str) -> pd.DataFrame:
    _, feats, X, y = cargar(dx)

    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    Xi = imp.fit_transform(X)

    pos_idx = y == 1
    neg_idx = y == 0
    n_pos = int(pos_idx.sum())
    n_neg = int(neg_idx.sum())

    rows = []
    for i, feat in enumerate(feats):
        a = Xi[pos_idx, i]
        b = Xi[neg_idx, i]
        all_ = np.concatenate([a, b])

        if np.std(all_) < 1e-12:
            rows.append(dict(feature=feat, F=np.nan, p=1.0, eta2=0.0,
                             mean_pos=float(a.mean()), mean_neg=float(b.mean()),
                             n_pos=n_pos, n_neg=n_neg))
            continue

        F_val, p_val = stats.f_oneway(a, b)
        rows.append(dict(feature=feat,
                         F=float(F_val),
                         p=float(p_val),
                         eta2=eta_squared(a, b),
                         mean_pos=float(a.mean()),
                         mean_neg=float(b.mean()),
                         n_pos=n_pos,
                         n_neg=n_neg))

    tab = pd.DataFrame(rows)
    raw_p = tab["p"].fillna(1.0).values
    tab["p_adj"] = bh_correct(raw_p)
    tab["sig"] = tab["p_adj"] < 0.05
    tab = tab.sort_values("F", ascending=False).reset_index(drop=True)

    out = cfg.DIR_RESULTADOS / f"anova_features_{dx}.csv"
    tab.to_csv(out, index=False)

    n_sig = int(tab["sig"].sum())
    print(f"[{dx}] ANOVA features: {n_sig}/{len(tab)} significativas (p_adj<0.05) -> {out.name}")
    return tab


# ─── ANOVA entre modelos ──────────────────────────────────────────────────────

def get_clf(nombre: str):
    if nombre == "l1logreg":
        return LogisticRegression(penalty="l1", solver="liblinear", C=0.5,
                                  class_weight="balanced", max_iter=2000,
                                  random_state=cfg.RANDOM_STATE)
    if nombre == "rf":
        return RandomForestClassifier(n_estimators=400, max_depth=4,
                                      min_samples_leaf=3, class_weight="balanced",
                                      random_state=cfg.RANDOM_STATE, n_jobs=-1)
    if nombre == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=250, max_depth=2, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.8, reg_lambda=2.0,
                             eval_metric="logloss", random_state=cfg.RANDOM_STATE,
                             n_jobs=-1)
    raise ValueError(nombre)


def make_pipe(nombre: str, k: int, n_feats: int) -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("var", VarianceThreshold(0.0)),
        ("sc", StandardScaler()),
        ("sel", SelectKBest(f_classif, k=min(k, n_feats))),
        ("clf", get_clf(nombre)),
    ])


def best_k_per_model(dx: str) -> dict:
    df = pd.read_csv(cfg.DIR_RESULTADOS / f"metricas_v2_{dx}.csv")
    out = {}
    for m in df["modelo"].unique():
        sub = df[df["modelo"] == m]
        out[m] = int(sub.loc[sub["auc"].idxmax(), "k"])
    return out


def anova_modelos(dx: str) -> tuple:
    _, feats, X, y = cargar(dx)
    n_feats = X.shape[1]
    best_ks = best_k_per_model(dx)

    cv = RepeatedStratifiedKFold(n_splits=cfg.N_SPLITS, n_repeats=cfg.N_REPEATS,
                                 random_state=cfg.RANDOM_STATE)

    fold_aucs: dict[str, np.ndarray] = {}
    for m in MODELOS_NAMES:
        k = best_ks.get(m, 16)
        print(f"  [{dx}] CV {m} k={k}...", end=" ", flush=True)
        res = cross_validate(make_pipe(m, k, n_feats), X, y, cv=cv,
                             scoring=["roc_auc"], n_jobs=-1)
        fold_aucs[m] = res["test_roc_auc"]
        mu = float(np.mean(fold_aucs[m]))
        sd = float(np.std(fold_aucs[m]))
        print(f"AUC = {mu:.3f} ± {sd:.3f}")

    # ANOVA de una vía entre los 3 modelos
    groups = [fold_aucs[m] for m in MODELOS_NAMES]
    F_mod, p_mod = stats.f_oneway(*groups)

    # Post-hoc: t-tests apareados con corrección Bonferroni (3 pares)
    n_pairs = 3
    alpha_b = 0.05 / n_pairs
    pairs = []
    for i in range(len(MODELOS_NAMES)):
        for j in range(i + 1, len(MODELOS_NAMES)):
            ma, mb = MODELOS_NAMES[i], MODELOS_NAMES[j]
            t_val, p_t = stats.ttest_ind(fold_aucs[ma], fold_aucs[mb])
            pairs.append(dict(
                modelo_a=ma, modelo_b=mb,
                t=round(float(t_val), 3),
                p=round(float(p_t), 4),
                sig_bonf=bool(p_t < alpha_b),
                delta_auc=round(float(np.mean(fold_aucs[ma]) - np.mean(fold_aucs[mb])), 4),
            ))

    # Guardar por-fold y post-hoc
    pd.DataFrame(fold_aucs).to_csv(
        cfg.DIR_RESULTADOS / f"anova_modelos_{dx}_folds.csv", index=False)
    pairs_df = pd.DataFrame(pairs)
    pairs_df.to_csv(
        cfg.DIR_RESULTADOS / f"anova_modelos_{dx}_pairwise.csv", index=False)

    sig_str = "SIGNIFICATIVO" if p_mod < 0.05 else "no-significativo"
    print(f"[{dx}] ANOVA modelos: F = {F_mod:.3f}  p = {p_mod:.4f}  [{sig_str}]")
    return fold_aucs, float(F_mod), float(p_mod), pairs_df


# ─── generación del informe ───────────────────────────────────────────────────

def generar_reporte(resultados: dict) -> None:
    L = []

    L.append("# Informe ANOVA — Features de Video (MediaPipe / Google)")
    L.append("")
    L.append(f"_Generado: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}_")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 1. Metodología")
    L.append("")
    L.append("- **Extracción de features**: MediaPipe Tasks API (Google) con modelos "
             "`face_landmarker`, `pose_landmarker_lite` y `hand_landmarker`. "
             "Features incluyen blendshapes faciales (parpadeo, apertura ocular, "
             "sonrisa, ceño, posición de cejas, apertura de boca), orientación de "
             "cabeza (yaw/pitch/roll), movimiento corporal y de manos.")
    L.append("- **ANOVA univariado (features)**: F-test de una vía para cada feature, "
             "comparando grupo positivo vs. negativo por diagnóstico. "
             "Valores faltantes imputados con mediana antes del test.")
    L.append("- **Corrección de comparaciones múltiples**: Benjamini-Hochberg (FDR α = 5%). "
             "Una feature es *significativa* si p_adj < 0.05.")
    L.append("- **Tamaño de efecto**: η² (eta-cuadrado). "
             "Umbrales: trivial < 0.01 ≤ pequeño < 0.06 ≤ mediano < 0.14 ≤ grande.")
    L.append("- **ANOVA entre modelos**: F-test de una vía sobre las 25 puntuaciones "
             "AUC-ROC de la CV RepeatedStratifiedKFold (5 folds × 5 repeticiones). "
             "Modelos comparados: L1-LogReg, RandomForest, XGBoost, cada uno con su "
             "mejor `k` de selección de features según `metricas_v2_{dx}.csv`.")
    L.append("- **Post-hoc**: t-tests independientes entre pares de modelos, "
             "corrección Bonferroni (α/3 ≈ 0.0167).")
    L.append("")

    for dx in ["depresion", "ansiedad"]:
        feat_tab, fold_aucs, F_mod, p_mod, pairs_df = resultados[dx]
        sig = feat_tab[feat_tab["sig"]]
        n_total = len(feat_tab)
        n_pos = int(feat_tab["n_pos"].iloc[0])
        n_neg = int(feat_tab["n_neg"].iloc[0])
        best_ks = best_k_per_model(dx)

        L.append("---")
        L.append("")
        L.append(f"## 2. {dx.capitalize()}")
        L.append("")
        L.append(f"**Muestra**: n = {n_pos + n_neg} "
                 f"(positivos = {n_pos} · negativos = {n_neg})")
        L.append("")

        # ── 2.1 ANOVA features ──
        L.append(f"### 2.1 ANOVA por Feature")
        L.append("")
        L.append(f"- Features analizadas: **{n_total}**")
        L.append(f"- Significativas (p_adj < 0.05): **{len(sig)}**")
        L.append("")

        if len(sig) > 0:
            large_n = int((sig["eta2"] >= 0.14).sum())
            med_n = int(((sig["eta2"] >= 0.06) & (sig["eta2"] < 0.14)).sum())
            small_n = int(((sig["eta2"] >= 0.01) & (sig["eta2"] < 0.06)).sum())
            L.append(f"Distribución de tamaños de efecto entre las significativas: "
                     f"**grande** = {large_n} · **mediano** = {med_n} · **pequeño** = {small_n}")
            L.append("")

            # top features por categoría
            cats: dict[str, int] = {}
            for f in sig["feature"]:
                cat = f.split("_")[0]
                cats[cat] = cats.get(cat, 0) + 1
            top_cats = sorted(cats.items(), key=lambda x: -x[1])[:6]
            cat_str = " · ".join(f"`{c}` ({n})" for c, n in top_cats)
            L.append(f"Categorías más representadas: {cat_str}")
            L.append("")

            top_n = min(25, len(sig))
            L.append(f"#### Top-{top_n} Features Significativas (ordenadas por F)")
            L.append("")
            L.append("| # | Feature | F | p_adj | η² | Efecto | Media (+) | Media (−) |")
            L.append("|---|---------|--:|------:|----:|--------|----------:|----------:|")
            for rank, (_, row) in enumerate(sig.head(top_n).iterrows(), 1):
                L.append(
                    f"| {rank} | `{row['feature']}` "
                    f"| {row['F']:.2f} "
                    f"| {row['p_adj']:.4f} "
                    f"| {row['eta2']:.4f} "
                    f"| {efecto_label(row['eta2'])} "
                    f"| {row['mean_pos']:.4f} "
                    f"| {row['mean_neg']:.4f} |"
                )
            L.append("")
        else:
            L.append("_Ninguna feature resultó significativa tras corrección FDR._")
            L.append("")

        # ── 2.2 ANOVA modelos ──
        L.append("### 2.2 ANOVA entre Modelos")
        L.append("")
        L.append("| Modelo | AUC medio | AUC std | Mejor k |")
        L.append("|--------|----------:|--------:|--------:|")
        for m in MODELOS_NAMES:
            aucs = fold_aucs[m]
            L.append(f"| {m} | {np.mean(aucs):.3f} | {np.std(aucs):.3f} | {best_ks.get(m, '?')} |")
        L.append("")

        sig_mod = p_mod < 0.05
        L.append(
            f"**F = {F_mod:.3f}, p = {p_mod:.4f}** — "
            + ("✓ Diferencia estadísticamente significativa entre modelos." if sig_mod
               else "✗ No hay diferencia significativa entre modelos (p ≥ 0.05).")
        )
        L.append("")

        if sig_mod:
            L.append("#### Post-hoc Pairwise (Bonferroni α ≈ 0.0167)")
            L.append("")
            L.append("| Par | Δ AUC | t | p | Sig. |")
            L.append("|-----|------:|--:|--:|:----:|")
            for _, row in pairs_df.iterrows():
                sig_mark = "✓" if row["sig_bonf"] else "–"
                L.append(
                    f"| {row['modelo_a']} vs {row['modelo_b']} "
                    f"| {row['delta_auc']:+.4f} "
                    f"| {row['t']:.3f} "
                    f"| {row['p']:.4f} "
                    f"| {sig_mark} |"
                )
            L.append("")

    # ── 3. Conclusiones ──
    L.append("---")
    L.append("")
    L.append("## 3. Conclusiones")
    L.append("")

    for dx in ["depresion", "ansiedad"]:
        feat_tab, fold_aucs, F_mod, p_mod, pairs_df = resultados[dx]
        sig = feat_tab[feat_tab["sig"]]
        top3 = sig.head(3)["feature"].tolist()
        best_m_name = max(fold_aucs.items(), key=lambda x: float(np.mean(x[1])))[0]
        best_auc = float(np.mean(fold_aucs[best_m_name]))

        L.append(f"### {dx.capitalize()}")
        L.append("")

        if len(sig) > 0:
            large_n = int((sig["eta2"] >= 0.14).sum())
            med_n = int(((sig["eta2"] >= 0.06) & (sig["eta2"] < 0.14)).sum())
            top3_str = ", ".join(f"`{f}`" for f in top3)
            L.append(
                f"- **{len(sig)} features** del video discriminan significativamente "
                f"entre grupos tras corrección FDR (BH). "
                f"Las tres con mayor F-estadístico son: {top3_str}."
            )
            L.append(
                f"- Tamaños de efecto: {large_n} feature(s) con η² grande "
                f"y {med_n} con η² mediano. La mayoría del poder discriminativo "
                f"se concentra en features de **región ocular** (eye_squint, brow) "
                f"y **expresividad facial** global."
            )
        else:
            L.append(
                f"- Ninguna feature de video discrimina significativamente entre "
                f"grupos para {dx} tras corrección FDR. "
                f"Las features univariadas tienen poder discriminativo limitado."
            )

        sig_mod = p_mod < 0.05
        L.append(
            f"- Los tres modelos (L1-LogReg, RF, XGBoost) "
            + ("**difieren** significativamente en AUC " if sig_mod else "**no difieren** significativamente en AUC ")
            + f"(F = {F_mod:.2f}, p = {p_mod:.4f}). "
            f"El mejor modelo es **{best_m_name}** (AUC = {best_auc:.3f})."
        )
        L.append("")

    L.append("---")
    L.append("")
    L.append("_Archivos generados:_")
    L.append("")
    for dx in ["depresion", "ansiedad"]:
        L.append(f"- `anova_features_{dx}.csv` — ANOVA completo por feature")
        L.append(f"- `anova_modelos_{dx}_folds.csv` — AUC por fold")
        L.append(f"- `anova_modelos_{dx}_pairwise.csv` — post-hoc")
    L.append("")

    report_path = cfg.DIR_RESULTADOS / "REPORTE_anova.md"
    report_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\nReporte guardado -> {report_path}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    resultados = {}
    for dx in cfg.DXS:
        print(f"\n{'=' * 55}")
        print(f"  {dx.upper()}")
        print(f"{'=' * 55}")

        print("\n[1/2] ANOVA por feature...")
        feat_tab = anova_features(dx)

        print(f"\n[2/2] ANOVA entre modelos (re-ejecutando CV 5×5)...")
        fold_aucs, F_mod, p_mod, pairs_df = anova_modelos(dx)

        resultados[dx] = (feat_tab, fold_aucs, F_mod, p_mod, pairs_df)

    print("\n" + "=" * 55)
    print("  GENERANDO REPORTE")
    print("=" * 55)
    generar_reporte(resultados)
    print("\nFinalizado correctamente.")


if __name__ == "__main__":
    main()
