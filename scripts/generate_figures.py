# scripts/generate_figures.py
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (confusion_matrix, precision_recall_curve,
                              average_precision_score)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─── CHEMINS ──────────────────────────────────────────────────────────────────
BASE     = Path(r"C:\Users\racim\Desktop\ENSIMAG 1A\projet_perso\pswrd_spray")
UNSW_DIR = BASE / "data/public/UNS"
RES_DIR  = BASE / "data/results"
FIG_DIR  = BASE / "reports/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
}
plt.rcParams.update(STYLE)

# ─── COULEURS ─────────────────────────────────────────────────────────────────
C_BLUE  = "#2980b9"
C_RED   = "#e74c3c"
C_GREEN = "#27ae60"
C_GRAY  = "#95a5a6"

# ─── CHARGEMENT MODÈLE + DONNÉES ──────────────────────────────────────────────
obj      = joblib.load(RES_DIR / "model_final.joblib")
clf      = obj['model']
scaler   = obj['scaler']
features = obj['features']

df_train = pd.read_parquet(UNSW_DIR / "UNSW_NB15_training-set.parquet")
df_test  = pd.read_parquet(UNSW_DIR / "UNSW_NB15_testing-set.parquet")

DROP    = ['attack_cat', 'label']
X_test  = df_test.drop(columns=DROP).select_dtypes(include=[np.number])
X_test  = X_test.reindex(columns=features, fill_value=0)
X_test  = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
y_test  = df_test['label'].values
cats    = df_test['attack_cat'].values

X_test_s = scaler.transform(X_test)
y_pred   = (clf.predict(X_test_s) == -1).astype(int)
scores   = -clf.score_samples(X_test_s)

print("Données chargées.")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Recall par type d'attaque (UNSW vs CIC-brut)
# ══════════════════════════════════════════════════════════════════════════════
print("Génération figure 1...")

summary = pd.read_csv(RES_DIR / "full_eval_summary.csv")
unsw = summary[summary['dataset'] == 'UNSW-NB15'].copy()
cic  = summary[summary['dataset'] == 'CIC-brut'].copy()

# Attaques communes
common = set(unsw['label']) & set(cic['label'])
unsw_c = unsw[unsw['label'].isin(common)].set_index('label')['recall']
cic_c  = cic[cic['label'].isin(common)].set_index('label')['recall']
labels_common = sorted(common, key=lambda x: unsw_c.get(x, 0), reverse=True)

x    = np.arange(len(labels_common))
w    = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w/2, [unsw_c.get(l, 0) for l in labels_common], w,
       label='UNSW-NB15', color=C_BLUE, alpha=0.85)
ax.bar(x + w/2, [cic_c.get(l, 0) for l in labels_common], w,
       label='CIC-IDS (flux brut)', color=C_RED, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(labels_common, rotation=30, ha='right')
ax.set_ylabel("Recall")
ax.set_ylim(0, 1.15)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.legend()
ax.set_title("Recall par type d'attaque — comparaison datasets")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_recall_par_attaque.pdf", bbox_inches='tight')
plt.savefig(FIG_DIR / "fig1_recall_par_attaque.png", bbox_inches='tight')
plt.close()
print("  ✓ fig1_recall_par_attaque")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Comparaison des 3 modèles (IF / LOF / OC-SVM)
# ══════════════════════════════════════════════════════════════════════════════
print("Génération figure 2...")

optim = pd.read_csv(RES_DIR / "optim_results.csv")
best_per_model = optim.loc[optim.groupby('model')['f1'].idxmax()]

models = best_per_model['model'].tolist()
f1s    = best_per_model['f1'].tolist()
recs   = best_per_model['recall'].tolist()
colors = [C_BLUE, C_GREEN, C_RED]

x   = np.arange(len(models))
w   = 0.35
fig, ax = plt.subplots(figsize=(7, 4))
b1 = ax.bar(x - w/2, f1s,  w, label='F1',    color=colors, alpha=0.85)
b2 = ax.bar(x + w/2, recs, w, label='Recall', color=colors, alpha=0.45)

for bar, val in zip(b1, f1s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha='center', va='bottom', fontsize=9)
for bar, val in zip(b2, recs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.2%}", ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1.15)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
solid  = mpatches.Patch(color='gray', alpha=0.85, label='F1')
transp = mpatches.Patch(color='gray', alpha=0.45, label='Recall')
ax.legend(handles=[solid, transp])
ax.set_title("Comparaison des modèles — meilleure configuration")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_comparaison_modeles.pdf", bbox_inches='tight')
plt.savefig(FIG_DIR / "fig2_comparaison_modeles.png", bbox_inches='tight')
plt.close()
print("  ✓ fig2_comparaison_modeles")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Distribution des scores d'anomalie
# ══════════════════════════════════════════════════════════════════════════════
print("Génération figure 3...")

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(scores[y_test == 0], bins=80, alpha=0.6, color=C_BLUE,
        label='Normal', density=True)
ax.hist(scores[y_test == 1], bins=80, alpha=0.6, color=C_RED,
        label='Attaque', density=True)

threshold = np.percentile(scores, 100 * (1 - 0.499))
ax.axvline(threshold, color='black', linestyle='--', linewidth=1.2,
           label=f'Seuil (contamination=0.499)')
ax.set_xlabel("Score d'anomalie (plus élevé = plus suspect)")
ax.set_ylabel("Densité")
ax.set_title("Distribution des scores d'anomalie — UNSW-NB15")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig3_distribution_scores.pdf", bbox_inches='tight')
plt.savefig(FIG_DIR / "fig3_distribution_scores.png", bbox_inches='tight')
plt.close()
print("  ✓ fig3_distribution_scores")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Matrice de confusion
# ══════════════════════════════════════════════════════════════════════════════
print("Génération figure 4...")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap='Blues')
plt.colorbar(im, ax=ax)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Prédit Normal', 'Prédit Attaque'])
ax.set_yticklabels(['Réel Normal', 'Réel Attaque'])
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i,j]:,}", ha='center', va='center',
                color='white' if cm[i,j] > cm.max()/2 else 'black',
                fontsize=13, fontweight='bold')
ax.set_title("Matrice de confusion — modèle final (UNSW-NB15)")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig4_confusion_matrix.pdf", bbox_inches='tight')
plt.savefig(FIG_DIR / "fig4_confusion_matrix.png", bbox_inches='tight')
plt.close()
print("  ✓ fig4_confusion_matrix")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Courbe Precision/Recall
# ══════════════════════════════════════════════════════════════════════════════
print("Génération figure 5...")

prec_curve, rec_curve, _ = precision_recall_curve(y_test, scores)
ap = average_precision_score(y_test, scores)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(rec_curve, prec_curve, color=C_BLUE, linewidth=2,
        label=f'Isolation Forest (AP = {ap:.3f})')
ax.axhline(y_test.mean(), color=C_GRAY, linestyle='--', linewidth=1,
           label=f'Baseline aléatoire ({y_test.mean():.2%})')
ax.scatter([y_pred[y_test==1].mean()], [y_pred[y_pred==1].mean()],
           color=C_RED, zorder=5, s=80, label='Seuil actuel')
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.05)
ax.set_title("Courbe Precision-Recall — UNSW-NB15")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig5_precision_recall.pdf", bbox_inches='tight')
plt.savefig(FIG_DIR / "fig5_precision_recall.png", bbox_inches='tight')
plt.close()
print("  ✓ fig5_precision_recall")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Recall par catégorie UNSW (bar horizontal)
# ══════════════════════════════════════════════════════════════════════════════
print("Génération figure 6...")

recall_by_cat = {}
for cat in np.unique(cats):
    if cat == 'Normal':
        continue
    idx = cats == cat
    recall_by_cat[cat] = y_pred[idx].mean()

cats_sorted = sorted(recall_by_cat, key=recall_by_cat.get)
vals = [recall_by_cat[c] for c in cats_sorted]
colors_bar = [C_GREEN if v >= 0.8 else C_BLUE if v >= 0.5 else C_RED for v in vals]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.barh(cats_sorted, vals, color=colors_bar, alpha=0.85)
for bar, val in zip(bars, vals):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.1%}", va='center', fontsize=9)
ax.set_xlim(0, 1.15)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("Recall")
ax.set_title("Recall par catégorie d'attaque — UNSW-NB15")
good   = mpatches.Patch(color=C_GREEN, alpha=0.85, label='≥ 80%')
medium = mpatches.Patch(color=C_BLUE,  alpha=0.85, label='50–80%')
bad    = mpatches.Patch(color=C_RED,   alpha=0.85, label='< 50%')
ax.legend(handles=[good, medium, bad], loc='lower right')
plt.tight_layout()
plt.savefig(FIG_DIR / "fig6_recall_categories_unsw.pdf", bbox_inches='tight')
plt.savefig(FIG_DIR / "fig6_recall_categories_unsw.png", bbox_inches='tight')
plt.close()
print("  ✓ fig6_recall_categories_unsw")

print(f"\n✅ 6 figures sauvegardées dans {FIG_DIR}")
print("   Format : .pdf (LaTeX) + .png (aperçu)")
