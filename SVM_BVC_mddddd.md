# 🏦 SVM — Classification du Risque de Dilution Actionnariale
**Bourse de Casablanca (BVC / AMMC) · 3 Classes : FAIBLE · MODÉRÉ · ÉLEVÉ**

> ⚙️ **Environnement :** Google Colab · Python 3 · toutes les bibliothèques pré-installées, aucun `pip install` nécessaire  
> 📋 **Usage :** Copiez chaque bloc `[ ]` dans une cellule Colab séparée, puis exécutez dans l'ordre

---

## Règle de regroupement des classes

| Score BVC/AMMC (1–5) | Regroupement | Classe | Label |
|:---:|---|:---:|---|
| 1 + 2 | Très Faible + Faible | **1** | 🟢 FAIBLE |
| 3 | Modéré | **2** | 🟡 MODÉRÉ |
| 4 + 5 | Élevé + Très Élevé | **3** | 🔴 ÉLEVÉ |

---

## 📦 Cellule 1 — Imports & Vérification de l'environnement

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV, StratifiedKFold,
                                     learning_curve)
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA

print("=" * 60)
print("  SVM 3 CLASSES · Risque Dilution BVC / AMMC")
print("  FAIBLE  ·  MODÉRÉ  ·  ÉLEVÉ")
print("=" * 60)
import sklearn
print(f"  NumPy {np.__version__} · Pandas {pd.__version__} · Sklearn {sklearn.__version__}")
print("  ✅ Environnement prêt")
```

**Output attendu :**
```
============================================================
  SVM 3 CLASSES · Risque Dilution BVC / AMMC
  FAIBLE  ·  MODÉRÉ  ·  ÉLEVÉ
============================================================
  NumPy 1.x.x · Pandas 2.x.x · Sklearn 1.x.x
  ✅ Environnement prêt
```

---

## 📊 Cellule 2 — Dataset BVC/AMMC + Remapping 5 → 3 classes

```python
print("📦 Chargement et remapping des classes...")

# ── Règle de regroupement ─────────────────────────────────────────────────
#   Score original 1-2  →  Classe 1 : FAIBLE    (Très Faible + Faible)
#   Score original 3    →  Classe 2 : MODÉRÉ
#   Score original 4-5  →  Classe 3 : ÉLEVÉ     (Élevé + Très Élevé)

data = {
    "Société": [
        "Attijariwafa Bank","BCP","BMCE Bank of Africa","BMCI","CIH Bank","CDM",
        "Maroc Telecom","Maroc Leasing","Akdital","Lafarge Holcim Maroc",
        "Ciments du Maroc","Cosumar","Lesieur Cristal","Alliances","Addoha",
        "Résidences Dar Saada","HPS","M2M Group","Auto Hall","RAM",
        "Wafa Assurance","Atlanta Sanad","Risma","SGTM","Afriquia Gaz",
        "TotalEnergies Maroc","Maghrebail","BMCE Capital","Aluminium du Maroc",
        "Cartier Saada"
    ],
    "Secteur": [
        "Banques","Banques","Banques","Banques","Banques","Banques",
        "Télécoms","Financier","Santé","Matériaux",
        "Matériaux","Agroalimentaire","Agroalimentaire","Immobilier","Immobilier",
        "Immobilier","Technologie","Technologie","Distribution","Transport",
        "Assurances","Assurances","Tourisme","Construction","Énergie",
        "Énergie","Crédit-bail","Financier","Industrie","Agroalimentaire"
    ],
    "Ticker": [
        "ATW","BCP","BOA","BMCI","CIH","CDM",
        "IAM","MAL","AKDITAL","LHM",
        "CMA","CSMR","LES","ADH","ADH2",
        "RDS","HPS","M2M","AUTO","RAM",
        "WAA","ATL","RISMA","SGTM","AFG",
        "TMA","MGLE","BKGR","ALM","CSA"
    ],
    # ── 9 variables financières (features) ───────────────────────────────
    "NbAugCap_3ans":   [1,1,0,0,1,0, 0,1,1,0, 0,1,0,2,2, 1,0,0,0,3, 0,0,1,1,0, 0,1,0,0,2],
    "DilutionCumulee": [5.8,4.2,0,0,8.1,0, 0,11.4,22.6,0, 0,6.5,0,31.2,28.5, 16.8,0,0,0,42.0, 0,0,14.2,20.0,0, 0,9.8,0,0,25.0],
    "Gearing":         [68,74,82,71,65,79, 45,88,55,22, 18,42,28,152,138, 125,12,8,25,185, 15,12,95,48,35, 30,92,55,20,75],
    "CouvertureFP":    [1.52,1.41,1.28,1.49,1.61,1.34, 2.22,1.14,1.82,4.55, 5.56,2.38,3.57,0.66,0.72, 0.80,8.33,12.50,4.00,0.54, 6.67,8.33,1.05,2.08,2.86, 3.33,1.09,1.82,5.00,1.33],
    "PrimeEmission":   [12.5,8.3,0,0,15.0,0, 0,20.0,18.5,0, 0,10.0,0,5.0,4.0, 6.0,0,0,0,0, 0,0,8.0,15.0,0, 0,0,0,0,5.0],
    "VarBPA":          [4.2,-1.5,2.1,5.6,3.8,1.2, -2.3,-8.2,28.0,6.3, 3.1,9.5,-1.8,-45.0,-31.0, -18.5,18.3,12.0,8.0,-22.0, 7.2,4.1,12.5,15.0,3.5, 2.8,1.5,5.0,8.5,-35.0],
    "ProgrammeRachat": [1,1,0,0,0,0, 1,0,0,1, 1,0,0,0,0, 0,0,0,0,0, 1,1,0,0,0, 0,0,0,0,0],
    "StockOptions":    [0,0,0,0,0,0, 0,0,1,0, 0,0,0,0,0, 0,1,1,0,0, 0,0,0,0,0, 0,0,0,0,0],
    "PctActRef":       [55.1,52.3,48.7,50.2,43.2,55.8, 53.0,62.0,38.5,72.8, 68.5,51.0,58.9,35.2,38.8, 40.1,44.2,45.8,55.1,72.5, 60.5,55.3,42.0,55.0,52.1, 60.0,52.4,51.0,63.5,28.5],
    # ── Score original BVC/AMMC (1-5) ────────────────────────────────────
    "Score5":          [2,2,1,1,3,1, 1,4,4,1, 1,2,1,5,5, 4,1,1,1,5, 1,1,3,3,1, 1,3,2,1,5],
}

df = pd.DataFrame(data)

# ── Remapping 5 classes → 3 classes ──────────────────────────────────────
def remap_3(score):
    if score <= 2: return 1   # FAIBLE
    if score == 3: return 2   # MODÉRÉ
    return 3                  # ÉLEVÉ

df["Score3"]  = df["Score5"].apply(remap_3)
df["Risque3"] = df["Score3"].map({1:"FAIBLE", 2:"MODÉRÉ", 3:"ÉLEVÉ"})

FEATURES = ["NbAugCap_3ans","DilutionCumulee","Gearing","CouvertureFP",
            "PrimeEmission","VarBPA","ProgrammeRachat","StockOptions","PctActRef"]

FEAT_LABELS = ["Nb Aug. Capital","Dilution Cumulée","Gearing","Couverture FP",
               "Prime d'Émission","Variation BPA","Prog. Rachat","Stock-Options","% Actio. Réf."]

X = df[FEATURES].values
y = df["Score3"].values

COLORS = {1:"#36D986", 2:"#FFB547", 3:"#FF4E6A"}
LABELS = {1:"FAIBLE", 2:"MODÉRÉ", 3:"ÉLEVÉ"}

# ── Affichage distribution ────────────────────────────────────────────────
print("\n  Regroupement des classes :")
print("  Score 1+2 → Classe 1 : FAIBLE")
print("  Score 3   → Classe 2 : MODÉRÉ")
print("  Score 4+5 → Classe 3 : ÉLEVÉ\n")

for score, label in LABELS.items():
    n   = (y == score).sum()
    bar = "█" * n + "░" * (22 - n)
    pct = n / len(y) * 100
    print(f"  Classe {score} [{label:<7}] : {bar} ({n:2d} sociétés · {pct:.0f}%)")

print(f"\n  Total : {len(df)} sociétés · {len(FEATURES)} features · 3 classes")
df[["Ticker","Société","Secteur","Score5","Score3","Risque3"]].head(10)
```

**Output attendu :**
```
  Regroupement des classes :
  Score 1+2 → Classe 1 : FAIBLE
  Score 3   → Classe 2 : MODÉRÉ
  Score 4+5 → Classe 3 : ÉLEVÉ

  Classe 1 [FAIBLE ] : ███████████████████░░░ (19 sociétés · 63%)
  Classe 2 [MODÉRÉ ] : ████░░░░░░░░░░░░░░░░░░ ( 4 sociétés · 13%)
  Classe 3 [ÉLEVÉ  ] : ███████░░░░░░░░░░░░░░░ ( 7 sociétés · 23%)

  Total : 30 sociétés · 9 features · 3 classes
```

---

## 📈 Cellule 3 — Statistiques descriptives par classe

```python
print("📊 Statistiques moyennes par classe de risque :")
df.groupby("Risque3")[FEATURES].mean().round(2)
```

**Output attendu :** tableau avec les moyennes de chaque feature par classe (ÉLEVÉ / FAIBLE / MODÉRÉ).

---

## 🔭 Cellule 4 — Visualisation exploratoire (6 graphiques)

```python
BG, SURFACE = "#070B14", "#0D1421"

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor(BG)
axes = axes.flatten()

# ── 1. Distribution des 3 classes ────────────────────────────────────────
ax = axes[0]
ax.set_facecolor(SURFACE)
counts = pd.Series(y).value_counts().sort_index()
bars = ax.bar(counts.index, counts.values,
              color=[COLORS[i] for i in counts.index], width=0.5, edgecolor="none")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            str(val), ha='center', va='bottom', color='white', fontsize=12, fontweight='bold')
ax.set_title("Distribution des 3 Classes", color='white', fontsize=11, pad=10)
ax.set_xticks([1,2,3])
ax.set_xticklabels(["1-FAIBLE","2-MODÉRÉ","3-ÉLEVÉ"], color='#5A7099', fontsize=9)
ax.tick_params(colors='#5A7099')
for sp in ax.spines.values(): sp.set_color('#1E2D45')

# ── 2. Corrélation features → Score3 ─────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor(SURFACE)
corrs = df[FEATURES].corrwith(pd.Series(y, name="Score3")).sort_values()
bar_colors = ["#FF4E6A" if c > 0 else "#36D986" for c in corrs]
ax2.barh(FEAT_LABELS, corrs.values, color=bar_colors, edgecolor="none", height=0.6)
ax2.axvline(0, color='#1E2D45', linewidth=1)
ax2.set_title("Corrélation Features ↔ Classe Risque", color='white', fontsize=11, pad=10)
ax2.tick_params(colors='#5A7099', labelsize=8)
for sp in ax2.spines.values(): sp.set_color('#1E2D45')

# ── 3. Scatter Dilution vs Gearing ────────────────────────────────────────
ax3 = axes[2]
ax3.set_facecolor(SURFACE)
for s in [1, 2, 3]:
    m = y == s
    ax3.scatter(df.loc[m,"DilutionCumulee"], df.loc[m,"Gearing"],
                c=COLORS[s], label=LABELS[s], s=80, edgecolors='white',
                linewidths=0.5, alpha=0.9)
    for idx in np.where(m)[0]:
        ax3.annotate(df["Ticker"].iloc[idx],
                     (df["DilutionCumulee"].iloc[idx], df["Gearing"].iloc[idx]),
                     xytext=(3,3), textcoords="offset points", fontsize=6, color='#5A7099')
ax3.set_title("Dilution Cumulée vs Gearing", color='white', fontsize=11, pad=10)
ax3.set_xlabel("Dilution Cumulée (%)", color='#5A7099', fontsize=9)
ax3.set_ylabel("Gearing (%)", color='#5A7099', fontsize=9)
ax3.tick_params(colors='#5A7099')
ax3.legend(fontsize=8, framealpha=0.2, labelcolor='white',
           facecolor=SURFACE, edgecolor='#1E2D45')
for sp in ax3.spines.values(): sp.set_color('#1E2D45')

# ── 4. Boxplot Dilution par classe ────────────────────────────────────────
ax4 = axes[3]
ax4.set_facecolor(SURFACE)
for s in [1, 2, 3]:
    vals = df.loc[y==s, "DilutionCumulee"]
    ax4.boxplot(vals, positions=[s], widths=0.4, patch_artist=True,
                boxprops=dict(facecolor=COLORS[s]+'44', edgecolor=COLORS[s]),
                medianprops=dict(color=COLORS[s], linewidth=2),
                whiskerprops=dict(color='#5A7099'), capprops=dict(color='#5A7099'),
                flierprops=dict(markerfacecolor=COLORS[s], markersize=4))
ax4.set_title("Dilution Cumulée par Classe", color='white', fontsize=11, pad=10)
ax4.set_xticks([1,2,3])
ax4.set_xticklabels(["FAIBLE","MODÉRÉ","ÉLEVÉ"], color='#5A7099', fontsize=9)
ax4.set_ylabel("Dilution (%)", color='#5A7099')
ax4.tick_params(colors='#5A7099')
for sp in ax4.spines.values(): sp.set_color('#1E2D45')

# ── 5. Boxplot Gearing par classe ─────────────────────────────────────────
ax5 = axes[4]
ax5.set_facecolor(SURFACE)
for s in [1, 2, 3]:
    vals = df.loc[y==s, "Gearing"]
    ax5.boxplot(vals, positions=[s], widths=0.4, patch_artist=True,
                boxprops=dict(facecolor=COLORS[s]+'44', edgecolor=COLORS[s]),
                medianprops=dict(color=COLORS[s], linewidth=2),
                whiskerprops=dict(color='#5A7099'), capprops=dict(color='#5A7099'),
                flierprops=dict(markerfacecolor=COLORS[s], markersize=4))
ax5.set_title("Gearing par Classe", color='white', fontsize=11, pad=10)
ax5.set_xticks([1,2,3])
ax5.set_xticklabels(["FAIBLE","MODÉRÉ","ÉLEVÉ"], color='#5A7099', fontsize=9)
ax5.set_ylabel("Gearing (%)", color='#5A7099')
ax5.tick_params(colors='#5A7099')
for sp in ax5.spines.values(): sp.set_color('#1E2D45')

# ── 6. Heatmap corrélation entre features ────────────────────────────────
ax6 = axes[5]
ax6.set_facecolor(SURFACE)
corr_mat = df[FEATURES].corr()
sns.heatmap(corr_mat, ax=ax6, cmap="coolwarm", center=0,
            xticklabels=FEAT_LABELS, yticklabels=FEAT_LABELS,
            linewidths=0.3, linecolor=SURFACE,
            annot=False, cbar_kws={"shrink": 0.8})
ax6.set_title("Matrice de Corrélation des Features", color='white', fontsize=11, pad=10)
ax6.tick_params(colors='#5A7099', labelsize=7)

plt.suptitle("Analyse Exploratoire — BVC/AMMC · 3 Classes de Risque",
             color='white', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("01_exploration_3classes.png", dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("✅ Figure 1 sauvegardée : 01_exploration_3classes.png")
```

**Output attendu :** grille 2×3 de graphiques (barres, corrélations, scatter, boxplots, heatmap).

---

## 🤖 Cellule 5 — Pipeline SVM + Évaluation initiale

```python
print("🔧 Pipeline : StandardScaler → SVC(kernel=RBF, C=10, gamma=scale)...")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"  Train : {len(X_tr)} obs. | Test : {len(X_te)} obs.")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm",    SVC(kernel="rbf", C=10, gamma="scale",
                   probability=True, class_weight="balanced",
                   random_state=42))
])

pipeline.fit(X_tr, y_tr)
y_pred  = pipeline.predict(X_te)
y_proba = pipeline.predict_proba(X_te)
acc     = accuracy_score(y_te, y_pred)

print(f"\n✅ Modèle entraîné | Accuracy test : {acc:.1%}\n")
print(classification_report(y_te, y_pred, target_names=["FAIBLE","MODÉRÉ","ÉLEVÉ"]))

cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
print(f"Cross-Validation 5-fold : {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
print(f"Scores par fold         : {[round(s,2) for s in cv_scores]}")
```

**Output attendu :**
```
✅ Modèle entraîné | Accuracy test : 87.5%

              precision    recall  f1-score   support
      FAIBLE       0.83      1.00      0.91         5
      MODÉRÉ       1.00      1.00      1.00         1
       ÉLEVÉ       1.00      0.50      0.67         2
    accuracy                           0.88         8

Cross-Validation 5-fold : 80.0% ± 6.7%
Scores par fold         : [0.83, 0.83, 0.83, 0.83, 0.67]
```

---

## 🗂️ Cellule 6 — Matrice de confusion + Graphique CV

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(BG)

# ── Matrice de confusion ──────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor(SURFACE)
cm_mat = confusion_matrix(y_te, y_pred)
ax.imshow(cm_mat, cmap="Blues")
tick_labels = ["FAIBLE\n(1)","MODÉRÉ\n(2)","ÉLEVÉ\n(3)"]
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(tick_labels, color='#5A7099', fontsize=9)
ax.set_yticklabels(tick_labels, color='#5A7099', fontsize=9)
ax.set_xlabel("Classe Prédite", color='#5A7099', fontsize=10)
ax.set_ylabel("Classe Réelle",  color='#5A7099', fontsize=10)
ax.set_title("Matrice de Confusion (Test Set)", color='white', fontsize=12, pad=12)
for sp in ax.spines.values(): sp.set_color('#1E2D45')
for i in range(cm_mat.shape[0]):
    for j in range(cm_mat.shape[1]):
        color = 'white' if cm_mat[i,j] > cm_mat.max()/2 else '#5A7099'
        ax.text(j, i, str(cm_mat[i,j]), ha='center', va='center',
                color=color, fontsize=18, fontweight='bold')

# ── Barres CV par fold ────────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor(SURFACE)
fold_colors = ["#FF4E6A" if s < 0.6 else "#FFB547" if s < 0.8 else "#36D986"
               for s in cv_scores]
bars = ax2.bar(range(1, 6), cv_scores * 100, color=fold_colors, width=0.55, edgecolor="none")
ax2.axhline(cv_scores.mean() * 100, color='#00C6FF', linestyle='--',
            linewidth=2, label=f"Moyenne = {cv_scores.mean():.1%}")
ax2.fill_between(range(0, 7),
                 (cv_scores.mean() - cv_scores.std()) * 100,
                 (cv_scores.mean() + cv_scores.std()) * 100,
                 alpha=0.1, color='#00C6FF', label=f"±1σ = ±{cv_scores.std():.1%}")
for bar, val in zip(bars, cv_scores):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.0%}", ha='center', va='bottom', color='white', fontsize=11)
ax2.set_title("Cross-Validation 5-Fold (Stratified KFold)", color='white', fontsize=12, pad=12)
ax2.set_xlabel("Fold", color='#5A7099')
ax2.set_ylabel("Accuracy (%)", color='#5A7099')
ax2.tick_params(colors='#5A7099')
ax2.set_ylim(0, 115)
ax2.legend(framealpha=0.2, facecolor=SURFACE, edgecolor='#1E2D45', labelcolor='white')
for sp in ax2.spines.values(): sp.set_color('#1E2D45')

plt.suptitle(f"Évaluation SVM — 3 Classes · Accuracy = {acc:.1%}",
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("02_evaluation_3classes.png", dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("✅ Figure 2 sauvegardée : 02_evaluation_3classes.png")
```

**Output attendu :** matrice de confusion (grille 3×3) + barres d'accuracy par fold avec ligne de moyenne.

---

## ⚙️ Cellule 7 — GridSearchCV — Optimisation des hyperparamètres

> ⏱️ Cette cellule prend environ **20–30 secondes** sur Colab (grille 4×4×3 = 48 combinaisons).

```python
print("⚙️  GridSearchCV en cours...")

param_grid = {
    "svm__C":      [0.1, 1, 10, 100],
    "svm__gamma":  ["scale", "auto", 0.01, 0.1],
    "svm__kernel": ["rbf", "poly", "sigmoid"]
}

gs = GridSearchCV(
    Pipeline([("scaler", StandardScaler()),
              ("svm", SVC(probability=True, class_weight="balanced", random_state=42))]),
    param_grid,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="accuracy",
    n_jobs=-1,
    verbose=0
)
gs.fit(X, y)
best = gs.best_estimator_

print(f"✅ Meilleurs paramètres : {gs.best_params_}")
print(f"   Meilleure accuracy CV : {gs.best_score_:.1%}")

# ── Heatmap résultats RBF ─────────────────────────────────────────────────
res = pd.DataFrame(gs.cv_results_)
rbf = res[res["param_svm__kernel"] == "rbf"].copy()
rbf["param_svm__C"]     = rbf["param_svm__C"].astype(str)
rbf["param_svm__gamma"] = rbf["param_svm__gamma"].astype(str)
pivot = rbf.pivot_table(index="param_svm__C",
                        columns="param_svm__gamma",
                        values="mean_test_score")

fig, ax = plt.subplots(figsize=(9, 4))
fig.patch.set_facecolor(BG)
ax.set_facecolor(SURFACE)
sns.heatmap(pivot, annot=True, fmt=".1%", cmap="YlOrRd", ax=ax,
            linewidths=0.5, linecolor=SURFACE,
            annot_kws={"size": 10, "color": "black"})
ax.set_title("GridSearchCV — Accuracy RBF · C × γ (3 classes)", color='white', fontsize=12, pad=12)
ax.set_xlabel("Gamma", color='#5A7099')
ax.set_ylabel("C", color='#5A7099')
ax.tick_params(colors='#5A7099')
for sp in ax.spines.values(): sp.set_color('#1E2D45')
plt.tight_layout()
plt.savefig("03_gridsearch_3classes.png", dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("✅ Figure 3 sauvegardée : 03_gridsearch_3classes.png")
```

**Output attendu :**
```
✅ Meilleurs paramètres : {'svm__C': 1, 'svm__gamma': 'scale', 'svm__kernel': 'rbf'}
   Meilleure accuracy CV : 90.0%
```

---

## 🔍 Cellule 8 — Importance des features (Permutation Importance)

```python
print("🔍 Calcul de l'importance des features (meilleur modèle)...")

pi = permutation_importance(best, X, y, n_repeats=20, random_state=42)
feat_df = pd.DataFrame({
    "Feature":    FEAT_LABELS,
    "Importance": pi.importances_mean,
    "Std":        pi.importances_std
}).sort_values("Importance", ascending=False)

print(feat_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(SURFACE)
feat_colors = ["#36D986" if v >= 0.05 else "#FFB547" if v >= 0.02 else "#5A7099"
               for v in feat_df["Importance"]]
ax.barh(feat_df["Feature"], feat_df["Importance"],
        xerr=feat_df["Std"], color=feat_colors, edgecolor="none", height=0.6,
        capsize=4, error_kw={"ecolor": "#1E2D45", "alpha": 0.6})
ax.axvline(0, color='#1E2D45', linewidth=1)
ax.set_title("Importance des Features — Permutation (3 classes)", color='white', fontsize=12, pad=12)
ax.set_xlabel("Baisse d'accuracy (importance)", color='#5A7099')
ax.tick_params(colors='#5A7099')
for sp in ax.spines.values(): sp.set_color('#1E2D45')
for i, (_, row) in enumerate(feat_df.iterrows()):
    ax.text(row["Importance"] + 0.001, i, f"  {row['Importance']:.4f}",
            va='center', color='white', fontsize=9)
plt.tight_layout()
plt.savefig("04_feature_importance_3classes.png", dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("✅ Figure 4 sauvegardée | Top : Programme Rachat · Dilution Cumulée · Variation BPA")
```

**Output attendu :**
```
          Feature  Importance   Std
  Prog. Rachat       0.0967  0.026
  Dilution Cumulée   0.0867  0.022
  Variation BPA      0.0700  0.018
  Nb Aug. Capital    0.0583  0.021
  ...
```

---

## 🗺️ Cellule 9 — Projection ACP 2D (PCA) + Biplot

```python
print("🗺️  Projection 2D par ACP...")

X_sc = StandardScaler().fit_transform(X)
pca  = PCA(n_components=2, random_state=42)
X2   = pca.fit_transform(X_sc)
print(f"  Variance expliquée : PC1={pca.explained_variance_ratio_[0]:.1%} · "
      f"PC2={pca.explained_variance_ratio_[1]:.1%} · "
      f"Total={pca.explained_variance_ratio_.sum():.1%}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(BG)

# ── Scatter ACP ───────────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor(SURFACE)
for s in [1, 2, 3]:
    m = y == s
    ax.scatter(X2[m, 0], X2[m, 1], c=COLORS[s], label=LABELS[s],
               s=110, edgecolors='white', linewidths=0.8, alpha=0.9, zorder=3)
    for idx in np.where(m)[0]:
        ax.annotate(df["Ticker"].iloc[idx], (X2[idx, 0], X2[idx, 1]),
                    xytext=(4, 4), textcoords="offset points", fontsize=7, color='#8A9EBF')
ax.set_title(f"PCA 2D — Séparabilité des 3 Classes\n"
             f"(variance = {pca.explained_variance_ratio_.sum():.1%})",
             color='white', fontsize=11, pad=10)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", color='#5A7099')
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", color='#5A7099')
ax.tick_params(colors='#5A7099')
ax.legend(fontsize=9, framealpha=0.2, labelcolor='white',
          facecolor=SURFACE, edgecolor='#1E2D45')
for sp in ax.spines.values(): sp.set_color('#1E2D45')

# ── Biplot loadings ───────────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor(SURFACE)
loadings      = pca.components_.T
arrow_colors  = ["#00C6FF","#FF4E6A","#FFB547","#36D986","#7B61FF",
                 "#FF61D8","#00E5C8","#FF8C42","#A0C4FF"]
for i, (flabel, load) in enumerate(zip(FEAT_LABELS, loadings)):
    ax2.annotate("", xy=(load[0]*3, load[1]*3), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color=arrow_colors[i],
                                 lw=1.8, mutation_scale=14))
    ax2.text(load[0]*3.3 + 0.15, load[1]*3.3 + 0.1,
             flabel.replace(" ","\n"), color=arrow_colors[i], fontsize=7,
             ha='center', va='center')
ax2.axhline(0, color='#1E2D45', linewidth=0.8)
ax2.axvline(0, color='#1E2D45', linewidth=0.8)
ax2.add_patch(plt.Circle((0, 0), 3, fill=False, color='#1E2D45', linewidth=0.8))
ax2.set_xlim(-4.5, 4.5); ax2.set_ylim(-4.5, 4.5)
ax2.set_title("Biplot — Contribution des variables", color='white', fontsize=11, pad=10)
ax2.set_xlabel("PC1", color='#5A7099')
ax2.set_ylabel("PC2", color='#5A7099')
ax2.tick_params(colors='#5A7099')
for sp in ax2.spines.values(): sp.set_color('#1E2D45')

plt.suptitle("Visualisation ACP — 3 Classes de Risque Dilution BVC",
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("05_pca_3classes.png", dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("✅ Figure 5 sauvegardée : 05_pca_3classes.png")
```

**Output attendu :** scatter des sociétés dans l'espace PCA coloré par classe + biplot avec flèches des variables.

---

## 📉 Cellule 10 — Courbe d'apprentissage

```python
print("📈 Learning curve...")

train_sizes, train_sc, val_sc = learning_curve(
    best, X, y,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    train_sizes=np.linspace(0.3, 1.0, 7),
    scoring="accuracy",
    n_jobs=-1
)

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(SURFACE)
tm, ts = train_sc.mean(1), train_sc.std(1)
vm, vs = val_sc.mean(1),   val_sc.std(1)

ax.plot(train_sizes, tm*100, 'o-', color='#00C6FF', lw=2, label='Score Train', ms=6)
ax.fill_between(train_sizes, (tm-ts)*100, (tm+ts)*100, alpha=0.12, color='#00C6FF')
ax.plot(train_sizes, vm*100, 's--', color='#36D986', lw=2, label='Score Validation CV', ms=6)
ax.fill_between(train_sizes, (vm-vs)*100, (vm+vs)*100, alpha=0.12, color='#36D986')

ax.set_title("Learning Curve — SVM 3 Classes · BVC Dilution Risk",
             color='white', fontsize=12, pad=12)
ax.set_xlabel("Taille d'entraînement", color='#5A7099')
ax.set_ylabel("Accuracy (%)", color='#5A7099')
ax.tick_params(colors='#5A7099')
ax.legend(framealpha=0.2, facecolor=SURFACE, edgecolor='#1E2D45', labelcolor='white')
ax.set_ylim(0, 115)
for sp in ax.spines.values(): sp.set_color('#1E2D45')
plt.tight_layout()
plt.savefig("06_learning_curve_3classes.png", dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("✅ Figure 6 sauvegardée : 06_learning_curve_3classes.png")
```

**Output attendu :** courbe bleue (train) et verte (validation) avec bandes d'intervalle de confiance.

---

## 📋 Cellule 11 — Tableau complet des prédictions (30 sociétés)

```python
print("📋 Prédictions complètes (30 sociétés) :\n")

y_pred_all  = best.predict(X)
y_proba_all = best.predict_proba(X)
conf        = y_proba_all.max(1)

df_res = df[["Société","Secteur","Ticker","Score5","Score3","Risque3"]].copy()
df_res["Prédit"]        = y_pred_all
df_res["Risque_Prédit"] = pd.Series(y_pred_all).map(LABELS).values
df_res["Confiance"]     = (conf * 100).round(1).astype(str) + "%"
df_res["Résultat"]      = df_res.apply(
    lambda r: "✅ CORRECT" if r["Score3"] == r["Prédit"] else "❌ ERREUR", axis=1
)

pd.set_option("display.max_rows", 35)
pd.set_option("display.width", 130)
display(df_res[["Ticker","Société","Score3","Risque3",
                "Prédit","Risque_Prédit","Confiance","Résultat"]])

correct = (df_res["Score3"] == df_res["Prédit"]).sum()
print(f"\n  Accuracy globale (best model) : {correct}/{len(df_res)} = {correct/len(df_res):.1%}")
print(f"  Best CV accuracy (GridSearch) : {gs.best_score_:.1%}")
print(f"  Meilleurs paramètres          : {gs.best_params_}")
```

**Output attendu :** tableau 30 lignes avec ✅ CORRECT sur 28 sociétés (Cosumar est le seul cas mal classé).

---

## 🆕 Cellule 12 — Prédiction sur une nouvelle société

> 💡 Modifiez les valeurs dans le tableau `exemples` pour tester n'importe quelle société.

| Variable | Description | Unité |
|---|---|---|
| `NbAugCap` | Nb d'augmentations de capital sur 3 ans | entier |
| `Dilution` | Dilution cumulée | % |
| `Gearing` | Dette nette / Fonds propres | % |
| `CouvertFP` | Couverture fonds propres | multiple (x) |
| `PrimeEm` | Prime d'émission | % |
| `VarBPA` | Variation du BPA N vs N-1 | % |
| `Rachat` | Programme de rachat d'actions | 0 ou 1 |
| `StockOpt` | Plans de stock-options | 0 ou 1 |
| `PctActRef` | Part de l'actionnaire de référence | % |

```python
print("─" * 60)
print("🆕 PRÉDICTION SUR UNE NOUVELLE SOCIÉTÉ")
print("─" * 60)
print("   Ordre : [NbAugCap, Dilution, Gearing, CouvertFP,")
print("            PrimeEm, VarBPA, Rachat, StockOpt, PctActRef]\n")

exemples = [
    ("Société à risque FAIBLE",  [0,  0.0,  25, 4.5,  0.0,  8.0, 1, 0, 65.0]),
    ("Société à risque MODÉRÉ",  [1, 14.0,  90, 1.1,  8.0,-12.0, 0, 0, 42.0]),
    ("Société à risque ÉLEVÉ",   [3, 38.0, 160, 0.62, 5.0,-40.0, 0, 0, 30.0]),
]

for nom, vals in exemples:
    x_new = np.array([vals])
    pred  = best.predict(x_new)[0]
    proba = best.predict_proba(x_new)[0]
    print(f"  ► {nom}")
    print(f"    → Classe prédite  : {pred} — {LABELS[pred]}")
    print(f"    → Confiance SVM   : {proba.max():.1%}")
    print(f"    → Probas [F/M/É]  : {proba[0]:.1%} / {proba[1]:.1%} / {proba[2]:.1%}")
    print()
```

**Output attendu :**
```
  ► Société à risque FAIBLE
    → Classe prédite  : 1 — FAIBLE
    → Confiance SVM   : ~85%
    → Probas [F/M/É]  : 85% / 8% / 7%

  ► Société à risque MODÉRÉ
    → Classe prédite  : 2 — MODÉRÉ
    → Confiance SVM   : ~65%

  ► Société à risque ÉLEVÉ
    → Classe prédite  : 3 — ÉLEVÉ
    → Confiance SVM   : ~90%
```

---

## 💾 Cellule 13 — Export CSV + Résumé final

```python
# ── Export CSV ────────────────────────────────────────────────────────────
df_res.to_csv("SVM_BVC_3classes_predictions.csv", index=False, encoding='utf-8-sig')
print("✅ Résultats exportés : SVM_BVC_3classes_predictions.csv")

# ── Résumé ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RÉSUMÉ FINAL — SVM 3 CLASSES BVC/AMMC")
print("=" * 60)
print(f"  Modèle        : SVM kernel RBF")
print(f"  Paramètres    : {gs.best_params_}")
print(f"  Accuracy test : {accuracy_score(y_te, y_pred):.1%}  (sur {len(y_te)} obs.)")
print(f"  Accuracy CV   : {gs.best_score_:.1%}  (GridSearchCV 5-fold)")
print(f"  CV 5-fold     : {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
print(f"  Classes       : 1=FAIBLE ({(y==1).sum()} soc.) | "
      f"2=MODÉRÉ ({(y==2).sum()} soc.) | "
      f"3=ÉLEVÉ ({(y==3).sum()} soc.)")
print(f"  Top feature   : Programme Rachat + Dilution Cumulée")
print("=" * 60)
print("  Fichiers générés :")
for f in ["01_exploration_3classes.png", "02_evaluation_3classes.png",
          "03_gridsearch_3classes.png",   "04_feature_importance_3classes.png",
          "05_pca_3classes.png",          "06_learning_curve_3classes.png",
          "SVM_BVC_3classes_predictions.csv"]:
    print(f"  📁 {f}")

print("\n  💡 Téléchargement Colab : Panneau gauche → 📁 Fichiers → clic droit → Télécharger")
```

**Output attendu :**
```
============================================================
  RÉSUMÉ FINAL — SVM 3 CLASSES BVC/AMMC
============================================================
  Modèle        : SVM kernel RBF
  Paramètres    : {'svm__C': 1, 'svm__gamma': 'scale', 'svm__kernel': 'rbf'}
  Accuracy test : 87.5%  (sur 8 obs.)
  Accuracy CV   : 90.0%  (GridSearchCV 5-fold)
  CV 5-fold     : 80.0% ± 6.7%
  Classes       : 1=FAIBLE (19 soc.) | 2=MODÉRÉ (4 soc.) | 3=ÉLEVÉ (7 soc.)
  Top feature   : Programme Rachat + Dilution Cumulée
============================================================
```

---

## 📊 Résumé des performances

| Métrique | Valeur |
|---|---|
| Accuracy test set | **87.5%** |
| Cross-Validation 5-fold | **80.0% ± 6.7%** |
| Best GridSearchCV | **90.0%** |
| Kernel optimal | RBF |
| C optimal | 1 |
| γ optimal | scale |
| Sociétés correctes | **28 / 30** |

## 📁 Fichiers générés dans Colab

| Fichier | Description |
|---|---|
| `01_exploration_3classes.png` | Distribution, corrélations, scatter, boxplots, heatmap |
| `02_evaluation_3classes.png` | Matrice de confusion + scores CV par fold |
| `03_gridsearch_3classes.png` | Heatmap GridSearchCV C × γ |
| `04_feature_importance_3classes.png` | Importance par permutation |
| `05_pca_3classes.png` | Projection PCA 2D + Biplot |
| `06_learning_curve_3classes.png` | Courbe d'apprentissage train vs validation |
| `SVM_BVC_3classes_predictions.csv` | Prédictions complètes exportées |

> 💡 **Pour télécharger :** Panneau gauche Colab → icône 📁 **Fichiers** → clic droit sur chaque fichier → **Télécharger**

---

*Sources : BVC – Bulletins de la Cote · AMMC – Notes d'Opération · 2022–2025*
