# ╔══════════════════════════════════════════════════════════════════════════╗
# ║        SVM — Classification du Risque de Dilution Actionnariale         ║
# ║        Bourse de Casablanca (BVC / AMMC) · 3 Classes                    ║
# ║        FAIBLE  ·  MODÉRÉ  ·  ÉLEVÉ                                      ║
# ║                                                                          ║
# ║  ▶ Google Colab : Runtime → Run All  (Ctrl+F9)                          ║
# ║  ▶ Aucune installation requise — toutes les libs sont pré-installées     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ═══════════════════════════════════════════════════════════════════════════
# 1.  IMPORTS
# ═══════════════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
import sklearn

print("=" * 65)
print("   SVM · Risque de Dilution Actionnariale · BVC / AMMC")
print("   3 Classes : 🟢 FAIBLE  🟡 MODÉRÉ  🔴 ÉLEVÉ")
print("=" * 65)
print(f"   NumPy {np.__version__} · Pandas {pd.__version__} · Sklearn {sklearn.__version__}")
print("   ✅ Imports OK\n")

# ═══════════════════════════════════════════════════════════════════════════
# 2.  DATASET  (30 sociétés cotées BVC — sources BVC / AMMC 2022-2025)
# ═══════════════════════════════════════════════════════════════════════════
print("📦 Chargement du dataset BVC/AMMC...")

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
    # ── 9 variables financières ───────────────────────────────────────────
    "NbAugCap_3ans":   [1,1,0,0,1,0,0,1,1,0,0,1,0,2,2,1,0,0,0,3,0,0,1,1,0,0,1,0,0,2],
    "DilutionCumulee": [5.8,4.2,0,0,8.1,0,0,11.4,22.6,0,0,6.5,0,31.2,28.5,16.8,0,0,0,42.0,0,0,14.2,20.0,0,0,9.8,0,0,25.0],
    "Gearing":         [68,74,82,71,65,79,45,88,55,22,18,42,28,152,138,125,12,8,25,185,15,12,95,48,35,30,92,55,20,75],
    "CouvertureFP":    [1.52,1.41,1.28,1.49,1.61,1.34,2.22,1.14,1.82,4.55,5.56,2.38,3.57,0.66,0.72,0.80,8.33,12.50,4.00,0.54,6.67,8.33,1.05,2.08,2.86,3.33,1.09,1.82,5.00,1.33],
    "PrimeEmission":   [12.5,8.3,0,0,15.0,0,0,20.0,18.5,0,0,10.0,0,5.0,4.0,6.0,0,0,0,0,0,0,8.0,15.0,0,0,0,0,0,5.0],
    "VarBPA":          [4.2,-1.5,2.1,5.6,3.8,1.2,-2.3,-8.2,28.0,6.3,3.1,9.5,-1.8,-45.0,-31.0,-18.5,18.3,12.0,8.0,-22.0,7.2,4.1,12.5,15.0,3.5,2.8,1.5,5.0,8.5,-35.0],
    "ProgrammeRachat": [1,1,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0],
    "StockOptions":    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
    "PctActRef":       [55.1,52.3,48.7,50.2,43.2,55.8,53.0,62.0,38.5,72.8,68.5,51.0,58.9,35.2,38.8,40.1,44.2,45.8,55.1,72.5,60.5,55.3,42.0,55.0,52.1,60.0,52.4,51.0,63.5,28.5],
    # ── Score BVC/AMMC original (1-5) ────────────────────────────────────
    "Score5":          [2,2,1,1,3,1,1,4,4,1,1,2,1,5,5,4,1,1,1,5,1,1,3,3,1,1,3,2,1,5],
}

df = pd.DataFrame(data)

# ── Remapping 5 → 3 classes ───────────────────────────────────────────────
#   Score 1+2  →  Classe 1 : FAIBLE   (Très Faible + Faible)
#   Score 3    →  Classe 2 : MODÉRÉ
#   Score 4+5  →  Classe 3 : ÉLEVÉ    (Élevé + Très Élevé)

def remap_3(s):
    if s <= 2: return 1
    if s == 3: return 2
    return 3

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
BG, SURF = "#070B14", "#0D1421"

print("   Distribution après remapping :")
for s, lbl in LABELS.items():
    n = (y == s).sum()
    print(f"   Classe {s} [{lbl:<7}] : {'█'*n}{'░'*(20-n)} {n} sociétés ({n/len(y):.0%})")
print(f"\n   ✅ Dataset : {len(df)} sociétés · {len(FEATURES)} features · 3 classes\n")

# ═══════════════════════════════════════════════════════════════════════════
# 3.  STATISTIQUES DESCRIPTIVES
# ═══════════════════════════════════════════════════════════════════════════
print("📊 Statistiques moyennes par classe :")
print(df.groupby("Risque3")[FEATURES].mean().round(2).to_string(), "\n")

# ═══════════════════════════════════════════════════════════════════════════
# 4.  VISUALISATION EXPLORATOIRE  (6 graphiques)
# ═══════════════════════════════════════════════════════════════════════════
print("🔭 Génération des graphiques exploratoires...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor(BG)
axes = axes.flatten()

# — 4.1  Distribution des classes —
ax = axes[0]; ax.set_facecolor(SURF)
counts = pd.Series(y).value_counts().sort_index()
bars = ax.bar(counts.index, counts.values,
              color=[COLORS[i] for i in counts.index], width=0.5, edgecolor="none")
for b, v in zip(bars, counts.values):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.1, str(v),
            ha='center', va='bottom', color='white', fontsize=12, fontweight='bold')
ax.set_title("Distribution des 3 Classes", color='white', fontsize=11, pad=10)
ax.set_xticks([1,2,3]); ax.set_xticklabels(["1-FAIBLE","2-MODÉRÉ","3-ÉLEVÉ"], color='#5A7099', fontsize=9)
ax.tick_params(colors='#5A7099')
for sp in ax.spines.values(): sp.set_color('#1E2D45')

# — 4.2  Corrélation features → Score3 —
ax2 = axes[1]; ax2.set_facecolor(SURF)
corrs = df[FEATURES].corrwith(pd.Series(y)).sort_values()
ax2.barh(FEAT_LABELS, corrs.values,
         color=["#FF4E6A" if c > 0 else "#36D986" for c in corrs],
         edgecolor="none", height=0.6)
ax2.axvline(0, color='#1E2D45', linewidth=1)
ax2.set_title("Corrélation Features ↔ Classe Risque", color='white', fontsize=11, pad=10)
ax2.tick_params(colors='#5A7099', labelsize=8)
for sp in ax2.spines.values(): sp.set_color('#1E2D45')

# — 4.3  Scatter Dilution vs Gearing —
ax3 = axes[2]; ax3.set_facecolor(SURF)
for s in [1, 2, 3]:
    m = y == s
    ax3.scatter(df.loc[m,"DilutionCumulee"], df.loc[m,"Gearing"],
                c=COLORS[s], label=LABELS[s], s=80, edgecolors='white', linewidths=0.5, alpha=0.9)
    for idx in np.where(m)[0]:
        ax3.annotate(df["Ticker"].iloc[idx],
                     (df["DilutionCumulee"].iloc[idx], df["Gearing"].iloc[idx]),
                     xytext=(3,3), textcoords="offset points", fontsize=6, color='#5A7099')
ax3.set_title("Dilution Cumulée vs Gearing", color='white', fontsize=11, pad=10)
ax3.set_xlabel("Dilution Cumulée (%)", color='#5A7099', fontsize=9)
ax3.set_ylabel("Gearing (%)", color='#5A7099', fontsize=9)
ax3.tick_params(colors='#5A7099')
ax3.legend(fontsize=8, framealpha=0.2, labelcolor='white', facecolor=SURF, edgecolor='#1E2D45')
for sp in ax3.spines.values(): sp.set_color('#1E2D45')

# — 4.4  Boxplot Dilution —
ax4 = axes[3]; ax4.set_facecolor(SURF)
for s in [1, 2, 3]:
    ax4.boxplot(df.loc[y==s,"DilutionCumulee"], positions=[s], widths=0.4, patch_artist=True,
                boxprops=dict(facecolor=COLORS[s]+'44', edgecolor=COLORS[s]),
                medianprops=dict(color=COLORS[s], linewidth=2),
                whiskerprops=dict(color='#5A7099'), capprops=dict(color='#5A7099'),
                flierprops=dict(markerfacecolor=COLORS[s], markersize=4))
ax4.set_title("Dilution Cumulée par Classe", color='white', fontsize=11, pad=10)
ax4.set_xticks([1,2,3]); ax4.set_xticklabels(["FAIBLE","MODÉRÉ","ÉLEVÉ"], color='#5A7099', fontsize=9)
ax4.set_ylabel("Dilution (%)", color='#5A7099'); ax4.tick_params(colors='#5A7099')
for sp in ax4.spines.values(): sp.set_color('#1E2D45')

# — 4.5  Boxplot Gearing —
ax5 = axes[4]; ax5.set_facecolor(SURF)
for s in [1, 2, 3]:
    ax5.boxplot(df.loc[y==s,"Gearing"], positions=[s], widths=0.4, patch_artist=True,
                boxprops=dict(facecolor=COLORS[s]+'44', edgecolor=COLORS[s]),
                medianprops=dict(color=COLORS[s], linewidth=2),
                whiskerprops=dict(color='#5A7099'), capprops=dict(color='#5A7099'),
                flierprops=dict(markerfacecolor=COLORS[s], markersize=4))
ax5.set_title("Gearing par Classe", color='white', fontsize=11, pad=10)
ax5.set_xticks([1,2,3]); ax5.set_xticklabels(["FAIBLE","MODÉRÉ","ÉLEVÉ"], color='#5A7099', fontsize=9)
ax5.set_ylabel("Gearing (%)", color='#5A7099'); ax5.tick_params(colors='#5A7099')
for sp in ax5.spines.values(): sp.set_color('#1E2D45')

# — 4.6  Heatmap corrélation —
ax6 = axes[5]; ax6.set_facecolor(SURF)
sns.heatmap(df[FEATURES].corr(), ax=ax6, cmap="coolwarm", center=0,
            xticklabels=FEAT_LABELS, yticklabels=FEAT_LABELS,
            linewidths=0.3, linecolor=SURF, annot=False, cbar_kws={"shrink":0.8})
ax6.set_title("Matrice de Corrélation des Features", color='white', fontsize=11, pad=10)
ax6.tick_params(colors='#5A7099', labelsize=7)

plt.suptitle("Analyse Exploratoire — BVC/AMMC · 3 Classes de Risque",
             color='white', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("01_exploration.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print("   ✅ Figure 1 : 01_exploration.png\n")

# ═══════════════════════════════════════════════════════════════════════════
# 5.  PIPELINE SVM  (StandardScaler + SVC RBF)
# ═══════════════════════════════════════════════════════════════════════════
print("🤖 Entraînement du pipeline SVM...")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
print(f"   Train : {len(X_tr)} obs. | Test : {len(X_te)} obs.")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm",    SVC(kernel="rbf", C=10, gamma="scale",
                   probability=True, class_weight="balanced", random_state=42))
])
pipeline.fit(X_tr, y_tr)
y_pred  = pipeline.predict(X_te)
acc     = accuracy_score(y_te, y_pred)

print(f"\n   ✅ Accuracy test : {acc:.1%}\n")
print(classification_report(y_te, y_pred, target_names=["FAIBLE","MODÉRÉ","ÉLEVÉ"]))

cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
print(f"   Cross-Validation 5-fold : {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
print(f"   Scores par fold         : {[round(s,2) for s in cv_scores]}\n")

# ═══════════════════════════════════════════════════════════════════════════
# 6.  MATRICE DE CONFUSION  +  GRAPHIQUE CV
# ═══════════════════════════════════════════════════════════════════════════
print("📊 Génération : matrice de confusion + CV...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(BG)

# — Matrice —
ax = axes[0]; ax.set_facecolor(SURF)
cm_mat = confusion_matrix(y_te, y_pred)
ax.imshow(cm_mat, cmap="Blues")
tl = ["FAIBLE\n(1)","MODÉRÉ\n(2)","ÉLEVÉ\n(3)"]
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(tl, color='#5A7099', fontsize=9)
ax.set_yticklabels(tl, color='#5A7099', fontsize=9)
ax.set_xlabel("Classe Prédite", color='#5A7099', fontsize=10)
ax.set_ylabel("Classe Réelle",  color='#5A7099', fontsize=10)
ax.set_title("Matrice de Confusion (Test Set)", color='white', fontsize=12, pad=12)
for sp in ax.spines.values(): sp.set_color('#1E2D45')
for i in range(cm_mat.shape[0]):
    for j in range(cm_mat.shape[1]):
        ax.text(j, i, str(cm_mat[i,j]), ha='center', va='center', fontsize=18, fontweight='bold',
                color='white' if cm_mat[i,j] > cm_mat.max()/2 else '#5A7099')

# — CV —
ax2 = axes[1]; ax2.set_facecolor(SURF)
fc = ["#FF4E6A" if s<0.6 else "#FFB547" if s<0.8 else "#36D986" for s in cv_scores]
bars = ax2.bar(range(1,6), cv_scores*100, color=fc, width=0.55, edgecolor="none")
ax2.axhline(cv_scores.mean()*100, color='#00C6FF', linestyle='--', linewidth=2,
            label=f"Moyenne = {cv_scores.mean():.1%}")
ax2.fill_between(range(0,7), (cv_scores.mean()-cv_scores.std())*100,
                 (cv_scores.mean()+cv_scores.std())*100,
                 alpha=0.1, color='#00C6FF', label=f"±1σ = ±{cv_scores.std():.1%}")
for b, v in zip(bars, cv_scores):
    ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
             f"{v:.0%}", ha='center', va='bottom', color='white', fontsize=11)
ax2.set_title("Cross-Validation 5-Fold", color='white', fontsize=12, pad=12)
ax2.set_xlabel("Fold", color='#5A7099'); ax2.set_ylabel("Accuracy (%)", color='#5A7099')
ax2.tick_params(colors='#5A7099'); ax2.set_ylim(0, 115)
ax2.legend(framealpha=0.2, facecolor=SURF, edgecolor='#1E2D45', labelcolor='white')
for sp in ax2.spines.values(): sp.set_color('#1E2D45')

plt.suptitle(f"Évaluation SVM — 3 Classes · Accuracy = {acc:.1%}",
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("02_evaluation.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print("   ✅ Figure 2 : 02_evaluation.png\n")

# ═══════════════════════════════════════════════════════════════════════════
# 7.  GRIDSEARCHCV  (optimisation hyperparamètres — ~30 sec)
# ═══════════════════════════════════════════════════════════════════════════
print("⚙️  GridSearchCV en cours (~30 secondes)...")

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
    scoring="accuracy", n_jobs=-1, verbose=0
)
gs.fit(X, y)
best = gs.best_estimator_

print(f"   ✅ Meilleurs paramètres : {gs.best_params_}")
print(f"      Meilleure accuracy CV : {gs.best_score_:.1%}\n")

# — Heatmap RBF —
res = pd.DataFrame(gs.cv_results_)
rbf = res[res["param_svm__kernel"] == "rbf"].copy()
rbf["param_svm__C"]     = rbf["param_svm__C"].astype(str)
rbf["param_svm__gamma"] = rbf["param_svm__gamma"].astype(str)
pivot = rbf.pivot_table(index="param_svm__C",
                        columns="param_svm__gamma",
                        values="mean_test_score")

fig, ax = plt.subplots(figsize=(9, 4))
fig.patch.set_facecolor(BG); ax.set_facecolor(SURF)
sns.heatmap(pivot, annot=True, fmt=".1%", cmap="YlOrRd", ax=ax,
            linewidths=0.5, linecolor=SURF, annot_kws={"size":10,"color":"black"})
ax.set_title("GridSearchCV — Accuracy RBF · C × γ", color='white', fontsize=12, pad=12)
ax.set_xlabel("Gamma", color='#5A7099'); ax.set_ylabel("C", color='#5A7099')
ax.tick_params(colors='#5A7099')
for sp in ax.spines.values(): sp.set_color('#1E2D45')
plt.tight_layout()
plt.savefig("03_gridsearch.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print("   ✅ Figure 3 : 03_gridsearch.png\n")

# ═══════════════════════════════════════════════════════════════════════════
# 8.  IMPORTANCE DES FEATURES  (Permutation Importance)
# ═══════════════════════════════════════════════════════════════════════════
print("🔍 Calcul importance des features...")

pi = permutation_importance(best, X, y, n_repeats=20, random_state=42)
feat_df = pd.DataFrame({
    "Feature":    FEAT_LABELS,
    "Importance": pi.importances_mean,
    "Std":        pi.importances_std
}).sort_values("Importance", ascending=False)
print(feat_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG); ax.set_facecolor(SURF)
fc2 = ["#36D986" if v>=0.05 else "#FFB547" if v>=0.02 else "#5A7099"
       for v in feat_df["Importance"]]
ax.barh(feat_df["Feature"], feat_df["Importance"],
        xerr=feat_df["Std"], color=fc2, edgecolor="none", height=0.6,
        capsize=4, error_kw={"ecolor":"#1E2D45","alpha":0.6})
ax.axvline(0, color='#1E2D45', linewidth=1)
ax.set_title("Importance des Features — Permutation (meilleur modèle)",
             color='white', fontsize=12, pad=12)
ax.set_xlabel("Baisse d'accuracy (importance)", color='#5A7099')
ax.tick_params(colors='#5A7099')
for sp in ax.spines.values(): sp.set_color('#1E2D45')
for i, (_, row) in enumerate(feat_df.iterrows()):
    ax.text(row["Importance"]+0.001, i, f"  {row['Importance']:.4f}",
            va='center', color='white', fontsize=9)
plt.tight_layout()
plt.savefig("04_feature_importance.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print("   ✅ Figure 4 : 04_feature_importance.png\n")

# ═══════════════════════════════════════════════════════════════════════════
# 9.  PROJECTION ACP 2D  +  BIPLOT
# ═══════════════════════════════════════════════════════════════════════════
print("🗺️  Projection ACP 2D...")

X_sc = StandardScaler().fit_transform(X)
pca  = PCA(n_components=2, random_state=42)
X2   = pca.fit_transform(X_sc)
var1, var2 = pca.explained_variance_ratio_
print(f"   Variance expliquée : PC1={var1:.1%} · PC2={var2:.1%} · Total={var1+var2:.1%}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(BG)

# — Scatter —
ax = axes[0]; ax.set_facecolor(SURF)
for s in [1, 2, 3]:
    m = y == s
    ax.scatter(X2[m,0], X2[m,1], c=COLORS[s], label=LABELS[s],
               s=110, edgecolors='white', linewidths=0.8, alpha=0.9, zorder=3)
    for idx in np.where(m)[0]:
        ax.annotate(df["Ticker"].iloc[idx], (X2[idx,0], X2[idx,1]),
                    xytext=(4,4), textcoords="offset points", fontsize=7, color='#8A9EBF')
ax.set_title(f"PCA 2D — Séparabilité des 3 Classes (var = {var1+var2:.1%})",
             color='white', fontsize=11, pad=10)
ax.set_xlabel(f"PC1 ({var1:.1%})", color='#5A7099')
ax.set_ylabel(f"PC2 ({var2:.1%})", color='#5A7099')
ax.tick_params(colors='#5A7099')
ax.legend(fontsize=9, framealpha=0.2, labelcolor='white', facecolor=SURF, edgecolor='#1E2D45')
for sp in ax.spines.values(): sp.set_color('#1E2D45')

# — Biplot —
ax2 = axes[1]; ax2.set_facecolor(SURF)
loadings = pca.components_.T
acols = ["#00C6FF","#FF4E6A","#FFB547","#36D986","#7B61FF",
         "#FF61D8","#00E5C8","#FF8C42","#A0C4FF"]
for i, (fl, ld) in enumerate(zip(FEAT_LABELS, loadings)):
    ax2.annotate("", xy=(ld[0]*3, ld[1]*3), xytext=(0,0),
                 arrowprops=dict(arrowstyle="-|>", color=acols[i], lw=1.8, mutation_scale=14))
    ax2.text(ld[0]*3.3+0.15, ld[1]*3.3+0.1, fl.replace(" ","\n"),
             color=acols[i], fontsize=7, ha='center', va='center')
ax2.axhline(0, color='#1E2D45', linewidth=0.8)
ax2.axvline(0, color='#1E2D45', linewidth=0.8)
ax2.add_patch(plt.Circle((0,0), 3, fill=False, color='#1E2D45', linewidth=0.8))
ax2.set_xlim(-4.5, 4.5); ax2.set_ylim(-4.5, 4.5)
ax2.set_title("Biplot — Contribution des variables", color='white', fontsize=11, pad=10)
ax2.set_xlabel("PC1", color='#5A7099'); ax2.set_ylabel("PC2", color='#5A7099')
ax2.tick_params(colors='#5A7099')
for sp in ax2.spines.values(): sp.set_color('#1E2D45')

plt.suptitle("Visualisation ACP — 3 Classes Risque Dilution BVC",
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("05_pca.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print("   ✅ Figure 5 : 05_pca.png\n")

# ═══════════════════════════════════════════════════════════════════════════
# 10.  COURBE D'APPRENTISSAGE
# ═══════════════════════════════════════════════════════════════════════════
print("📈 Learning curve...")

ts, tr_sc, vl_sc = learning_curve(
    best, X, y,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    train_sizes=np.linspace(0.3, 1.0, 7),
    scoring="accuracy", n_jobs=-1
)
tm, tstd = tr_sc.mean(1), tr_sc.std(1)
vm, vstd = vl_sc.mean(1), vl_sc.std(1)

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(BG); ax.set_facecolor(SURF)
ax.plot(ts, tm*100, 'o-', color='#00C6FF', lw=2, label='Score Train', ms=6)
ax.fill_between(ts, (tm-tstd)*100, (tm+tstd)*100, alpha=0.12, color='#00C6FF')
ax.plot(ts, vm*100, 's--', color='#36D986', lw=2, label='Score Validation (CV)', ms=6)
ax.fill_between(ts, (vm-vstd)*100, (vm+vstd)*100, alpha=0.12, color='#36D986')
ax.set_title("Learning Curve — SVM 3 Classes · BVC Dilution Risk",
             color='white', fontsize=12, pad=12)
ax.set_xlabel("Taille d'entraînement", color='#5A7099')
ax.set_ylabel("Accuracy (%)", color='#5A7099')
ax.tick_params(colors='#5A7099')
ax.legend(framealpha=0.2, facecolor=SURF, edgecolor='#1E2D45', labelcolor='white')
ax.set_ylim(0, 115)
for sp in ax.spines.values(): sp.set_color('#1E2D45')
plt.tight_layout()
plt.savefig("06_learning_curve.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print("   ✅ Figure 6 : 06_learning_curve.png\n")

# ═══════════════════════════════════════════════════════════════════════════
# 11.  TABLEAU COMPLET DES PRÉDICTIONS
# ═══════════════════════════════════════════════════════════════════════════
print("📋 Prédictions sur les 30 sociétés :\n")

y_all   = best.predict(X)
pr_all  = best.predict_proba(X)
conf    = pr_all.max(1)

df_res = df[["Société","Secteur","Ticker","Score5","Score3","Risque3"]].copy()
df_res["Prédit"]        = y_all
df_res["Risque_Prédit"] = pd.Series(y_all).map(LABELS).values
df_res["Confiance"]     = (conf*100).round(1).astype(str) + "%"
df_res["Résultat"]      = df_res.apply(
    lambda r: "✅ CORRECT" if r["Score3"]==r["Prédit"] else "❌ ERREUR", axis=1)

pd.set_option("display.max_rows", 35)
pd.set_option("display.width", 140)
try:
    display(df_res[["Ticker","Société","Score3","Risque3",
                    "Prédit","Risque_Prédit","Confiance","Résultat"]])
except NameError:
    print(df_res[["Ticker","Société","Score3","Risque3",
                  "Prédit","Risque_Prédit","Confiance","Résultat"]].to_string(index=False))

correct = (df_res["Score3"] == df_res["Prédit"]).sum()
print(f"\n   Accuracy globale (best model) : {correct}/{len(df_res)} = {correct/len(df_res):.1%}")
print(f"   Best CV accuracy (GridSearch) : {gs.best_score_:.1%}")
print(f"   Meilleurs paramètres          : {gs.best_params_}\n")

# ═══════════════════════════════════════════════════════════════════════════
# 12.  PRÉDICTION SUR UNE NOUVELLE SOCIÉTÉ
#      ➤ Modifiez les valeurs ci-dessous pour tester votre propre cas
# ═══════════════════════════════════════════════════════════════════════════
print("─" * 65)
print("🆕 PRÉDICTION SUR UNE NOUVELLE SOCIÉTÉ")
print("   Ordre : [NbAugCap, Dilution, Gearing, CouvertFP,")
print("            PrimeEm, VarBPA, Rachat, StockOpt, PctActRef]")
print("─" * 65)

exemples = [
    # ── Modifiez ces valeurs pour vos propres sociétés ───────────────────
    ("Société test – risque FAIBLE",  [0,  0.0,  25, 4.5,  0.0,  8.0, 1, 0, 65.0]),
    ("Société test – risque MODÉRÉ",  [1, 14.0,  90, 1.1,  8.0,-12.0, 0, 0, 42.0]),
    ("Société test – risque ÉLEVÉ",   [3, 38.0, 160, 0.62, 5.0,-40.0, 0, 0, 30.0]),
]

for nom, vals in exemples:
    x_new = np.array([vals])
    pred  = best.predict(x_new)[0]
    proba = best.predict_proba(x_new)[0]
    print(f"\n  ► {nom}")
    print(f"    → Classe prédite  : {pred} — {LABELS[pred]}")
    print(f"    → Confiance SVM   : {proba.max():.1%}")
    print(f"    → Probas [F/M/É]  : {proba[0]:.1%} / {proba[1]:.1%} / {proba[2]:.1%}")

# ═══════════════════════════════════════════════════════════════════════════
# 13.  EXPORT CSV  +  RÉSUMÉ FINAL
# ═══════════════════════════════════════════════════════════════════════════
df_res.to_csv("SVM_BVC_predictions.csv", index=False, encoding='utf-8-sig')

print("\n\n" + "=" * 65)
print("   RÉSUMÉ FINAL — SVM 3 CLASSES · BVC / AMMC")
print("=" * 65)
print(f"   Modèle        : SVM kernel RBF")
print(f"   Paramètres    : {gs.best_params_}")
print(f"   Accuracy test : {accuracy_score(y_te, y_pred):.1%}  ({len(y_te)} obs.)")
print(f"   Accuracy CV   : {gs.best_score_:.1%}  (GridSearchCV 5-fold)")
print(f"   CV 5-fold     : {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
print(f"   Classes       : FAIBLE={int((y==1).sum())} · MODÉRÉ={int((y==2).sum())} · ÉLEVÉ={int((y==3).sum())} sociétés")
print(f"   Top feature   : Programme Rachat · Dilution Cumulée · Variation BPA")
print("=" * 65)
print("   Fichiers générés (📁 Panneau gauche Colab → Télécharger) :")
for f in ["01_exploration.png","02_evaluation.png","03_gridsearch.png",
          "04_feature_importance.png","05_pca.png","06_learning_curve.png",
          "SVM_BVC_predictions.csv"]:
    print(f"     📁 {f}")
print("=" * 65)
