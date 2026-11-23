"""Comparaison de familles de modèles sur la cible « nouvelles pénuries » (substances calmes depuis 12 mois).

Même protocole que analyse/faisabilite.py::nouvelles_penuries : mêmes variables, découpage temporel
(entraînement → validation → test 2025). Hyperparamètres choisis sur la validation, puis modèle réentraîné
sur entraînement + validation et évalué une seule fois sur le test. IC 95 % par rééchantillonnage des substances.
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
PANEL, K = sys.argv[1], 12

t = pd.read_parquet(PANEL)
t = t[(t[f"calme_{K}m"] == 1) & t["jeu"].isin(["entrainement", "validation", "test"])].copy()
y = f"cible_nouvelle_{K}m"
for entree, sortie in [("conso_6m_pub", "conso"), ("sig_annee_pub", "sig"), ("sig_cumul_pub", "sig_cumul")]:
    t[sortie] = np.log1p(t[entree].astype("float64"))
t["tendance"] = np.log(t["conso_tendance_6m"].astype("float64"))
t["saison"] = np.log(t["conso_saison_cible"].astype("float64"))
europe = ["it_affichee_3m", "it_couvert", "se_nouvelle_3m", "cz_nouvelle_3m", "sk_nouvelle_3m"]
for c in europe + ["fr_affichee_3m", "fr_autres_classe_3m", "ue_critique", "mois_num"]:
    t[c] = t[c].astype("float64")
base = ["conso", "tendance", "saison", "ue_critique", "sig", "sig_cumul", "fr_affichee_3m"]
tr, va, te = (t[t["jeu"] == j] for j in ["entrainement", "validation", "test"])
trva = pd.concat([tr, va])

GRILLES = {
    "Logistic regression": [lambda C=C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
                            for C in [0.01, 0.1, 1.0]],
    "Logistic regression (class-weighted)": [lambda C=C: make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
                                              for C in [0.01, 0.1, 1.0]],
    "Random forest": [lambda d=d, m=m: RandomForestClassifier(n_estimators=500, max_depth=d, min_samples_leaf=m, n_jobs=-1, random_state=0)
                      for d in [4, 8, None] for m in [20, 100]],
    "Gradient boosting (HistGB)": [lambda lr=lr, l=l: HistGradientBoostingClassifier(learning_rate=lr, max_leaf_nodes=l, max_iter=300,
                                                                                    early_stopping=False, l2_regularization=1.0, random_state=0)
                                   for lr in [0.03, 0.1] for l in [7, 15, 31]],
    "MLP": [lambda a=a, h=h: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=h, alpha=a, max_iter=500,
                                                                         early_stopping=True, random_state=0))
            for a in [1e-3, 1e-1] for h in [(32,), (64, 32)]],
}


def boot_ci(df, score, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    subs = df["atc5"].unique()
    idx = {s: np.where(df["atc5"].values == s)[0] for s in subs}
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx[s] for s in rng.choice(subs, len(subs))])
        yy = df[y].values[pick]
        if yy.min() == yy.max():
            continue
        vals.append(roc_auc_score(yy, score[pick]))
    return np.percentile(vals, [2.5, 97.5])


resultats, scores = [], {}
print(f"test 2025 : {len(te)} lignes, {int(te[y].sum())} positives, taux {te[y].mean():.3f}")
# référence : persistance (historique de signalements seul)
for nom, s in [("Baseline: cumulative past reports", te["sig_cumul"].values)]:
    resultats.append((nom, "-", roc_auc_score(te[y], s), average_precision_score(te[y], s), boot_ci(te, s)))
for nom_feat, feats in [("base", base), ("base+europe", base + europe)]:
    for nom, grille in GRILLES.items():
        best, best_auc = None, -1
        for f in grille:
            m = f().fit(tr[feats], tr[y])
            a = roc_auc_score(va[y], m.predict_proba(va[feats])[:, 1])
            if a > best_auc:
                best, best_auc = f, a
        m = best().fit(trva[feats], trva[y])
        p = m.predict_proba(te[feats])[:, 1]
        scores[(nom, nom_feat)] = p
        resultats.append((nom, nom_feat, roc_auc_score(te[y], p), average_precision_score(te[y], p), boot_ci(te, p)))
        print(f"  {nom:38} {nom_feat:12} val {best_auc:.3f}", flush=True)

print(f"\n{'model':38} {'features':12} {'AUC':>6} {'PR-AUC':>7}  95% CI (AUC)")
for nom, f, a, pr, ci in resultats:
    print(f"{nom:38} {f:12} {a:6.3f} {pr:7.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]")
