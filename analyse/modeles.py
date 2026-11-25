"""Étape 6 : référence à battre, modèles, et test de Q1 (cascade) et Q2 (Italie) sur `panel_fr`.

Usage :
    python -m nettoyage panel      # construit data/final/panel_fr.parquet
    python -m analyse.modeles      # modèles dans models/, métriques et prédictions dans outputs/

Protocole (EXPLICATIONS.md, section 6, étapes 6 et 8) :
- entraînement sur le jeu `entrainement`, choix du modèle sur `validation`, test (2025) lu une seule fois
  pour le modèle retenu et les comparaisons Q1/Q2 ;
- références à battre : « déjà affichée en pénurie sur la page ANSM » et « signalée l'année précédente » ;
- modèles : régression logistique (variables standardisées) et gradient boosting sur arbres
  (HistGradientBoosting, qui accepte les valeurs vides : un mois non couvert reste vide, pas un zéro) ;
- métriques pour des classes déséquilibrées : PR-AUC (précision moyenne), ROC-AUC, et, mois par mois,
  rappel et précision parmi les k substances jugées les plus à risque ;
- Q1 et Q2 : le même modèle avec et sans la variable, sur les mêmes lignes (celles où la source est
  couverte). L'écart est accompagné d'un intervalle à 95 % par bootstrap apparié sur les substances
  (les lignes d'une même substance ne sont pas indépendantes).
"""
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RACINE = Path(__file__).resolve().parent.parent
PANEL = RACINE / "data" / "final" / "panel_fr.parquet"
MODELES = RACINE / "models"
SORTIES = RACINE / "outputs"
K = (50, 100)  # substances les plus à risque, chaque mois (≈ 1 230 substances, ≈ 340 positives par mois)
TIRAGES = 500  # bootstrap apparié par substance
GRAINE = 0

# Variables de base : connues pour toute la cohorte (vides seulement avant la couverture de la source)
BASE = ["log_conso", "log_presentations", "conso_absente", "ue_critique", "log_sig_an_prec", "log_sig_cumul",
        "sig_absent", "mois_num", "hiver"]
FRANCE = ["fr_affichee_3m"]  # page ANSM : à partir de 06/2021
Q1 = ["fr_autres_classe_3m"]  # cascade : autres substances de la même ATC4 affichées
Q2_IT = ["it_affichee_3m"]  # Italie : à partir de 12/2021


def preparer(panel):
    """Variables du modèle à partir de `panel_fr` : logarithmes des comptes, indicateurs de valeur absente."""
    t = panel[panel["jeu"] != "ecart"].copy()
    t["jeu"] = t["jeu"].astype(str)
    conso = t["conso_boites_an_prec"].astype("float64").clip(lower=0)  # une substance-année est négative (remboursements nets)
    t["log_conso"] = np.log1p(conso)
    t["log_presentations"] = np.log1p(t["conso_presentations_an_prec"].astype("float64"))
    t["conso_absente"] = conso.isna().astype("int8")
    t["log_sig_an_prec"] = np.log1p(t["sig_an_prec"].astype("float64"))
    t["log_sig_cumul"] = np.log1p(t["sig_cumul_avant"].astype("float64"))
    t["sig_absent"] = t["sig_an_prec"].isna().astype("int8")  # 2019 : pas d'année précédente
    t["hiver"] = t["mois_num"].isin([11, 12, 1, 2]).astype("int8")
    for c in FRANCE + Q1 + Q2_IT + ["de_affichee_3m"]:
        t[c] = t[c].astype("float64")
    return t


def logistique():
    # Les vides (consommation, signalements) sont remplacés par 0 ; l'indicateur d'absence garde l'information.
    return make_pipeline(SimpleImputer(strategy="constant", fill_value=0), StandardScaler(),
                         LogisticRegression(max_iter=2000))


def boosting():
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=100,
                                          l2_regularization=1.0, early_stopping=False, random_state=GRAINE)


FAMILLES = {"logistique": logistique, "boosting": boosting}


def rappel_top_k(donnees, score, k):
    """Chaque mois, parmi les k substances au score le plus élevé : part des positives trouvées (rappel) et
    part des k qui sont positives (précision). Moyenne sur les mois."""
    d = donnees[["mois", "cible"]].assign(score=score)
    rappels, precisions = [], []
    for _, g in d.groupby("mois"):
        haut = g.nlargest(k, "score")
        positives = g["cible"].sum()
        if positives:
            rappels.append(haut["cible"].sum() / positives)
        precisions.append(haut["cible"].mean())
    return float(np.mean(rappels)), float(np.mean(precisions))


def mesurer(donnees, score):
    m = {"pr_auc": float(average_precision_score(donnees["cible"], score)),
         "roc_auc": float(roc_auc_score(donnees["cible"], score)), "taux_positifs": float(donnees["cible"].mean())}
    for k in K:
        m[f"rappel_top{k}"], m[f"precision_top{k}"] = rappel_top_k(donnees, score, k)
    return m


def entrainer(famille, donnees, colonnes):
    entrainement = donnees[donnees["jeu"] == "entrainement"]
    return FAMILLES[famille]().fit(entrainement[colonnes], entrainement["cible"])


def scorer(modele, donnees, colonnes):
    return modele.predict_proba(donnees[colonnes])[:, 1]


def references(donnees):
    """Règles simples, sans apprentissage. Le petit terme ajouté départage les ex aequo par l'historique long."""
    sig = donnees["sig_an_prec"].fillna(0).to_numpy()
    cumul = donnees["sig_cumul_avant"].fillna(0).to_numpy()
    regles = {"ref_signalee_annee_prec": sig + 1e-3 * cumul}
    if donnees["fr_affichee_3m"].notna().all():
        regles["ref_affichee_3m"] = donnees["fr_affichee_3m"].to_numpy() + 1e-3 * sig
    return regles


def bootstrap_ecart(donnees, score_a, score_b, metrique=average_precision_score):
    """Écart de métrique (b − a) et intervalle à 95 %, en tirant les substances avec remise."""
    rng = np.random.default_rng(GRAINE)
    substances = donnees["atc5"].to_numpy()
    uniques, code = np.unique(substances, return_inverse=True)
    indices = [np.where(code == i)[0] for i in range(len(uniques))]
    y = donnees["cible"].to_numpy()
    ecarts = []
    for _ in range(TIRAGES):
        lignes = np.concatenate([indices[i] for i in rng.integers(0, len(uniques), len(uniques))])
        ecarts.append(metrique(y[lignes], score_b[lignes]) - metrique(y[lignes], score_a[lignes]))
    return {"ecart": float(metrique(y, score_b) - metrique(y, score_a)),
            "ic95": [float(np.percentile(ecarts, 2.5)), float(np.percentile(ecarts, 97.5))]}


def comparer(donnees, sans, avec, famille, nom):
    """Q1 ou Q2 : même famille de modèle, mêmes lignes, avec ou sans la variable testée."""
    m_sans, m_avec = entrainer(famille, donnees, sans), entrainer(famille, donnees, avec)
    resultat = {"lignes_entrainement": int((donnees["jeu"] == "entrainement").sum()),
                "debut": str(donnees["mois"].min().date())}
    for jeu in ("validation", "test"):
        d = donnees[donnees["jeu"] == jeu]
        s_sans, s_avec = scorer(m_sans, d, sans), scorer(m_avec, d, avec)
        resultat[jeu] = {"sans": mesurer(d, s_sans), "avec": mesurer(d, s_avec),
                         "ecart_pr_auc": bootstrap_ecart(d, s_sans, s_avec)}
    print(f"  {nom:22} {famille:10} PR-AUC test {resultat['test']['sans']['pr_auc']:.3f} → "
          f"{resultat['test']['avec']['pr_auc']:.3f}  écart {resultat['test']['ecart_pr_auc']['ecart']:+.4f} "
          f"[{resultat['test']['ecart_pr_auc']['ic95'][0]:+.4f} ; {resultat['test']['ecart_pr_auc']['ic95'][1]:+.4f}]")
    return resultat


def importances(modele, donnees, colonnes):
    """Importance par permutation sur la validation (baisse de PR-AUC quand la variable est mélangée)."""
    from sklearn.inspection import permutation_importance
    d = donnees[donnees["jeu"] == "validation"]
    r = permutation_importance(modele, d[colonnes], d["cible"], scoring="average_precision", n_repeats=5,
                               random_state=GRAINE)
    return dict(sorted(zip(colonnes, map(float, r.importances_mean)), key=lambda x: -x[1]))


def main():
    t = preparer(pd.read_parquet(PANEL))
    france = t.dropna(subset=FRANCE)  # page ANSM couverte : 06/2021 →
    italie = t.dropna(subset=Q2_IT)  # Italie couverte : 12/2021 →
    resultats = {"date": date.today().isoformat(), "panel": str(PANEL.relative_to(RACINE)),
                 "lignes": {jeu: int((t["jeu"] == jeu).sum()) for jeu in ("entrainement", "validation", "test")}}

    # 1. Références et modèles, sur les lignes où la page ANSM est couverte (même périmètre pour tous)
    print("Validation (choix du modèle), lignes 06/2021 → :")
    candidats = {"base": BASE, "base+france": BASE + FRANCE}
    validation = france[france["jeu"] == "validation"]
    resultats["validation"] = {nom: mesurer(validation, s) for nom, s in references(validation).items()}
    modeles = {}
    for famille in FAMILLES:
        for nom_col, colonnes in candidats.items():
            nom = f"{famille}:{nom_col}"
            modeles[nom] = entrainer(famille, france, colonnes)
            resultats["validation"][nom] = mesurer(validation, scorer(modeles[nom], validation, colonnes))
    for nom, m in resultats["validation"].items():
        print(f"  {nom:26} PR-AUC {m['pr_auc']:.3f}  ROC-AUC {m['roc_auc']:.3f}  "
              f"rappel top100 {m['rappel_top100']:.3f}  précision top100 {m['precision_top100']:.3f}")

    appris = {n: m for n, m in resultats["validation"].items() if ":" in n}
    retenu = max(appris, key=lambda n: appris[n]["pr_auc"])
    famille, nom_col = retenu.split(":")
    colonnes = candidats[nom_col]
    resultats["retenu"] = {"nom": retenu, "variables": colonnes}

    # 2. Test, lu une fois : modèle retenu face aux références
    test = france[france["jeu"] == "test"]
    resultats["test"] = {nom: mesurer(test, s) for nom, s in references(test).items()}
    score_test = scorer(modeles[retenu], test, colonnes)
    resultats["test"][retenu] = mesurer(test, score_test)
    print(f"\nTest 2025 (modèle retenu : {retenu}) :")
    for nom, m in resultats["test"].items():
        print(f"  {nom:26} PR-AUC {m['pr_auc']:.3f}  ROC-AUC {m['roc_auc']:.3f}  "
              f"rappel top100 {m['rappel_top100']:.3f}  précision top100 {m['precision_top100']:.3f}")
    resultats["importances_validation"] = importances(modeles[retenu], france, colonnes)

    # 3. Q1 et Q2 : avec et sans la variable, pour les deux familles
    print("\nQ1 et Q2 (même modèle avec et sans la variable ; écart de PR-AUC sur le test, IC 95 % par substance) :")
    resultats["questions"] = {}
    for fam in FAMILLES:
        resultats["questions"][f"Q1 cascade | {fam}"] = comparer(france, BASE + FRANCE, BASE + FRANCE + Q1, fam, "Q1 cascade")
        resultats["questions"][f"Q2 Italie | {fam}"] = comparer(italie, BASE + FRANCE, BASE + FRANCE + Q2_IT, fam, "Q2 Italie")
    resultats["questions"]["Q2 Allemagne"] = ("non testable : la source ne couvre que 13 mois (09/2024 →), "
                                             "absents de l'entraînement")

    # 4. Rangement (étape 8) : modèle retenu, métriques, prédictions de validation et de test
    MODELES.mkdir(exist_ok=True)
    SORTIES.mkdir(exist_ok=True)
    jour = date.today().isoformat()
    chemin_modele = MODELES / f"{famille}_{nom_col.replace('+', '_')}_{jour}.joblib"
    joblib.dump({"modele": modeles[retenu], "variables": colonnes, "entraine_sur": "entrainement 06/2021 → 09/2023",
                 "panel": resultats["panel"]}, chemin_modele)
    predictions = pd.concat([d.assign(score=scorer(modeles[retenu], d, colonnes))
                             for d in (validation, test)])[["atc5", "mois", "jeu", "cible", "score"]]
    predictions.to_parquet(SORTIES / f"predictions_{jour}.parquet", index=False)
    (SORTIES / f"metriques_{jour}.json").write_text(json.dumps(resultats, indent=2, ensure_ascii=False))
    print(f"\nModèle : {chemin_modele.relative_to(RACINE)} ; métriques et prédictions : outputs/*_{jour}.*")


if __name__ == "__main__":
    main()
