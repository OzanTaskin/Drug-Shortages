"""Découpages temporels de `panel_fr` pour évaluer un modèle.

Le découpage fixe de la colonne `jeu` (entraînement → validation → test 2025) donne un seul test : 9 mois et
environ 120 nouvelles ruptures distinctes, donc des intervalles de confiance larges (AUC de 0,63 à 0,74). La
validation glissante (rolling origin) multiplie les fenêtres de test : le modèle est entraîné sur tout le passé,
avec trois mois d'écart (la cible regarde trois mois en avant), puis évalué sur les six mois suivants, et on
recommence en avançant de six mois. On rapporte la médiane et la dispersion des fenêtres, jamais une seule.

    from analyse.decoupage import decoupage_glissant
    for fenetre in decoupage_glissant(panel):
        entrainement, test = panel[fenetre["entrainement"]], panel[fenetre["test"]]
"""
import pandas as pd

HORIZON = 3  # mois de cible en avant : écart minimal entre la fin de l'entraînement et le début du test


def decoupage_glissant(panel, premier_test="2022-01", duree_test=6, pas=6, ecart=HORIZON):
    """Fenêtres successives : {nom, entrainement, test} (masques booléens alignés sur `panel`).

    Une ligne d'entraînement à t a sa cible sur t+1…t+3 : on garde `ecart` mois entre le dernier t d'entraînement
    et le premier t de test, pour qu'aucune cible d'entraînement ne recouvre le test.
    """
    mois = panel["mois"]
    debut, dernier = pd.Period(premier_test, "M"), mois.max().to_period("M")
    fenetres = []
    while debut + duree_test - 1 <= dernier:
        fin = debut + duree_test - 1
        dernier_entrainement = debut - ecart - 1
        masque_test = (mois >= debut.to_timestamp()) & (mois <= fin.to_timestamp())
        masque_entrainement = mois <= dernier_entrainement.to_timestamp()
        fenetres.append({"nom": f"{debut}→{fin}", "entrainement": masque_entrainement, "test": masque_test})
        debut += pas
    return fenetres
