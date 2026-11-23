"""Familles de causes : regroupe les libellés de cause de chaque agence (italien, allemand, norvégien, suédois,
tchèque, anglais) en cinq familles comparables entre pays.

Le premier motif qui correspond l'emporte. « Non renseignée » veut dire que la source ne donne pas de cause
(France, Slovaquie, 60 % de la Suède) : à ne pas confondre avec « Autre » (cause donnée mais non classée, par
exemple le « Sonstige » allemand). Les colonnes brutes (`raison`) restent dans `shortage_events` : le regroupement
peut être refait sans reconstruire les données.
"""
import re

import pandas as pd

CAUSES = ["Production / qualité", "Demande en hausse", "Distribution / capacité", "Arrêt / raison commerciale",
          "Réglementaire / sécurité", "Autre", "Non renseignée"]

FAMILLES = [
    ("Demande en hausse", r"richiesta|økt salg|økt etterspørsel|etterspørsel|demand|increased|poptávk|zvýšen|spotřeb"),
    ("Production / qualité", r"produtt|produk|produc|výrob|manufactur|quality|kvalit|råvare|frigivelse|gmp"),
    ("Distribution / capacité", r"distribu|capacit|kapasitet|levering|deliver|logist|forniture|transport"),
    ("Arrêt / raison commerciale", r"commerci|cessat|withdraw|revocat|lavt salg|marketing|switch|reimburs|"
                                   r"sospensione|avvenuta|endring|discontinu|avregistr|utgått|inventur|"
                                   r"nízké odběry|price|přeprodej|přechod|stažení"),
    ("Réglementaire / sécurité", r"regolator|regulat|procedur|regulator|myndighet|safety|correction"),
]


def famille_cause(raison):
    if not isinstance(raison, str) or not raison.strip():
        return "Non renseignée"
    texte = raison.lower()
    for nom, motif in FAMILLES:
        if re.search(motif, texte):
            return nom
    return "Autre"


def type_cause():
    return pd.CategoricalDtype(CAUSES)
