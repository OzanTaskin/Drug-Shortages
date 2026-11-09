"""Slovaquie — déclarations de première mise sur le marché, d'interruption, de reprise et d'arrêt (ŠÚKL).

Le portail public du ŠÚKL (« Hlásenie o prerušení ») liste toutes les déclarations des titulaires depuis
04/2019 et les exporte en CSV : date de dépôt, titulaire, code ŠÚKL du produit, nom, date d'effet, objet
(U première mise sur le marché, R interruption, O reprise, Z arrêt). L'export n'a pas d'ATC : on le reprend de
la liste des médicaments du ŠÚKL (code ŠÚKL → ATC), dans sa version actuelle et dans les archives annuelles
2019 et 2022, qui contiennent les produits retirés depuis.
"""
from datetime import date

from .commun import telecharger

EXPORT = "https://portal.sukl.sk/PreruseniePublic/?act=PrerusenieOznList&export=csv"
LISTES = {  # nom du fichier → adresse
    "zoznam_liekov_aktualny.zip": "https://www.sukl.sk/verejne/Zoznam_liekov/zoznam_liekov_aktualny.zip",
    "Lieky2019.zip": "https://www.sukl.sk/verejne/Archiv/Lieky2019.zip",
    "Lieky2022.zip": "https://www.sukl.sk/verejne/Archiv/Lieky2022.zip",
}


def est_un_zip(contenu):
    return contenu[:4] == b"PK\x03\x04"


def est_le_csv(contenu):
    return contenu.lstrip(b"\xef\xbb\xbf")[:7] == b"Podanie"


def collecter(session):
    jour = date.today().isoformat()
    echecs = []
    if not telecharger(session, EXPORT, f"sukl_sk/{jour}_hlasenia_prerusenia.csv", "sukl_sk", timeout=600,
                       valider=est_le_csv):
        echecs.append("sukl_sk/hlasenia_prerusenia.csv")
    for nom, adresse in LISTES.items():
        # La liste actuelle change chaque mois : datée ; les archives annuelles ne changent plus.
        destination = f"sukl_sk/{jour}_{nom}" if "aktualny" in nom else f"sukl_sk/{nom}"
        if not telecharger(session, adresse, destination, "sukl_sk", timeout=900, valider=est_un_zip):
            echecs.append(destination)
    print(f"  sukl_sk : {'ok' if not echecs else 'ÉCHEC ' + ', '.join(echecs)}")
    return echecs
