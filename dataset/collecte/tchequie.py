"""Tchéquie — déclarations de mise sur le marché, d'interruption, de reprise et d'arrêt (SÚKL, « Market report »).

Les titulaires doivent déclarer au SÚKL chaque interruption des livraisons (au moins deux mois avant, ou au
moment même) et chaque reprise. Le SÚKL publie toutes les déclarations depuis 2009 dans le « Market report »
(prehledy.sukl.cz), avec le code SÚKL du produit, l'ATC, la date de déclaration, la date d'effet, la date de
reprise prévue, la cause et le produit de remplacement. On les lit par l'interface du site (POST JSON),
100 déclarations par page au plus, par type et par année de déclaration, triées par date de déclaration.

Une photo par jour de collecte : data/raw/sukl/<date>/<type>_<année>_p<page>.json. Les années passées ne
changent presque plus ; l'année en cours, si.
"""
import json
import time
from datetime import date

from .commun import RAW, enregistrer

API = "https://prehledy.sukl.cz/prehledy/v1/marketreport"
TYPES = ["preruseni", "obnoveni", "ukonceni"]  # interruption, reprise, arrêt définitif
PREMIERE_ANNEE = 2018
PAR_PAGE = 100  # maximum accepté par l'interface
PAUSE = 1.0


def est_une_page(contenu):
    try:
        return "data" in json.loads(contenu)
    except ValueError:
        return False


def collecter(session):
    jour = date.today().isoformat()
    echecs = []
    for type_ in TYPES:
        for annee in range(PREMIERE_ANNEE, date.today().year + 1):
            page, total = 1, None
            while total is None or (page - 1) * PAR_PAGE < total:
                destination = f"sukl/{jour}/{type_}_{annee}_p{page:03d}.json"
                chemin = RAW / destination
                if chemin.exists():
                    contenu = chemin.read_bytes()
                else:
                    requete = {"typOmezeni": type_, "jenPosledni": False, "pocet": PAR_PAGE, "stranka": page,
                               "sort": ["datum_hlaseni"], "smer": "asc",
                               "datumHlaseniMin": f"{annee}-01-01", "datumHlaseniMax": f"{annee}-12-31"}
                    try:
                        reponse = session.post(API, json=requete, timeout=120)
                        reponse.raise_for_status()
                    except Exception as erreur:  # noqa: BLE001 — on note l'échec et on passe à la suite
                        print(f"  ÉCHEC sukl {type_} {annee} p{page} : {erreur}")
                        echecs.append(destination)
                        break
                    if not est_une_page(reponse.content):
                        print(f"  ÉCHEC sukl {type_} {annee} p{page} : réponse inattendue")
                        echecs.append(destination)
                        break
                    enregistrer(reponse, destination, "sukl")
                    contenu = reponse.content
                    time.sleep(PAUSE)
                total = json.loads(contenu)["celkem"]
                page += 1
            print(f"  sukl {type_} {annee} : {total} déclarations")
    return echecs
