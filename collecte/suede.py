"""Suède — interruptions de vente déclarées à l'Agence suédoise des médicaments (Läkemedelsverket), en données ouvertes.

Deux fichiers XML (schéma « medicine shortages 3.0 »), mis à jour chaque nuit : les interruptions en cours et
l'historique des interruptions terminées, depuis 2018. Chaque interruption a sa date de première déclaration,
sa date de première publication, ses dates de début et de fin (prévues, puis réelles), sa cause et, pour chaque
produit, le numéro NPL et le code ATC.
Page de présentation : https://www.lakemedelsverket.se/sv/om-webbplatsen/oppna-data/anmalda-forsaljningsuppehall-av-lakemedel-som-oppna-data
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from .commun import telecharger

BASE = "https://docetp.mpa.se/LMF/Reports/"
FICHIERS = ["opendata-medicine-shortages-current-3-0.xml", "opendata-medicine-shortages-historic-3-0.xml"]


def est_du_xml(contenu):
    return contenu.lstrip()[:5] == b"<?xml"


def collecter(session):
    jour = date.today().isoformat()

    def un_fichier(nom):
        chemin = telecharger(session, BASE + nom, f"lakemedelsverket/{jour}_{nom}", "lakemedelsverket",
                             timeout=900, valider=est_du_xml)
        print(f"  lakemedelsverket {nom} {'ok' if chemin else 'ÉCHEC'}")
        return None if chemin else nom

    with ThreadPoolExecutor(max_workers=2) as executeur:
        return [nom for nom in executeur.map(un_fichier, FICHIERS) if nom]
