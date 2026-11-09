"""Norvège — registres annuels des pénuries de l'Agence norvégienne des médicaments (DMP).

Jusqu'en 07/2024, la DMP publiait chaque année un classeur « Legemiddelmangel og avregistreringer »
(pénuries et retraits) : toutes les pénuries déclarées dans l'année et celles encore en cours, avec la
date de déclaration (Meldingsdato), la période de pénurie, la cause, le statut et, depuis 2020, le code
ATC. Depuis, la liste n'est plus qu'une recherche en ligne, sans fichier : les classeurs ne se trouvent
plus que dans la Wayback Machine. On les y prend (suffixe `id_` : fichier original, sans la barre de
la Wayback Machine). Le classeur 2024 s'arrête au 12/07/2024.
On part de 2019 : le classeur 2019 reprend les pénuries de 2018 encore en cours, et les classeurs plus
anciens ont un autre format (période « fra-til » en texte libre) alors que le panel commence en 2020.
"""
import time

from .commun import telecharger

DMP = "https://www.dmp.no/globalassets/documents/legemiddelmangel/"
ANCIENS = DMP + "mangeloversikter-tidligere-ar/"
# (année, adresse archivée). Les adresses suivent les rangements successifs du site de la DMP.
CLASSEURS = [
    (2019, ANCIENS + "legemiddelmangel-og-avregistreringer-2019.xlsx"),
    (2020, ANCIENS + "legemiddelmangel-og-avregistreringer-2020.xlsx"),
    (2021, DMP + "covid-19/legemiddelmangel-og-avregistreringer-2021.xlsx"),
    (2022, DMP + "pakningsvedlegg/legemiddelmangel-og-avregistreringer-2022.xlsx"),
    (2023, DMP + "legemiddelmangel-og-avregistreringer-2023.xlsx"),
    (2024, DMP + "legemiddelmangel-og-avregistreringer-2024.xlsx"),
]
PAUSE = 1.5  # secondes entre deux requêtes à la Wayback Machine


def est_un_classeur(contenu):
    """Refuse une page HTML reçue avec un code 200 : on attend un .xlsx (ZIP) ou un .xls (OLE)."""
    return contenu[:4] in (b"PK\x03\x04", b"\xd0\xcf\x11\xe0")


def collecter(session):
    echecs = []
    for annee, adresse in CLASSEURS:
        extension = adresse.rsplit(".", 1)[-1]
        destination = f"dmp/legemiddelmangel_{annee}.{extension}"
        # « 2026id_ » : la Wayback Machine sert la capture la plus récente de ce fichier.
        archive = f"https://web.archive.org/web/2026id_/{adresse}"
        if telecharger(session, archive, destination, "dmp", valider=est_un_classeur):
            print(f"  dmp {annee} ok")
        else:
            echecs.append(destination)
        time.sleep(PAUSE)
    return echecs
