"""France — Medic'AM (Assurance Maladie) : boîtes remboursées en ville, MOIS par mois, depuis 2015.

Open Medic est annuel ; Medic'AM, lui, détaille chaque mois (boîtes, base et montant remboursés)
pour chaque présentation (CIP13), avec son code ATC. Il est publié par semestre sur data.gouv.fr
(fichiers 1er semestre, 2e semestre et « annuel » pour chaque année). La liste des fichiers est lue
dans l'API de data.gouv.fr : un nouveau semestre publié plus tard est donc collecté à la relance.

Les fichiers « 2012-2014 » (annuels) sont ignorés : le projet commence en 2015.
"""
import re

from .commun import telecharger

API = "https://www.data.gouv.fr/api/1/datasets/medicaments-rembourses-par-lassurance-maladie/"
LIEN = "https://www.data.gouv.fr/api/1/datasets/r/{}"


def periode(titre):
    """« Medic'AM - 1er semestre 2026 » → « 2026_s1 » ; « Medic'AM annuel 2024 » → « 2024_annuel »."""
    annee = re.search(r"(20\d\d)", titre)
    if annee is None or "2012-2014" in titre:
        return None
    if "annuel" in titre.lower():
        return f"{annee.group(1)}_annuel"
    semestre = re.search(r"(1er|2e)\s+semestre", titre, flags=re.IGNORECASE)
    return f"{annee.group(1)}_s{'1' if semestre.group(1).lower() == '1er' else '2'}" if semestre else None


def est_un_classeur(contenu):
    """Refuse une page d'erreur HTML reçue avec un code HTTP 200 : on attend une archive ZIP ou un
    classeur Excel (ancien format .xls ou .xlsx)."""
    return contenu[:4] in (b"PK\x03\x04", b"\xd0\xcf\x11\xe0")


def corriger_adresses(ressources):
    """Défaut de la source : la ressource « 2e semestre 2025 » de data.gouv pointe vers le fichier du
    1er semestre (janvier à juin), les deux ressources ayant la même adresse. Le vrai fichier existe
    chez l'Assurance Maladie, avec le nom régulier « 2025-07-a-12_... » : on le déduit du nom du
    1er semestre. Renvoie (titre, id, adresse) de chaque ressource."""
    par_adresse = {}
    for r in ressources:
        par_adresse.setdefault(r["url"], []).append(r)
    resultat = []
    for r in ressources:
        adresse = r["url"]
        if len(par_adresse[adresse]) > 1 and periode(r["title"]).endswith("_s2") and "01-a-06" in adresse:
            adresse = adresse.replace("01-a-06", "07-a-12")
            print(f"  medicam : adresse de « {r['title']} » corrigée (doublon du 1er semestre)")
        resultat.append((r["title"], r["id"], adresse))
    return resultat


def collecter(session):
    ressources = [r for r in session.get(API, timeout=60).json()["resources"] if periode(r["title"])]
    echecs = []
    for titre, identifiant, adresse in sorted(corriger_adresses(ressources)):
        nom = periode(titre)
        extension = next(r["url"] for r in ressources if r["id"] == identifiant).rsplit(".", 1)[-1].lower()
        destination = f"medicam/MEDICAM_{nom}.{extension if extension in ('zip', 'xls', 'xlsx') else 'zip'}"
        # Adresse directe quand elle a été corrigée, sinon le lien stable de data.gouv.
        lien = adresse if adresse != next(r["url"] for r in ressources if r["id"] == identifiant) \
            else LIEN.format(identifiant)
        if telecharger(session, lien, destination, "medicam", timeout=900, valider=est_un_classeur):
            print(f"  medicam {nom} ok")
        else:
            echecs.append(destination)
    return echecs
