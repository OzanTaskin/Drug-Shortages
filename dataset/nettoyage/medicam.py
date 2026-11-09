"""France — Medic'AM (Assurance Maladie) → `consommation_mensuelle`, `medicam_non_classe`,
`medicam_reconciliation`.

Chaque fichier semestriel donne, pour chaque présentation (CIP13) et chaque mois, la base de
remboursement, le nombre de boîtes et le montant remboursés (« tous prescripteurs », ville : les
médicaments délivrés à l'hôpital ne sont pas inclus). On agrège au grain du panel, la substance
(ATC5) et le mois, comme `consommation` (Open Medic) l'est par année.

Tables produites :
- consommation_mensuelle  une ligne par ATC5 × mois : boîtes, montant, base, nb de présentations vendues
- medicam_non_classe      par mois : boîtes dont le code ATC n'a pas 7 caractères (non rattachables)
- medicam_reconciliation  par année : boîtes du fichier annuel et somme des 12 mois (contrôle)

Les fichiers « annuel » ne servent qu'au contrôle : les ajouter aux semestres compterait deux fois.
Medic'AM est exprimé en date de remboursement, pas de délivrance.
"""
import io
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from .commun import RAW, enregistrer

ATC5 = r"[A-Z]\d{2}[A-Z]{2}\d{2}"


def lire_fichier(chemin):
    """Une feuille « tous prescripteurs » → (une ligne par CIP13 × mois, nom de la période)."""
    if zipfile.is_zipfile(chemin):
        with zipfile.ZipFile(chemin) as archive:
            classeur = next(i for i in archive.infolist() if i.filename.lower().endswith((".xls", ".xlsx")))
            contenu = archive.read(classeur)
    else:  # classeur Excel livré sans archive (2019 annuel)
        contenu = chemin.read_bytes()
    excel = pd.ExcelFile(io.BytesIO(contenu))
    feuille = next(s for s in excel.sheet_names if "tous" in s.lower())
    table = excel.parse(feuille, dtype=object)
    table.columns = [re.sub(r"\s+", " ", str(c)).strip().lower() for c in table.columns]
    # Les en-têtes mensuels : « nombre de boites remboursées 2026-01 » ; annuels : « ... 2024 ».
    # Le fichier 2017 S2 nomme ses colonnes « Code » au lieu de « Code ATC » : l'ATC est toujours la 6e.
    colonne_atc = "code atc" if "code atc" in table.columns else table.columns[5]
    mesures = {"nombre de boites remboursées": "boites", "montant remboursé": "montant",
               "base de remboursement": "base"}
    morceaux = []
    for colonne in table.columns:
        for debut, nom in mesures.items():
            if colonne.startswith(debut):
                periode = colonne[len(debut):].strip()
                morceaux.append(pd.DataFrame({"cip13": table["cip13"], "atc5": table[colonne_atc],
                                              "periode": periode, "mesure": nom,
                                              "valeur": pd.to_numeric(table[colonne], errors="coerce")}))
    longue = pd.concat(morceaux, ignore_index=True)
    longue["cip13"] = longue["cip13"].astype(str).str.strip()
    longue["atc5"] = longue["atc5"].astype(str).str.strip().str.upper()
    larges = longue.pivot_table(index=["cip13", "atc5", "periode"], columns="mesure", values="valeur",
                                aggfunc="sum").reset_index()
    return chemin.stem.replace("MEDICAM_", ""), larges


def construire():
    fichiers = sorted((RAW / "medicam").glob("MEDICAM_*.*"))
    mensuels = [f for f in fichiers if "annuel" not in f.name]
    annuels = [f for f in fichiers if "annuel" in f.name]
    with ProcessPoolExecutor() as executeur:
        lus_mensuels = list(executeur.map(lire_fichier, mensuels))
        lus_annuels = list(executeur.map(lire_fichier, annuels))

    mois = pd.concat([t for _, t in lus_mensuels], ignore_index=True)
    mois["mois"] = pd.to_datetime(mois.pop("periode"), format="%Y-%m")
    mois["classe"] = mois["atc5"].str.fullmatch(ATC5)

    # Boîtes qu'on ne peut rattacher à aucune substance : mesurées, pas cachées.
    totaux = mois.groupby("mois").agg(boites_total=("boites", "sum"))
    non_classees = mois[~mois["classe"]].groupby("mois")["boites"].sum().rename("boites_non_classees")
    non_classe = totaux.join(non_classees).fillna(0).reset_index()
    enregistrer(non_classe, "medicam_non_classe")

    classees = mois[mois["classe"]]
    table = (classees.groupby(["atc5", "mois"], as_index=False)
             .agg(boites=("boites", "sum"), montant_rembourse=("montant", "sum"),
                  base_remboursement=("base", "sum"),
                  nb_presentations_vendues=("boites", lambda s: int((s > 0).sum()))))
    table["boites"] = table["boites"].round().astype("int64")
    table["montant_rembourse"] = table["montant_rembourse"].round(2)
    table["base_remboursement"] = table["base_remboursement"].round(2)
    enregistrer(table.sort_values(["atc5", "mois"]), "consommation_mensuelle")

    # Réconciliation avec les fichiers annuels (même champ, autre découpage dans le temps).
    annee_mensuel = mois.assign(annee=mois["mois"].dt.year).groupby("annee")["boites"].sum()
    lignes = []
    for nom, table_annuelle in lus_annuels:
        annee = int(nom[:4])
        annuel = table_annuelle["boites"].sum()
        lignes.append({"annee": annee, "boites_annuel": round(float(annuel)),
                       "boites_mensuel": round(float(annee_mensuel.get(annee, 0))),
                       "mois_presents": int(mois.loc[mois["mois"].dt.year == annee, "mois"].nunique())})
    enregistrer(pd.DataFrame(lignes).sort_values("annee"), "medicam_reconciliation")
