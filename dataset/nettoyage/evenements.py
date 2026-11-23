"""`shortage_events` : la table centrale du niveau 2. Une ligne = un épisode de pénurie,
dans un pays, au même format pour la France, l'Italie, l'Allemagne, la Norvège, la Suède, la Tchéquie et la Slovaquie.

Colonnes communes :
    pays, id_episode, medicament, substance, atc5, atc4, debut, fin, en_cours,
    type_episode (penurie / arret_definitif), gravite (FR), criticite (DE), raison (IT, DE),
    debut_connu (faux si le début n'est connu qu'approximativement),
    publication (première photo où l'épisode est publié : la date à laquelle on le sait, à utiliser
    pour les variables de prédiction),
    censure_gauche (déjà publié dans la première photo du pays : début réel inconnu, à exclure des
    durées et des comptes de nouvelles pénuries),
    censure_droite (France : épisode encore en cours au passage de l'ancien au nouveau site de l'ANSM,
    en 04/2021 ; sa `fin` est la dernière photo de l'ancien site, sa vraie fin est inconnue : à exclure des
    durées, comme `en_cours`),
    fin_connue_le (photo où l'on voit que l'épisode est terminé : `fin` ne peut servir à une date d
    que si fin_connue_le < d, voir `ansm_episodes.py`)
La France vient de `ansm_episodes`, les autres pays de `europe_episodes`.
Norvège, Suède, Tchéquie, Slovaquie : registres et non photos ; un épisode n'est censuré à gauche que s'il a été publié
avant le début collecté (2019 pour la Norvège, 2018 pour la Suède et la Tchéquie, 04/2019 pour la Slovaquie).

Types : les colonnes à peu de valeurs distinctes sont en dtype `category` (une valeur stockée une
fois, un petit code par ligne) ; la gravité est une catégorie ordonnée.
"""
import numpy as np
import pandas as pd

from .ansm_episodes import GRAVITE
from .causes import famille_cause, type_cause
from .commun import CLEAN, enregistrer

COLONNES = ["pays", "id_episode", "medicament", "substance", "atc5", "atc4", "debut", "fin",
            "en_cours", "type_episode", "gravite", "criticite", "raison", "debut_connu", "publication",
            "censure_gauche", "censure_droite", "fin_connue_le",
            "cause", "cause_connue", "anomalie_date", "duree_j", "age_j", "ouvert_longue_duree", "doublon"]
OUVERT_LONGTEMPS_JOURS = 730  # un épisode encore « en cours » après 2 ans est suspect (reprise non déclarée ?)
DATES = ["debut", "fin", "publication", "fin_connue_le"]
# Registres (et non photos) : un épisode n'est censuré à gauche que s'il a été publié avant le début collecté.
DEBUT_REGISTRES = {"NO": pd.Timestamp("2019-01-01"), "SE": pd.Timestamp("2018-01-01"), "CZ": pd.Timestamp("2018-01-01"),
                   "SK": pd.Timestamp("2019-04-15")}
CATEGORIES = {"pays": None, "type_episode": None, "criticite": None,
              "gravite": pd.CategoricalDtype(list(GRAVITE), ordered=True)}


def dernieres_observations(evenements):
    """Date de la dernière observation de chaque pays : dernière photo (France, Italie, Allemagne, Norvège), ou
    dernière déclaration publiée pour les registres complets (Suède, Tchéquie, Slovaquie)."""
    photos = pd.concat([pd.read_parquet(CLEAN / "ansm_observations.parquet", columns=["date_photo"]).assign(pays="FR"),
                        pd.read_parquet(CLEAN / "europe_observations.parquet", columns=["pays", "date_photo"])])
    derniere = photos.groupby("pays")["date_photo"].max().astype("datetime64[us]")
    registres = evenements[~evenements["pays"].astype(str).isin(derniere.index)]
    derniere = pd.concat([derniere, registres.groupby(registres["pays"].astype(str))["publication"].max()])
    return derniere


def qualite_des_episodes(evenements):
    """Colonnes de qualité : on ne supprime aucun épisode, on dit ce qui est fiable.
    - cause, cause_connue : famille de cause (voir causes.py) ;
    - anomalie_date : `fin_avant_debut` (fin antérieure au début), `debut_futur` (début postérieur à la dernière
      observation du pays : date prévue, pas observée), sinon `aucune` ;
    - duree_j : durée en jours, seulement pour un épisode terminé, au début connu, sans anomalie, non censuré ;
    - age_j : ancienneté d'un épisode encore en cours à la dernière observation ;
    - ouvert_longue_duree : en cours depuis plus de 2 ans (épisode peut-être jamais clos) ;
    - doublon : même pays, même substance, mêmes dates et même type qu'une ligne précédente (plusieurs
      présentations ou déclarations d'une même substance) ; à écarter pour compter des substances."""
    derniere = evenements["pays"].astype(str).map(dernieres_observations(evenements))
    evenements["cause"] = evenements["raison"].map(famille_cause).astype(type_cause())
    evenements["cause_connue"] = evenements["cause"] != "Non renseignée"
    evenements["anomalie_date"] = pd.Series(
        np.select([evenements["fin"] < evenements["debut"], evenements["debut"] > derniere],
                  ["fin_avant_debut", "debut_futur"], "aucune"), index=evenements.index).astype("category")
    propre = evenements["anomalie_date"] == "aucune"
    termine = ~evenements["en_cours"] & evenements["fin"].notna()
    mesurable = propre & termine & evenements["debut_connu"] & ~evenements["censure_gauche"] & ~evenements["censure_droite"]
    evenements["duree_j"] = ((evenements["fin"] - evenements["debut"]).dt.days.where(mesurable)).astype("Int64")
    en_cours = propre & evenements["en_cours"]
    evenements["age_j"] = ((derniere - evenements["debut"]).dt.days.where(en_cours)).astype("Int64")
    evenements["ouvert_longue_duree"] = (evenements["age_j"] > OUVERT_LONGTEMPS_JOURS).fillna(False).astype(bool)
    evenements["doublon"] = evenements["atc5"].notna() & evenements.duplicated(
        ["pays", "atc5", "debut", "fin", "type_episode"], keep="first")
    return evenements


def construire():
    france = pd.read_parquet(CLEAN / "ansm_episodes.parquet")
    europe = pd.read_parquet(CLEAN / "europe_episodes.parquet")

    france = france.assign(
        pays="FR",
        id_episode=france["fiche"] + "_" + france["numero_episode"].astype(str),
        # Épisode vu seulement à sa remise à disposition : début borné par la photo précédente.
        debut_connu=france["debut"].notna(),
        debut=france["debut"].fillna(france["debut_au_plus_tot"]),
        type_episode="penurie", gravite=france["gravite_max"], criticite=None, raison=None,
        publication=france["premiere_photo"],
        censure_droite=france["type_fin"] == "changement_de_site")
    premiere_photo_du_pays = europe.groupby("pays")["premiere_photo"].transform("min")
    europe = europe.assign(gravite=None, debut_connu=True, censure_droite=False,
                           censure_gauche=europe["premiere_photo"] == premiere_photo_du_pays,
                           fin_connue_le=europe["fin_au_plus_tard"])
    for pays, debut in DEBUT_REGISTRES.items():
        registre = europe["pays"] == pays
        europe.loc[registre, "censure_gauche"] = europe.loc[registre, "publication"] < debut

    evenements = pd.concat([france, europe], ignore_index=True)
    evenements["atc4"] = evenements["atc5"].str[:5]
    # Les sources n'ont pas la même précision de date : une seule pour toute la table.
    evenements[DATES] = evenements[DATES].astype("datetime64[us]")
    for colonne, dtype in CATEGORIES.items():
        evenements[colonne] = evenements[colonne].astype(dtype or "category")
    evenements = evenements.sort_values(["pays", "debut"]).reset_index(drop=True)
    evenements = qualite_des_episodes(evenements)[COLONNES]
    enregistrer(evenements, "shortage_events")
    print("  " + evenements.groupby(["pays", "type_episode"]).size().to_string().replace("\n", "\n  "))
    print("  anomalies de dates :", evenements["anomalie_date"].value_counts().to_dict(),
          "; doublons :", int(evenements["doublon"].sum()), "; ouverts depuis plus de 2 ans :",
          int(evenements["ouvert_longue_duree"].sum()))
