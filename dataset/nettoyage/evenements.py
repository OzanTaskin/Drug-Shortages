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
import pandas as pd

from .ansm_episodes import GRAVITE
from .commun import CLEAN, enregistrer

COLONNES = ["pays", "id_episode", "medicament", "substance", "atc5", "atc4", "debut", "fin",
            "en_cours", "type_episode", "gravite", "criticite", "raison", "debut_connu", "publication",
            "censure_gauche", "censure_droite", "fin_connue_le"]
DATES = ["debut", "fin", "publication", "fin_connue_le"]
# Registres (et non photos) : un épisode n'est censuré à gauche que s'il a été publié avant le début collecté.
DEBUT_REGISTRES = {"NO": pd.Timestamp("2019-01-01"), "SE": pd.Timestamp("2018-01-01"), "CZ": pd.Timestamp("2018-01-01"),
                   "SK": pd.Timestamp("2019-04-15")}
CATEGORIES = {"pays": None, "type_episode": None, "criticite": None,
              "gravite": pd.CategoricalDtype(list(GRAVITE), ordered=True)}


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
    evenements = evenements[COLONNES].sort_values(["pays", "debut"]).reset_index(drop=True)
    enregistrer(evenements, "shortage_events")
    print("  " + evenements.groupby(["pays", "type_episode"]).size().to_string().replace("\n", "\n  "))
