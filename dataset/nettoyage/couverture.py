"""`couverture_mensuelle` : pour chaque pays et chaque mois, ce que la source permet de dire, et le stock de pénuries.

Une ligne = un pays × un mois entre la première observation du pays et son dernier mois complet. Elle sert à
ne pas confondre « aucune pénurie » et « on ne regarde pas » (un dashboard ou un modèle qui trace une courbe
doit la masquer ou la griser là où `fiable` vaut faux).

Colonnes :
- type_source       `photos` (page web archivée : France, Italie, Allemagne), `fichiers_annuels` (Norvège : un classeur
                    par an, sans photo mensuelle) ou `registre` (Suède, Tchéquie, Slovaquie : registre complet lu le
                    jour de la collecte) ;
- nb_photos         photos de la page collectées dans le mois (0 pour un registre) ;
- jours_depuis_photo jours entre la fin du mois et la dernière photo connue (photos seulement) ;
- trou_archive      vrai si la dernière photo remonte à plus de 60 jours : l'état du mois n'est pas observé,
                    il est déduit d'une photo ancienne ;
- demarrage_registre vrai pendant les 12 premiers mois d'un registre : il ne contient encore que les déclarations
                    récentes, le stock y est sous-estimé ;
- fiable            ni trou d'archive ni démarrage de registre ;
- stock_fiable      `fiable` et fins connues : faux pour la Norvège, dont la fin d'un épisode n'est connue qu'au classeur
                    annuel suivant (seules les nouvelles déclarations sont fiables) ;
- stock_episodes    épisodes de pénurie en cours à la fin du mois (sans anomalie de date) ;
- stock_substances  substances (ATC5) distinctes concernées, sans compter les doublons ;
- stock_recents     part du stock commencée depuis moins de 2 ans (le reste peut être un épisode jamais clos) ;
- nouvelles_substances substances dont une pénurie commence dans le mois (hors épisodes déjà publiés à la
                    première observation, dont le vrai début est inconnu).
"""
import numpy as np
import pandas as pd

from .commun import CLEAN, enregistrer
from .evenements import DEBUT_REGISTRES, OUVERT_LONGTEMPS_JOURS, dernieres_observations

SEUIL_TROU_JOURS = 60
MOIS_DEMARRAGE = 12
PHOTOS = ("FR", "IT", "DE")  # pages archivées ; la Norvège (classeurs annuels) a ses propres règles


def photos_par_pays():
    photos = pd.concat([pd.read_parquet(CLEAN / "ansm_observations.parquet", columns=["date_photo"]).assign(pays="FR"),
                        pd.read_parquet(CLEAN / "europe_observations.parquet", columns=["pays", "date_photo"])])
    photos["date_photo"] = photos["date_photo"].astype("datetime64[us]")
    return {p: np.sort(g["date_photo"].unique()) for p, g in photos.groupby("pays")}


def construire():
    evenements = pd.read_parquet(CLEAN / "shortage_events.parquet")
    evenements["pays"] = evenements["pays"].astype(str)
    penuries = evenements[(evenements["type_episode"] == "penurie") & (evenements["anomalie_date"] == "aucune")
                          & evenements["debut"].notna()]
    derniere = dernieres_observations(evenements)
    photos = photos_par_pays()
    lignes = []
    for pays in ["FR", "IT", "DE", "NO", "SE", "CZ", "SK"]:
        type_source = "photos" if pays in PHOTOS else "fichiers_annuels" if pays == "NO" else "registre"
        premier = (pd.Timestamp(photos[pays][0]) if pays in PHOTOS else DEBUT_REGISTRES[pays]).to_period("M")
        dernier = ((derniere[pays] + pd.Timedelta(days=1)).to_period("M") - 1)  # dernier mois complet
        ep = penuries[penuries["pays"] == pays]
        debut, fin = ep["debut"].to_numpy(), ep["fin"].to_numpy()
        atc, publiee_au_depart = ep["atc5"].to_numpy(dtype=object), ep["censure_gauche"].to_numpy()
        for mois in pd.period_range(premier, dernier, freq="M"):
            debut_mois, fin_mois = mois.start_time, mois.end_time.normalize()
            ouvert = (debut <= fin_mois.to_datetime64()) & (pd.isna(fin) | (fin > fin_mois.to_datetime64()))
            age = (fin_mois.to_datetime64() - debut) / np.timedelta64(1, "D")
            nouvelles = ((debut >= debut_mois.to_datetime64()) & (debut <= fin_mois.to_datetime64()) & ~publiee_au_depart)
            ligne = {"pays": pays, "mois": debut_mois, "type_source": type_source, "nb_photos": 0, "jours_depuis_photo": pd.NA,
                     "stock_episodes": int(ouvert.sum()),
                     "stock_substances": len({a for a in atc[ouvert] if isinstance(a, str)}),
                     "stock_recents": int((ouvert & (age <= OUVERT_LONGTEMPS_JOURS)).sum()),
                     "nouvelles_substances": len({a for a in atc[nouvelles] if isinstance(a, str)})}
            if type_source == "photos":
                dates = photos[pays]
                dans_le_mois = (dates >= debut_mois.to_datetime64()) & (dates <= fin_mois.to_datetime64())
                avant = dates[dates <= fin_mois.to_datetime64()]
                ligne["nb_photos"] = int(dans_le_mois.sum())
                if len(avant):
                    ligne["jours_depuis_photo"] = int((fin_mois - pd.Timestamp(avant[-1])).days)
            lignes.append(ligne)
    table = pd.DataFrame(lignes)
    table["jours_depuis_photo"] = table["jours_depuis_photo"].astype("Int64")
    table["trou_archive"] = (table["type_source"] == "photos") & (table["jours_depuis_photo"] > SEUIL_TROU_JOURS).fillna(False)
    depart = table["pays"].map({p: DEBUT_REGISTRES[p].to_period("M").to_timestamp() for p in DEBUT_REGISTRES if p != "NO"})
    table["demarrage_registre"] = (table["type_source"] == "registre") & (
        table["mois"] < depart + pd.DateOffset(months=MOIS_DEMARRAGE))
    table["fiable"] = ~table["trou_archive"] & ~table["demarrage_registre"]
    table["stock_fiable"] = table["fiable"] & (table["pays"] != "NO")
    table["type_source"] = table["type_source"].astype("category")
    table["mois"] = table["mois"].astype("datetime64[us]")
    enregistrer(table, "couverture_mensuelle")
    print("  mois fiables par pays :", table.groupby("pays")["fiable"].agg(["sum", "size"]).apply(
        lambda r: f"{int(r['sum'])}/{int(r['size'])}", axis=1).to_dict())
