"""Construit les tables résumées du dashboard (dashboard/donnees/*.parquet).

Lit les tables nettoyées du dataset (dataset/data/clean et dataset/data/final, non versionnées) et écrit
quelques petites tables pré-agrégées, versionnées avec le dashboard : l'enseignant lance le dashboard sans
avoir à reconstruire les 500 Mo de données.

    .venv/bin/python dashboard/construire_donnees.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage

ICI = Path(__file__).resolve().parent
DATA = ICI.parent / "dataset" / "data"
SORTIE = ICI / "donnees"

PAYS = ["FR", "IT", "DE", "NO", "SE", "CZ", "SK"]
DERNIER_MOIS = pd.Period("2026-10", "M")
# Mois pendant lesquels chaque source observe réellement les pénuries (archives ou registre).
COUVERTURE = {"FR": "2015-02", "IT": "2019-08", "DE": "2024-07", "NO": "2019-01", "SE": "2018-01",
              "CZ": "2018-01", "SK": "2019-04"}
FIN_COUVERTURE = {"NO": "2024-06"}
MIN_TAILLE = 5  # substances minimum par sous-classe ATC2 dans la carte de chaleur

# Familles de causes : mots-clés (minuscules) dans les libellés en italien, allemand, norvégien, suédois,
# tchèque (traduit en anglais par l'agence). Le premier motif qui correspond l'emporte.
FAMILLES = [
    ("Demande en hausse", r"richiesta|økt salg|demand|increased"),
    ("Production / qualité", r"produtt|produk|produc|manufactur|quality|kvalit|råvare|frigivelse|gmp"),
    ("Distribution / capacité", r"distribu|capacit|kapasitet|levering|deliver|logist|forniture|transport"),
    ("Arrêt / raison commerciale", r"commerci|cessat|withdraw|revocat|lavt salg|marketing|switch|reimburs|"
                                   r"sospensione|avvenuta|endring|discontinu"),
    ("Réglementaire", r"regolator|regulat|procedur|regulator|myndighet"),
]


def famille_cause(raison: object) -> str:
    if not isinstance(raison, str) or not raison.strip():
        return "Non renseignée"
    texte = raison.lower()
    for nom, motif in FAMILLES:
        if re.search(motif, texte):
            return nom
    return "Autre"


def mois_actifs(ep: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (pays, atc5, mois) où une pénurie est en cours, limitée à la couverture du pays."""
    debut = ep["debut"].dt.to_period("M")
    fin = ep["fin"].fillna(DERNIER_MOIS.to_timestamp()).dt.to_period("M").clip(upper=DERNIER_MOIS)
    lignes = []
    for pays, d, f, atc in zip(ep["pays"], debut, fin, ep["atc5"]):
        premier = max(d, pd.Period(COUVERTURE[pays], "M"))
        dernier = min(f, pd.Period(FIN_COUVERTURE.get(pays, str(DERNIER_MOIS)), "M"))
        if premier <= dernier:
            lignes.append((pays, atc, premier.ordinal, dernier.ordinal))
    bornes = pd.DataFrame(lignes, columns=["pays", "atc5", "o1", "o2"])
    n = (bornes["o2"] - bornes["o1"] + 1).to_numpy()
    rep = bornes.loc[bornes.index.repeat(n)].reset_index(drop=True)
    rep["ordinal"] = rep["o1"] + _rang(n)
    rep["mois"] = pd.PeriodIndex.from_ordinals(rep["ordinal"], freq="M").to_timestamp()
    return rep[["pays", "atc5", "mois"]]


def _rang(n: np.ndarray) -> np.ndarray:
    """0, 1, …, n_i - 1 concaténés (position de chaque mois dans son épisode)."""
    debuts = np.repeat(np.cumsum(n) - n, n)
    return np.arange(n.sum()) - debuts


def ordre_cluster(matrice: pd.DataFrame) -> list[str]:
    """Ordre des lignes par clustering hiérarchique (Ward) sur les profils mensuels."""
    if len(matrice) < 3:
        return list(matrice.index)
    return list(matrice.index[leaves_list(linkage(matrice.fillna(0).to_numpy(), "ward"))])


def main() -> None:
    SORTIE.mkdir(exist_ok=True)
    ev = pd.read_parquet(DATA / "clean" / "shortage_events.parquet")
    sig = pd.read_parquet(DATA / "clean" / "ansm_signalements.parquet")
    atc = pd.read_parquet(DATA / "clean" / "atc.parquet")
    conso = pd.read_parquet(DATA / "clean" / "consommation.parquet")
    ue = pd.read_parquet(DATA / "clean" / "ue_liste_critique.parquet")
    panel = pd.read_parquet(DATA / "final" / "panel_fr.parquet")
    obs_fr = pd.read_parquet(DATA / "clean" / "ansm_observations.parquet")
    obs_eu = pd.read_parquet(DATA / "clean" / "europe_observations.parquet")

    # ── Épisodes ───────────────────────────────────────────────────────────────────────────────
    ep = ev[["pays", "id_episode", "medicament", "substance", "atc5", "debut", "fin", "en_cours", "type_episode",
             "raison", "publication", "censure_gauche", "censure_droite"]].copy()
    ep["pays"] = ep["pays"].astype(str)
    ep["type_episode"] = ep["type_episode"].astype(str)
    ep["cause"] = ep["raison"].map(famille_cause)
    ep["duree_j"] = (ep["fin"] - ep["debut"]).dt.days.where(~ep["en_cours"])
    ep.loc[ep["duree_j"] < 0, "duree_j"] = np.nan
    ep = ep.drop(columns="raison")
    ep.to_parquet(SORTIE / "episodes.parquet", index=False)

    # ── Substances (une ligne par code ATC5) ──────────────────────────────────────────────────
    mitm = set(panel["atc5"])
    codes = sorted(set(ep["atc5"].dropna()) | mitm | set(sig["atc5"].dropna()))
    codes = [c for c in codes if re.fullmatch(r"[A-Z]\d{2}[A-Z]{2}\d{2}", c)]
    sub = pd.DataFrame({"atc5": codes})
    for niveau, longueur in [("atc1", 1), ("atc2", 3), ("atc3", 4), ("atc4", 5)]:
        sub[niveau] = sub["atc5"].str[:longueur]
    for niveau in ["atc1", "atc2", "atc3", "atc4", "atc5"]:
        libelles = atc.dropna(subset=[niveau]).drop_duplicates(niveau).set_index(niveau)[f"libelle_{niveau}"]
        sub[f"libelle_{niveau}"] = sub[niveau].map(libelles).str.capitalize()
    noms_ev = ep.dropna(subset=["atc5", "substance"]).groupby("atc5")["substance"].agg(lambda s: s.mode().iat[0])
    sub["nom"] = sub["libelle_atc5"].fillna(sub["atc5"].map(noms_ev).str.capitalize()).fillna(sub["atc5"])
    sub["mitm"] = sub["atc5"].isin(mitm)
    sub["ue_critique"] = sub["atc5"].isin(set(ue["atc5"]))
    sub["nb_signalements"] = sub["atc5"].map(sig["atc5"].value_counts()).fillna(0).astype(int)
    pen = ep[ep["type_episode"] == "penurie"]
    comptes = pen.groupby(["atc5", "pays"]).size().unstack(fill_value=0).reindex(columns=PAYS, fill_value=0)
    for p in PAYS:
        sub[f"ep_{p}"] = sub["atc5"].map(comptes[p]).fillna(0).astype(int)
    sub["nb_pays_touches"] = (sub[[f"ep_{p}" for p in PAYS]] > 0).sum(axis=1)
    sub["duree_mediane_j"] = sub["atc5"].map(pen.groupby("atc5")["duree_j"].median())
    derniere = conso[conso["annee"] == 2024].set_index("atc5")
    sub["boites_2024"] = sub["atc5"].map(derniere["boites"])
    sub["nb_presentations"] = sub["atc5"].map(derniere["nb_presentations"])
    sub.to_parquet(SORTIE / "substances.parquet", index=False)

    # ── Séries par pays et par mois ───────────────────────────────────────────────────────────
    actifs = mois_actifs(pen.dropna(subset=["debut"]))
    serie = actifs.groupby(["pays", "mois"]).size().rename("en_cours").reset_index()
    nouv = pen.assign(mois=pen["debut"].dt.to_period("M").dt.to_timestamp())
    nouv = nouv[nouv["mois"] >= nouv["pays"].map(lambda p: pd.Period(COUVERTURE[p], "M").to_timestamp())]
    nouv = nouv.groupby(["pays", "mois"]).size().rename("nouvelles").reset_index()
    serie = serie.merge(nouv, on=["pays", "mois"], how="left").fillna({"nouvelles": 0})
    serie["nouvelles"] = serie["nouvelles"].astype(int)
    serie.to_parquet(SORTIE / "serie_pays.parquet", index=False)

    # ── Signalements ANSM (bruts, petits) ─────────────────────────────────────────────────────
    sig[["date_signalement", "annee", "laboratoire", "atc5", "medicament"]].to_parquet(
        SORTIE / "signalements.parquet", index=False)
    conso[["atc5", "annee", "boites", "montant_rembourse", "nb_presentations"]].to_parquet(
        SORTIE / "consommation.parquet", index=False)

    # ── Classes ATC2 × mois : part des substances MITM touchées ──────────────────────────────
    univers = sub[sub["mitm"]]
    taille = univers.groupby("atc2").size()
    tables, ordres = [], {}
    sig_m = sig[sig["atc5"].isin(mitm)].assign(mois=sig["date_signalement"].dt.to_period("M").dt.to_timestamp())
    sources = {
        "signalements": sig_m[["atc5", "mois"]].drop_duplicates(),
        "episodes_fr": actifs[(actifs["pays"] == "FR") & actifs["atc5"].isin(mitm)
                              & (actifs["mois"] >= "2019-01-01")][["atc5", "mois"]].drop_duplicates(),
    }
    for source, touches in sources.items():
        touches = touches.assign(atc2=touches["atc5"].str[:3])
        t = touches.groupby(["atc2", "mois"]).size().rename("n").reset_index()
        t["taille"] = t["atc2"].map(taille)
        t["part"] = t["n"] / t["taille"]
        t["source"] = source
        tables.append(t)
        # Une sous-classe d'une ou deux substances passe de 0 à 100 % en un signalement : on ne garde que
        # celles d'au moins MIN_TAILLE substances (45 sous-classes, 822 des 885 substances).
        large = t.pivot(index="atc2", columns="mois", values="part").reindex(
            taille[taille >= MIN_TAILLE].index).fillna(0)
        ordres[source] = ordre_cluster(large)
    classes = pd.concat(tables, ignore_index=True)
    classes.to_parquet(SORTIE / "classes_mois.parquet", index=False)

    # ── Co-occurrence entre pays : P(pénurie en B | pénurie en A, même substance, même mois) ─
    act = actifs.dropna(subset=["atc5"]).drop_duplicates()
    cles = {p: set(zip(g["atc5"], g["mois"])) for p, g in act.groupby("pays")}
    mois_p = {p: set(g["mois"]) for p, g in act.groupby("pays")}
    cooc = []
    for a in PAYS:
        for b in PAYS:
            communs = mois_p[a] & mois_p[b]
            ka = {k for k in cles[a] if k[1] in communs}
            kb = {k for k in cles[b] if k[1] in communs}
            cooc.append({"a": a, "b": b, "p": len(ka & kb) / len(ka) if ka else np.nan,
                         "n_a": len(ka), "n_ab": len(ka & kb), "mois_communs": len(communs)})
    pd.DataFrame(cooc).to_parquet(SORTIE / "cooccurrence.parquet", index=False)

    # ── Couverture : dates des photos / fichiers par pays ─────────────────────────────────────
    photos = pd.concat([
        obs_fr[["date_photo", "site"]].drop_duplicates().assign(pays="FR").rename(columns={"site": "detail"}),
        obs_eu[["pays", "date_photo"]].drop_duplicates().assign(detail="archive"),
    ], ignore_index=True)
    photos["date_photo"] = photos["date_photo"].astype("datetime64[us]")
    photos.to_parquet(SORTIE / "photos.parquet", index=False)

    # ── Panel : résumé mensuel et part des signaux ────────────────────────────────────────────
    signaux = ["fr_affichee_3m", "it_affichee_3m", "no_nouvelle_3m", "se_nouvelle_3m", "cz_nouvelle_3m",
               "sk_nouvelle_3m", "ue_critique"]
    pm = panel.assign(fr_autres=(panel["fr_autres_classe_3m"] > 0).astype(int),
                      deja_signale=(panel["sig_cumul_pub"] > 0).astype(int))
    agg = {"lignes": ("atc5", "size"), "cible": ("cible", "mean"), "calmes_12m": ("calme_12m", "sum")}
    resume = pm.groupby("mois").agg(**agg).reset_index()
    calmes = pm[pm["calme_12m"] == 1].groupby("mois")["cible_nouvelle_12m"].agg(["mean", "sum"])
    resume = resume.merge(calmes.rename(columns={"mean": "cible_nouvelle_12m", "sum": "nouvelles_12m"}),
                          left_on="mois", right_index=True, how="left")
    resume["jeu"] = resume["mois"].map(pm.drop_duplicates("mois").set_index("mois")["jeu"].astype(str))
    resume.to_parquet(SORTIE / "panel_mensuel.parquet", index=False)
    parts = pm.groupby("mois")[signaux + ["fr_autres", "deja_signale"]].mean()
    couv = pm.groupby("mois")[["it_couvert", "no_couvert"]].mean()
    parts.loc[couv["it_couvert"] == 0, "it_affichee_3m"] = np.nan
    parts.loc[couv["no_couvert"] == 0, "no_nouvelle_3m"] = np.nan
    parts.reset_index().to_parquet(SORTIE / "panel_signaux.parquet", index=False)

    meta = {"ordres_atc2": ordres, "min_taille_atc2": MIN_TAILLE,"couverture": COUVERTURE, "fin_couverture": FIN_COUVERTURE,
            "dernier_mois": str(DERNIER_MOIS), "panel_lignes": int(len(panel)),
            "panel_substances": int(panel["atc5"].nunique()), "episodes": int(len(ev))}
    (SORTIE / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))

    for f in sorted(SORTIE.glob("*")):
        print(f"{f.name:28s} {f.stat().st_size / 1024:8.0f} Ko")


if __name__ == "__main__":
    main()
