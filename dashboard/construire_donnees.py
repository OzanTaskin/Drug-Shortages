"""Construit les petites tables du dashboard (dashboard/donnees/*.parquet, versionnées).

Lit data/clean/ et data/final/ (non versionnés, régénérés par `python -m nettoyage`), et outputs/
(`python -m analyse.modeles` puis `python -m analyse.prediction`). L'enseignant lance ensuite le dashboard
sans reconstruire les 500 Mo de données :

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
RACINE = ICI.parent
CLEAN, FINAL, SORTIES = RACINE / "data" / "clean", RACINE / "data" / "final", RACINE / "outputs"
SORTIE = ICI / "donnees"

PAYS = ["FR", "IT", "DE"]
MIN_TAILLE = 5  # substances minimum par sous-classe ATC2 dans la carte de chaleur

# Familles de causes (Italie, Allemagne ; la France n'en publie pas). Le premier motif trouvé l'emporte.
FAMILLES = [
    ("Arrêt de commercialisation", r"cessata commercializzazione"),
    ("Demande en hausse", r"richiesta"),
    ("Production / qualité", r"produtt|produktion|qualit"),
    ("Distribution", r"distribu|forniture"),
    ("Réglementaire", r"regolator"),
]


def famille_cause(raison: object) -> str:
    if not isinstance(raison, str) or not raison.strip():
        return "Non renseignée"
    texte = raison.lower()
    return next((nom for nom, motif in FAMILLES if re.search(motif, texte)), "Autre")


def dernier(nom: str) -> Path:
    fichiers = sorted(SORTIES.glob(nom))
    if not fichiers:
        raise SystemExit(f"{nom} introuvable dans outputs/ : lancer python -m analyse.modeles puis analyse.prediction")
    return fichiers[-1]


def mois_actifs(ep: pd.DataFrame, couverture: dict, fin: pd.Period) -> pd.DataFrame:
    """Une ligne par (pays, atc5, mois) où une pénurie est en cours, limitée à la période observée du pays."""
    d = ep["debut"].dt.to_period("M")
    f = ep["fin"].dt.to_period("M").fillna(fin)
    lignes = []
    for pays, a, b, atc in zip(ep["pays"], d, f, ep["atc5"]):
        a, b = max(a, couverture[pays]), min(b, fin)
        if a <= b:
            lignes.append((pays, atc, a.ordinal, b.ordinal))
    bornes = pd.DataFrame(lignes, columns=["pays", "atc5", "o1", "o2"])
    n = (bornes["o2"] - bornes["o1"] + 1).to_numpy()
    rep = bornes.loc[bornes.index.repeat(n)].reset_index(drop=True)
    rep["mois"] = pd.PeriodIndex.from_ordinals(rep["o1"] + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n),
                                               freq="M").to_timestamp()
    return rep[["pays", "atc5", "mois"]]


def ordre_cluster(matrice: pd.DataFrame) -> list[str]:
    """Ordre des lignes par clustering hiérarchique (Ward) sur les profils mensuels."""
    if len(matrice) < 3:
        return list(matrice.index)
    return list(matrice.index[leaves_list(linkage(matrice.fillna(0).to_numpy(), "ward"))])


def main() -> None:
    SORTIE.mkdir(exist_ok=True)
    lire = lambda nom: pd.read_parquet(CLEAN / f"{nom}.parquet")
    ev, sig, atc, conso, ue = (lire(n) for n in ("shortage_events", "ansm_signalements", "atc", "consommation",
                                                 "ue_liste_critique"))
    panel = pd.read_parquet(FINAL / "panel_fr.parquet")
    photos_fr, photos_eu = lire("ansm_observations"), lire("europe_observations")
    metriques = json.loads(dernier("metriques_*.json").read_text())
    pred_eval = pd.read_parquet(dernier("predictions_*.parquet"))
    prediction = pd.read_parquet(dernier("prediction_*.parquet"))

    # Période observée de chaque pays : de la première à la dernière photo archivée
    photos = pd.concat([photos_fr[["date_photo"]].drop_duplicates().assign(pays="FR"),
                        photos_eu[["pays", "date_photo"]].drop_duplicates()], ignore_index=True)
    photos["date_photo"] = photos["date_photo"].astype("datetime64[us]")
    photos.to_parquet(SORTIE / "photos.parquet", index=False)
    couverture = {p: pd.Period(g["date_photo"].min(), "M") for p, g in photos.groupby("pays")}
    fin = pd.Period(photos["date_photo"].max(), "M")

    # ── Épisodes ────────────────────────────────────────────────────────────────────────────
    ep = ev[["pays", "id_episode", "medicament", "substance", "atc5", "debut", "fin", "en_cours", "type_episode",
             "gravite", "raison", "publication", "censure_gauche"]].copy()
    for c in ("pays", "type_episode", "gravite"):
        ep[c] = ep[c].astype("string")
    ep["cause"] = ep["raison"].map(famille_cause)
    ep["duree_j"] = (ep["fin"] - ep["debut"]).dt.days.where(~ep["en_cours"])
    ep.loc[ep["duree_j"] < 0, "duree_j"] = np.nan
    ep = ep.drop(columns="raison")
    ep.to_parquet(SORTIE / "episodes.parquet", index=False)
    pen = ep[ep["type_episode"] == "penurie"]

    # ── Substances : une ligne par ATC5 ─────────────────────────────────────────────────────
    suivies = set(panel["atc5"])
    codes = sorted(c for c in set(ep["atc5"].dropna()) | suivies | set(sig["atc5"].dropna())
                   if re.fullmatch(r"[A-Z]\d{2}[A-Z]{2}\d{2}", c))
    sub = pd.DataFrame({"atc5": codes})
    for niveau, n in [("atc1", 1), ("atc2", 3), ("atc3", 4), ("atc4", 5)]:
        sub[niveau] = sub["atc5"].str[:n]
    for niveau in ["atc1", "atc2", "atc3", "atc4", "atc5"]:
        libelles = atc.dropna(subset=[niveau]).drop_duplicates(niveau).set_index(niveau)[f"libelle_{niveau}"]
        sub[f"libelle_{niveau}"] = sub[niveau].map(libelles).str.capitalize()
    noms = pd.concat([ep[["atc5", "substance"]], sig[["atc5", "dci"]].rename(columns={"dci": "substance"})])
    noms = noms.dropna().groupby("atc5")["substance"].agg(lambda s: s.mode().iat[0])
    sub["nom"] = sub["libelle_atc5"].fillna(sub["atc5"].map(noms).str.capitalize()).fillna(sub["atc5"])
    sub["mitm"] = sub["atc5"].isin(suivies)
    sub["ue_critique"] = sub["atc5"].isin(set(ue["atc5"]))
    sub["nb_signalements"] = sub["atc5"].map(sig["atc5"].value_counts()).fillna(0).astype(int)
    comptes = pen.groupby(["atc5", "pays"]).size().unstack(fill_value=0).reindex(columns=PAYS, fill_value=0)
    for p in PAYS:
        sub[f"ep_{p}"] = sub["atc5"].map(comptes[p]).fillna(0).astype(int)
    sub["nb_pays_touches"] = (sub[[f"ep_{p}" for p in PAYS]] > 0).sum(axis=1)
    sub["duree_mediane_j"] = sub["atc5"].map(pen.groupby("atc5")["duree_j"].median())
    annee_conso = int(conso["annee"].max())
    derniere = conso[conso["annee"] == annee_conso].set_index("atc5")
    sub["boites"] = sub["atc5"].map(derniere["boites"])
    sub["nb_presentations"] = sub["atc5"].map(derniere["nb_presentations"])
    sub["score"] = sub["atc5"].map(prediction.set_index("atc5")["score"])
    sub["rang"] = sub["atc5"].map(prediction.set_index("atc5")["rang"]).astype("Int64")
    sub.to_parquet(SORTIE / "substances.parquet", index=False)

    # ── Séries mensuelles par pays : pénuries en cours et nouvelles ─────────────────────────
    actifs = mois_actifs(pen.dropna(subset=["debut"]), couverture, fin)
    serie = actifs.groupby(["pays", "mois"]).size().rename("en_cours").reset_index()
    nouv = pen[~pen["censure_gauche"]].assign(mois=pen["debut"].dt.to_period("M").dt.to_timestamp())
    nouv = nouv[nouv["mois"] >= nouv["pays"].map(lambda p: couverture[p].to_timestamp())]
    nouv = nouv.groupby(["pays", "mois"]).size().rename("nouvelles").reset_index()
    serie = serie.merge(nouv, on=["pays", "mois"], how="left").fillna({"nouvelles": 0})
    serie["nouvelles"] = serie["nouvelles"].astype(int)
    serie.to_parquet(SORTIE / "serie_pays.parquet", index=False)

    sig[["date_signalement", "annee", "laboratoire", "atc5"]].to_parquet(SORTIE / "signalements.parquet", index=False)
    conso[["atc5", "annee", "boites", "nb_presentations"]].to_parquet(SORTIE / "consommation.parquet", index=False)

    # ── Sous-classes ATC2 × mois : part des substances suivies touchées ─────────────────────
    taille = sub[sub["mitm"]].groupby("atc2").size()
    sig_m = sig[sig["atc5"].isin(suivies)].assign(mois=sig["date_signalement"].dt.to_period("M").dt.to_timestamp())
    sources = {"signalements": sig_m[["atc5", "mois"]].drop_duplicates(),
               "episodes_fr": actifs[(actifs["pays"] == "FR") & actifs["atc5"].isin(suivies)][["atc5", "mois"]]
               .drop_duplicates()}
    tables, ordres = [], {}
    for source, touches in sources.items():
        t = touches.assign(atc2=touches["atc5"].str[:3]).groupby(["atc2", "mois"]).size().rename("n").reset_index()
        t["part"] = t["n"] / t["atc2"].map(taille)
        t["source"] = source
        tables.append(t)
        # Une sous-classe d'une ou deux substances passe de 0 à 100 % en un signalement : on l'écarte.
        large = t.pivot(index="atc2", columns="mois", values="part").reindex(taille[taille >= MIN_TAILLE].index)
        ordres[source] = ordre_cluster(large.fillna(0))
    pd.concat(tables, ignore_index=True).to_parquet(SORTIE / "classes_mois.parquet", index=False)

    # ── Pénuries simultanées : P(pénurie en B | pénurie en A, même substance, même mois) ────
    act = actifs.dropna(subset=["atc5"]).drop_duplicates()
    cles = {p: set(zip(g["atc5"], g["mois"])) for p, g in act.groupby("pays")}
    mois_p = {p: set(g["mois"]) for p, g in act.groupby("pays")}
    cooc = []
    for a in PAYS:
        for b in PAYS:
            communs = mois_p[a] & mois_p[b]
            ka, kb = ({k for k in cles[x] if k[1] in communs} for x in (a, b))
            cooc.append({"a": a, "b": b, "p": len(ka & kb) / len(ka) if ka else np.nan, "n_a": len(ka),
                         "n_ab": len(ka & kb), "mois_communs": len(communs)})
    pd.DataFrame(cooc).to_parquet(SORTIE / "cooccurrence.parquet", index=False)

    # ── Panel : cible et signaux, mois par mois ─────────────────────────────────────────────
    pm = panel.assign(fr_autres=(panel["fr_autres_classe_3m"] > 0).astype("Int8"),
                      deja_signale=(panel["sig_cumul_avant"] > 0).astype("Int8"))
    resume = pm.groupby("mois").agg(lignes=("atc5", "size"), cible=("cible", "mean"),
                                    jeu=("jeu", lambda s: str(s.iat[0]))).reset_index()
    resume.to_parquet(SORTIE / "panel_mensuel.parquet", index=False)
    signaux = ["deja_signale", "fr_affichee_3m", "fr_autres", "it_affichee_3m", "de_affichee_3m", "ue_critique"]
    parts = pm.groupby("mois")[signaux].agg(lambda s: s.astype("float64").mean())  # tout vide → NaN (non couvert)
    parts = parts.astype("Float64").astype("float64")  # NaN numpy (pas pd.NA) pour plotly
    parts.reset_index().to_parquet(SORTIE / "panel_signaux.parquet", index=False)
    manquants = panel.drop(columns=["atc5", "mois", "cible", "annee", "mois_num", "jeu"]).isna().mean()

    # ── Modèle : prédictions de validation et de test, et prédiction du mois courant ───────
    # Avec la référence « signalée l'année précédente » (même formule que analyse/modeles.py), pour comparer
    histo = panel[["atc5", "mois", "sig_an_prec", "sig_cumul_avant"]]
    pred_eval = pred_eval.merge(histo, on=["atc5", "mois"], how="left")
    pred_eval["score_reference"] = (pred_eval["sig_an_prec"].astype("float64").fillna(0)
                                    + 1e-3 * pred_eval["sig_cumul_avant"].astype("float64").fillna(0))
    pred_eval = pred_eval.drop(columns=["sig_an_prec", "sig_cumul_avant"])
    pred_eval.to_parquet(SORTIE / "predictions_evaluation.parquet", index=False)
    prediction.drop(columns=["annee", "mois_num"]).to_parquet(SORTIE / "prediction_courante.parquet", index=False)

    meta = {"ordres_atc2": ordres, "min_taille_atc2": MIN_TAILLE,
            "couverture": {p: str(m) for p, m in couverture.items()}, "dernier_mois": str(fin),
            "annee_conso": annee_conso, "panel_lignes": int(len(panel)), "panel_substances": int(panel["atc5"].nunique()),
            "panel_mois": [str(panel["mois"].min().date())[:7], str(panel["mois"].max().date())[:7]],
            "episodes": int(len(ev)), "manquants_panel": {k: float(v) for k, v in manquants.items()},
            "metriques": metriques, "mois_prediction": str(prediction["mois"].iat[0].date())[:7]}
    (SORTIE / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    for f in sorted(SORTIE.glob("*")):
        print(f"{f.name:30s} {f.stat().st_size / 1024:8.0f} Ko")


if __name__ == "__main__":
    main()
