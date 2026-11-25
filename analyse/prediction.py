"""Étape 7 : la table de prédiction. Le même calcul que `panel_fr`, pour une seule date (le mois courant),
sans la cible (EXPLICATIONS.md, section 6, étape 7 ; cours 1, diapo 63).

Usage :
    python -m analyse.modeles        # d'abord : le modèle retenu dans models/
    python -m analyse.prediction     # mois courant = mois de la dernière photo des pages de pénurie
    python -m analyse.prediction 2026-06

Règle de temps (section 5.3) : pour le mois t, on n'utilise que ce qui était connu à la fin de t.
- signalements ANSM : années civiles précédentes seulement (publiés avec retard) ;
- consommation : année civile précédente ;
- liste critique de l'UE : inscription au plus tard à la fin de t ;
- pages de pénurie (France, Italie, Allemagne) : affichées pendant t-2, t-1 ou t (publication ≤ fin de t,
  et pas encore vues disparues avant le début de t-2). Ces variables ne servent pas au modèle retenu
  (variables de base) ; elles sont affichées à côté du score pour la lecture.

Contrôle : le même code appliqué au dernier mois de `panel_fr` (09/2025) doit redonner exactement ses
variables. Sortie : outputs/prediction_<mois>_<jour>.parquet, une ligne par substance de la cohorte, triée
par risque.
"""
import sys
from datetime import date

import joblib
import numpy as np
import pandas as pd

from analyse.modeles import MODELES, PANEL, RACINE, SORTIES, preparer
from nettoyage.panel import cohorte, lire

PAGES = {"fr_affichee_3m": "FR", "it_affichee_3m": "IT", "de_affichee_3m": "DE"}
COUVERTURE = {"FR": pd.Period("2021-06", "M"), "IT": pd.Period("2021-12", "M"), "DE": pd.Period("2024-09", "M")}


def affichee_3m(evenements, mois, substances):
    """Substances dont un épisode figure sur la page publiée pendant t-2, t-1 ou t."""
    debut_fenetre = (mois - 2).start_time
    visible = (evenements["publication"] <= mois.end_time) & (
        evenements["fin_connue_le"].isna() | (evenements["fin_connue_le"] >= debut_fenetre))
    return pd.Series(substances).isin(set(evenements.loc[visible, "atc5"].dropna())).to_numpy()


def variables(mois, substances=None):
    """Une ligne par substance de la cohorte : les colonnes de `panel_fr`, sans la cible, au mois `mois`."""
    substances = substances if substances is not None else cohorte()[0]
    signalements = lire("ansm_signalements").dropna(subset=["atc5"])
    evenements = lire("shortage_events")
    evenements = evenements[evenements["type_episode"] == "penurie"]
    t = pd.DataFrame({"atc5": substances, "mois": mois.to_timestamp().as_unit("us")})
    t["annee"] = np.int16(mois.year)
    t["mois_num"] = np.int8(mois.month)

    for nom, pays in PAGES.items():
        valeurs = affichee_3m(evenements[evenements["pays"] == pays], mois, substances).astype("int8")
        t[nom] = pd.array(valeurs, dtype="Int8") if mois >= COUVERTURE[pays] else pd.array([pd.NA] * len(t), dtype="Int8")
    # Cascade (Q1) : autres substances de la même ATC4 affichées en France, parmi toutes les substances connues
    toutes = sorted(set(substances) | set(signalements["atc5"]) | set(evenements["atc5"].dropna()))
    fr = pd.Series(affichee_3m(evenements[evenements["pays"] == "FR"], mois, toutes), index=toutes)
    par_classe = fr.groupby(fr.index.str[:5]).sum()
    autres = t["atc5"].str[:5].map(par_classe).fillna(0).to_numpy() - t["fr_affichee_3m"].fillna(0).to_numpy()
    t["fr_autres_classe_3m"] = pd.array(autres.astype("int16"), dtype="Int16")
    if mois < COUVERTURE["FR"]:
        t["fr_autres_classe_3m"] = pd.NA

    par_annee = signalements.groupby(["atc5", "annee"]).size()
    precedente = par_annee[par_annee.index.get_level_values("annee") == mois.year - 1].droplevel("annee")
    cumul = par_annee[par_annee.index.get_level_values("annee") < mois.year].groupby("atc5").sum()
    couvert = mois.year >= 2020
    t["sig_an_prec"] = pd.array(t["atc5"].map(precedente).fillna(0).to_numpy() if couvert else [pd.NA] * len(t),
                                dtype="Int32")
    t["sig_cumul_avant"] = pd.array(t["atc5"].map(cumul).fillna(0).to_numpy() if couvert else [pd.NA] * len(t),
                                    dtype="Int32")

    conso = lire("consommation")
    conso = conso[conso["annee"] == mois.year - 1].set_index("atc5")
    t["conso_boites_an_prec"] = pd.array(t["atc5"].map(conso["boites"]).to_numpy(), dtype="Int64")
    t["conso_presentations_an_prec"] = pd.array(t["atc5"].map(conso["nb_presentations"]).to_numpy(), dtype="Int64")

    ue = lire("ue_liste_critique").drop_duplicates("atc5").set_index("atc5")["date_inscription"]
    t["ue_critique"] = (t["atc5"].map(ue) <= mois.end_time).fillna(False).astype("int8")
    return t


def controler():
    """Le calcul pour une date redonne les variables de `panel_fr` au dernier mois du panel."""
    panel = pd.read_parquet(PANEL)
    dernier = pd.Period(panel["mois"].max(), "M")
    attendu = panel[panel["mois"] == panel["mois"].max()].sort_values("atc5").reset_index(drop=True)
    obtenu = variables(dernier, list(attendu["atc5"]))
    if list(obtenu["atc5"]) != list(attendu["atc5"]) or set(cohorte()[0]) != set(attendu["atc5"]):
        raise SystemExit("Contrôle en échec : la cohorte diffère de celle de panel_fr")
    ecarts = [c for c in obtenu.columns if c not in ("atc5", "mois")
              if not attendu[c].astype("float64").fillna(-1).equals(obtenu[c].astype("float64").fillna(-1))]
    if ecarts:
        raise SystemExit(f"Contrôle en échec : variables différentes de panel_fr au mois {dernier} : {ecarts}")
    print(f"  ✓ variables recalculées pour {dernier} identiques à panel_fr ({len(obtenu)} substances)")


def dernier_mois_observe():
    """Mois de la dernière photo des pages de pénurie : on ne connaît rien au-delà."""
    return pd.Period(lire("shortage_events")["publication"].max(), "M")


def main():
    mois = pd.Period(sys.argv[1], "M") if len(sys.argv) > 1 else dernier_mois_observe()
    controler()
    chemin = max(MODELES.glob("*.joblib"), key=lambda p: p.stat().st_mtime)
    paquet = joblib.load(chemin)
    t = variables(mois)
    x = preparer(t.assign(jeu="prediction", cible=0))
    t["score"] = paquet["modele"].predict_proba(x[paquet["variables"]])[:, 1]
    t = t.sort_values("score", ascending=False, ignore_index=True)
    t["rang"] = np.arange(1, len(t) + 1)
    SORTIES.mkdir(exist_ok=True)
    sortie = SORTIES / f"prediction_{mois}_{date.today().isoformat()}.parquet"
    t.to_parquet(sortie, index=False)
    print(f"Mois {mois} : {len(t)} substances, risque prédit pour {mois + 1} → {mois + 3}, modèle {chemin.name}")
    print(f"  score moyen {t['score'].mean():.3f} ; 100 premières : score ≥ {t['score'].iat[99]:.3f}")
    print(t.head(10)[["rang", "atc5", "score", "sig_an_prec", "fr_affichee_3m"]].to_string(index=False))
    print(f"→ {sortie.relative_to(RACINE)}")


if __name__ == "__main__":
    main()
