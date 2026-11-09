"""Contrôles des tables nettoyées, lancés en dernière étape de `python -m nettoyage`.

Une requête fausse ne donne pas d'erreur : elle donne un chiffre plausible. Chaque contrôle
vise une erreur silencieuse précise (cours « Building a Dataset from a Warehouse ») :
- grain : la clé annoncée identifie bien une seule ligne (sinon doublons, fan-out des jointures) ;
- codes ATC5 : 7 caractères, sinon un code de classe (niveau 4) passe pour une substance ;
- intervalles des dimensions historisées : pas de chevauchement, sinon une jointure as-of
  duplique des lignes sans message ;
- réconciliation : un total recalculé après jointure est égal au total de départ, et la part
  des lignes sans clé (ATC5 non trouvé) est mesurée et bornée ;
- dates de connaissance : publication et fin connue présentes, dans le bon ordre : ce sont les
  dates à utiliser pour les variables.
Le premier contrôle en échec arrête la construction, avec le détail. Si tout passe, le manifeste
des tables est copié dans manifestes/.
"""
import shutil
from datetime import date

import pandas as pd

from .commun import CLEAN, RACINE

# table : colonnes qui identifient une ligne
GRAINS = {
    "produits": ["cis"],
    "atc": ["atc5"],
    "consommation": ["atc5", "annee"],
    "consommation_mensuelle": ["atc5", "mois"],
    "cip_atc": ["cip13"],
    "ansm_fiches": ["fiche"],
    "ansm_episodes": ["fiche", "numero_episode"],
    "europe_episodes": ["pays", "id_episode"],
    "shortage_events": ["pays", "id_episode"],
    "penuries_fr_mensuel": ["atc5", "mois"],
    "ue_liste_critique": ["atc5"],
    "bdpm_specialites_historique": ["cis", "valid_from"],
    "bdpm_presentations_historique": ["cip13", "valid_from"],
    "bdpm_generiques_historique": ["groupe_generique", "cis", "valid_from"],
}
TABLES_ATC5 = ["consommation_mensuelle", "produits", "ansm_fiches", "ansm_signalements", "shortage_events", "penuries_fr_mensuel",
               "ue_liste_critique"]
HISTORISEES = {"bdpm_specialites_historique": ["cis"], "bdpm_presentations_historique": ["cip13"],
               "bdpm_generiques_historique": ["groupe_generique", "cis"]}


def lire(nom):
    return pd.read_parquet(CLEAN / f"{nom}.parquet")


def controler_grains():
    for table, cle in GRAINS.items():
        lignes = lire(table)
        doublons = lignes.duplicated(cle).sum()
        yield f"grain {table} ({', '.join(cle)})", doublons == 0, f"{doublons} doublons sur {len(lignes)} lignes"


def controler_atc5():
    for table in TABLES_ATC5:
        codes = lire(table)["atc5"].dropna()
        mauvais = codes[~codes.str.fullmatch(r"[A-Z]\d{2}[A-Z]{2}\d{2}")]
        yield f"ATC5 valides dans {table}", mauvais.empty, f"{len(mauvais)} codes invalides : {sorted(set(mauvais))[:5]}"


def controler_intervalles():
    for table, cle in HISTORISEES.items():
        versions = lire(table).sort_values([*cle, "valid_from"])
        fin = versions["valid_to"].fillna(pd.Timestamp.max)
        vides = (fin <= versions["valid_from"]).sum()
        # La version suivante du même objet doit commencer après la fin de celle-ci.
        debut_suivant = versions.groupby(cle)["valid_from"].shift(-1)
        chevauchements = (debut_suivant < fin).sum()
        yield (f"intervalles de {table}", vides == 0 and chevauchements == 0,
               f"{vides} intervalles vides, {chevauchements} chevauchements")
        # Au plus une version encore valable par objet (pas de « trou » à tester : un objet absent
        # d'une photo a vraiment disparu de la base à cette date).
        courantes = versions[versions["valid_to"].isna()].duplicated(cle).sum()
        yield f"une seule version courante par objet dans {table}", courantes == 0, f"{courantes} en trop"


def controler_reconciliation():
    """Troncature (cours, diapos 43-44) : les lignes passent d'une table à l'autre sans perte
    ni doublon, ou la perte est mesurée et reste sous un seuil (clés non résolues, diapo 27)."""
    signalements = lire("ansm_signalements")
    dans_le_panel = lire("penuries_fr_mensuel")["nb_signalements"].sum()
    yield ("signalements avec ATC5 = total du panel mensuel", signalements["atc5"].notna().sum() == dans_le_panel,
           f"{signalements['atc5'].notna().sum()} signalements, {dans_le_panel} dans le panel")

    evenements = lire("shortage_events")
    for source, pays in (("ansm_episodes", ["FR"]), ("europe_episodes", ["IT", "DE", "NO"])):
        attendu, obtenu = len(lire(source)), evenements["pays"].isin(pays).sum()
        yield f"{source} → shortage_events sans perte", attendu == obtenu, f"{attendu} épisodes, {obtenu} gardés"

    for table, seuil in (("ansm_signalements", 0.05), ("shortage_events", 0.10)):
        part = lire(table)["atc5"].isna().mean()
        yield f"part sans ATC5 dans {table} : {part:.1%} (seuil {seuil:.0%})", part <= seuil, f"{part:.1%}"
    # Fiches ANSM : l'ancien site (2015-2021) cite surtout des médicaments retirés depuis, souvent
    # hospitaliers (absents d'Open Medic) : seuil plus large, mesuré à part.
    fiches = lire("ansm_fiches")
    ancien = fiches["fiche"].str.startswith("ancien:")
    for nom, masque, seuil in (("site actuel", ~ancien, 0.10), ("ancien site", ancien, 0.25)):
        part = fiches.loc[masque, "atc5"].isna().mean()
        yield f"part sans ATC5 dans ansm_fiches, {nom} : {part:.1%} (seuil {seuil:.0%})", part <= seuil, f"{part:.1%}"


def controler_medicam():
    """Consommation mensuelle (Medic'AM) : mois continus, réconciliation avec les fichiers annuels et
    avec Open Medic, part des boîtes sans ATC5 mesurée (cours : troncature, clés non résolues)."""
    mensuelle = lire("consommation_mensuelle")
    attendus = pd.date_range(mensuelle["mois"].min(), mensuelle["mois"].max(), freq="MS")
    manquants = attendus.difference(mensuelle["mois"].unique())
    yield "aucun mois manquant dans consommation_mensuelle", len(manquants) == 0, f"manquants : {list(manquants.strftime('%Y-%m'))}"
    negatives = (mensuelle["boites"] < 0).sum()
    yield "boîtes négatives rares (régularisations)", negatives < 0.001 * len(mensuelle), f"{negatives} lignes sur {len(mensuelle)}"

    # Les fichiers annuels et semestriels viennent de la même source mais ne coïncident pas toujours
    # (2016 : -2,6 %, 2018 : +0,5 %, révisions de la source) : l'écart est borné, pas supprimé.
    rapprochement = lire("medicam_reconciliation")
    ecart = (rapprochement["boites_mensuel"] / rapprochement["boites_annuel"] - 1).abs()
    yield ("somme des 12 mois = fichier annuel (écart max 3 %)", ecart.max() <= 0.03,
           f"écart max {ecart.max():.1%} en {int(rapprochement.loc[ecart.idxmax(), 'annee'])}")
    yield "12 mois présents chaque année annuelle", (rapprochement["mois_presents"] == 12).all(), \
        f"{rapprochement['mois_presents'].tolist()}"

    # Boîtes sans ATC5 valide : environ 6 % avant 2021, moins de 2 % ensuite.
    nc = lire("medicam_non_classe")
    part = nc["boites_non_classees"] / nc["boites_total"]
    yield (f"part de boîtes sans ATC5 : max {part.max():.1%} (seuil 15 %)", part.max() <= 0.15, f"{part.max():.1%}")
    recent = part[nc["mois"] >= "2021-01-01"].max()
    yield f"part de boîtes sans ATC5 depuis 2021 : max {recent:.1%} (seuil 3 %)", recent <= 0.03, f"{recent:.1%}"

    # Même champ qu'Open Medic (ville, remboursé) : les totaux annuels coïncident à 2 % près.
    annuel = (mensuelle.assign(annee=mensuelle["mois"].dt.year).groupby("annee")["boites"].sum()
              .rename("mensuel").to_frame().join(lire("consommation").groupby("annee")["boites"].sum().rename("open_medic"),
                                                 how="inner"))
    rapport = (annuel["mensuel"] / annuel["open_medic"] - 1).abs()
    yield (f"total annuel Medic'AM = Open Medic (écart max {rapport.max():.1%}, seuil 2 %)", rapport.max() <= 0.02,
           f"{rapport.max():.1%}")


def controler_dates():
    """Dates de connaissance (fuite temporelle) : chaque épisode a une date de publication, et
    chaque épisode terminé une date à laquelle sa fin est connue, jamais avant sa publication."""
    evenements = lire("shortage_events")
    manquantes = evenements["publication"].isna().sum()
    yield "date de publication de chaque épisode", manquantes == 0, f"{manquantes} épisodes sans publication"
    termines = evenements[~evenements["en_cours"]]
    sans_date = termines["fin_connue_le"].isna().sum()
    yield "date de connaissance de la fin de chaque épisode terminé", sans_date == 0, f"{sans_date} sans date"
    avant = (termines["fin_connue_le"] < termines["publication"]).sum()
    yield "fin connue après la publication", avant == 0, f"{avant} épisodes"


def garder_le_manifeste():
    """Copie datée du manifeste dans manifestes/ (suivi par git) : il reste disponible pendant
    tout le projet pour comparer une exécution future à celle-ci (cours, diapos 65-66)."""
    dossier = RACINE / "manifestes"
    dossier.mkdir(exist_ok=True)
    copie = dossier / f"niveau2_{date.today().isoformat()}.json"
    shutil.copyfile(CLEAN / "manifeste.json", copie)
    print(f"  manifeste copié dans {copie.relative_to(RACINE)} (à commiter)")


def construire():
    for controles in (controler_grains, controler_atc5, controler_intervalles, controler_reconciliation,
                      controler_medicam, controler_dates):
        for nom, reussi, detail in controles():
            if not reussi:
                raise SystemExit(f"Contrôle en échec : {nom} — {detail}")
            print(f"  ✓ {nom}")
    garder_le_manifeste()
