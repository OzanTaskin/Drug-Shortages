"""Niveau 3 — `panel_fr` : la table d'entraînement, une ligne par substance (ATC5) et par mois t.

Panel COMPLET : aucune valeur vide. Les choix qui l'assurent (décidés avec les porteurs du projet) :
- période : t de 04/2020 à 09/2025. La page ANSM est couverte depuis 2015 (ancien site jusqu'en 03/2021,
  puis le site actuel) ; les signalements de 2019, première année publiée, sont lus à partir d'avril 2020.
  Le dernier t a ses trois mois de cible complets (signalements publiés jusqu'en 12/2025) ;
- cohorte : substances à médicament d'intérêt thérapeutique majeur (MITM) commercialisé (les signalements
  ne concernent que ces médicaments) ET délivrées en ville, donc présentes dans Medic'AM. Les substances
  réservées à l'hôpital (consommation inconnue) sont exclues ;
- signaux européens couverts sur une partie seulement de la période : la valeur vaut 0 hors couverture et
  une colonne `*_couvert` (0/1) dit si la source couvrait le mois. Un modèle doit lire les deux ensemble ;
  « 0 et couvert » veut dire « pas de pénurie publiée », « 0 et non couvert » veut dire « on ne sait pas ».
    Italie (`it_affichee_3m`) : photos mensuelles de la liste AIFA depuis 10/2021, couvert pour t ≥ 12/2021 ;
    Norvège (`no_nouvelle_3m`) : registre de la DMP, classeurs 2019 au 12/07/2024, couvert pour
    t ≤ 06/2024 ;
  la Suède (`se_nouvelle_3m`, Läkemedelsverket depuis 2018), la Tchéquie (`cz_nouvelle_3m`, SÚKL depuis 2018) et la
  Slovaquie (`sk_nouvelle_3m`, ŠÚKL depuis 04/2019) couvrent toute la période : pas d'indicateur de couverture ;
- l'Allemagne n'est pas dans le panel : 13 mois d'archives seulement (depuis 09/2024).

Cible `cible` : 1 si au moins un signalement de rupture est déclaré à l'ANSM pour la substance
entre t+1 et t+3.

Cible des NOUVELLES pénuries (la cible qui compte : `cible` est surtout prédite par la répétition, 97 % de ses
cas positifs en 2025 concernent une substance déjà signalée) :
- `calme_6m`, `calme_12m` : 1 si la substance n'a eu AUCUN signalement de t-5 à t (6 mois) ou de t-11 à t (12 mois) ;
- `cible_nouvelle_6m`, `cible_nouvelle_12m` : `cible` ET `calme_*` : une rupture qui arrive sur une substance calme.
On évalue la cible nouvelle sur les seules lignes calmes. ATTENTION : `calme_*` et `cible_nouvelle_*` lisent les
vraies dates des signalements (pas encore publiés à t) : ce sont des définitions de cible et de sous-population,
JAMAIS des variables d'entrée d'un modèle. Pas de seuil à 24 mois : les signalements commencent en 01/2019, les
24 mois précédents ne seraient pas couverts avant t = 01/2021.

Règle de temps : une variable d'une ligne (substance, t) n'utilise que ce qui était PUBLIÉ à la fin
du mois t. Un épisode de pénurie compte comme affiché de sa `publication` jusqu'à `fin_connue_le`
(date où l'on a vu qu'il avait disparu), jamais jusqu'à sa `fin` réelle, connue plus tard.
- Signalements ANSM : fichiers annuels publiés au plus tôt en avril de l'année suivante. À t, la dernière
  année publiée est l'année précédente si t est d'avril à décembre, l'année d'avant sinon.
- Consommation mensuelle (Medic'AM) : publiée par semestre, environ deux mois après sa fin (le 1er semestre
  fin août, le 2e fin février). À la fin du mois t, le dernier semestre connu finit en juin (t de septembre à
  décembre), en décembre de l'année précédente (t de mars à août) ou en juin de l'année précédente (t de
  janvier et février).
Pour les deux, les dates de publication sont des hypothèses documentées (data.gouv.fr, data.ansm) ; les
contrôles vérifient qu'aucune donnée postérieure n'est lue.

Variables ajoutées le 23/11/2026 (voir DICTIONNAIRE.md) :
- marché (`mkt_*`) : nombre de médicaments commercialisés, de présentations, de titulaires, de génériques et part
  d'injectables de la substance à la fin du mois t, lus dans la BDPM historisée (jointure as-of : une version n'est
  connue qu'à partir de la photo où on la voit) ; `mkt_connu` vaut 0 si aucun médicament de la substance n'a pu
  être rattaché à son ATC5 à cette date (environ 20 % des médicaments n'ont pas d'ATC5) ;
- Q1 normalisée : `taille_classe` (substances de la classe ATC4) et `fr_part_classe_3m` (part des autres
  substances de la classe affichées), car le nombre brut `fr_autres_classe_3m` croît avec la taille de la classe ;
- `fr_affichee_6m`, `fr_affichee_12m` : substance affichée sur la page ANSM au cours des 6 ou 12 derniers mois ;
- cible des nouvelles pénuries SANS fuite de définition : `calme_pub_*m` = ni affichée sur la page ANSM depuis k
  mois, ni signalée l'année publiée la plus récente (tout cela est connu à t), et `cible_nouvelle_pub_*m` ;
  contrairement à `calme_*m` (vraies dates des signalements, inconnues à t), c'est utilisable pour évaluer un
  modèle dans les conditions de la production ;
- fraîcheur de l'archive : `fr_age_photo_j`, `it_age_photo_j` (jours depuis la dernière photo connue à la fin du mois) ;
- `semestre` : bloc de 6 mois, pour une validation glissante (voir analyse/decoupage.py).

Découpage `jeu` (temporel, avec trois mois d'écart car la cible regarde trois mois en avant) :
entraînement t ≤ 09/2023, écart, validation 01/2024 à 09/2024, écart, test 01/2025 à 09/2025.
Les lignes d'écart restent dans la table (jeu = "ecart") mais ne servent à aucun jeu.
"""
import shutil
from datetime import date

import numpy as np
import pandas as pd

from .commun import CLEAN, RACINE, enregistrer

FINAL = RACINE / "data" / "final"
MOIS = pd.period_range("2019-01", "2025-12", freq="M")  # période couverte par les signalements
HORIZON = 3
T = len(MOIS) - HORIZON  # dernier t : ses 3 mois de cible sont complets
PAYS = ("FR", "IT")  # pays dont on lit la liste affichée ; l'Allemagne (depuis 09/2024) est hors panel
PREMIER_T = pd.Period("2020-04", "M")  # premier t où la dernière année de signalements publiée (2019) existe
DEBUT = int((PREMIER_T - MOIS[0]).n)
REGISTRES = ("NO", "SE", "CZ", "SK")  # registres avec date de publication : signal = nouvelle déclaration sur 3 mois
# Couverture des signaux européens : (premier t couvert, dernier t couvert), bornes comprises.
COUVERTURE = {"it": (pd.Period("2021-12", "M"), None), "no": (None, pd.Period("2024-06", "M"))}
CALME = (6, 12)  # mois sans signalement qui font une substance « calme »
DECOUPAGE = [("entrainement", "2020-04", "2023-09"), ("ecart", "2023-10", "2023-12"),
             ("validation", "2024-01", "2024-09"), ("ecart", "2024-10", "2024-12"), ("test", "2025-01", "2025-09")]


VARIABLES_CONSO = ["conso_6m_pub", "conso_tendance_6m", "conso_saison_cible"]
# Lissage des rapports : on ajoute K boîtes aux deux termes. Sans cela, une substance qui vend quelques boîtes
# donne des rapports aberrants (jusqu'à 1,8 million) ; avec, le rapport tend vers 1 quand le volume est faible,
# et il est défini pour toutes les substances vendues en ville.
LISSAGE_SEMESTRE = 100
LISSAGE_MOIS = 10


ORDRE_COLONNES = [
    "atc5", "mois", "annee", "mois_num", "semestre", "jeu",
    "cible", "calme_6m", "cible_nouvelle_6m", "calme_12m", "cible_nouvelle_12m",
    "calme_pub_6m", "cible_nouvelle_pub_6m", "calme_pub_12m", "cible_nouvelle_pub_12m",
    "fr_affichee_3m", "fr_affichee_6m", "fr_affichee_12m", "fr_autres_classe_3m", "fr_part_classe_3m", "taille_classe",
    "fr_age_photo_j", "it_couvert", "it_affichee_3m", "it_age_photo_j", "no_couvert", "no_nouvelle_3m",
    "se_nouvelle_3m", "cz_nouvelle_3m", "sk_nouvelle_3m", "sig_annee_pub", "sig_cumul_pub",
    "conso_6m_pub", "conso_tendance_6m", "conso_saison_cible",
    "mkt_nb_specialites", "mkt_nb_presentations", "mkt_nb_titulaires", "mkt_nb_generiques", "mkt_part_injectable", "mkt_hhi_titulaires",
    "mkt_connu", "prix_boite_pub", "prix_tendance_pub", "fr_mois_depuis_affichage", "fr_nb_episodes_publies",
    "sig_mois_depuis_pub", "ue_critique"]


def lire(nom):
    return pd.read_parquet(CLEAN / f"{nom}.parquet")


def cohorte():
    """Substances avec au moins un MITM commercialisé, et celles sans."""
    produits, mitm = lire("produits"), lire("bdpm_mitm")
    commercialises = produits[(produits["etat_commercialisation"] == "Commercialisée") & produits["atc5"].str.len().eq(7)]
    avec = set(commercialises[commercialises["cis"].isin(mitm["cis"])]["atc5"])
    return sorted(avec), sorted(set(commercialises["atc5"]) - avec)


def presence_mensuelle(table, colonne_date, substances):
    """Matrice substances × mois (MOIS) : vrai si au moins une ligne dans le mois."""
    croisement = pd.crosstab(table["atc5"], table[colonne_date].dt.to_period("M"))
    return (croisement.reindex(index=substances, columns=MOIS, fill_value=0) > 0).to_numpy()


def affichage_mensuel(episodes, substances):
    """Matrice substances × mois : l'épisode figure sur la liste publiée pendant le mois.
    De la `publication` jusqu'au mois de `fin_connue_le` compris (jusqu'à la fin si l'on n'a pas vu sa fin)."""
    matrice = np.zeros((len(substances), len(MOIS)), dtype=bool)
    ligne = {s: i for i, s in enumerate(substances)}
    episodes = episodes.dropna(subset=["atc5", "publication"])
    def indice(dates):
        return (dates.dt.year - MOIS[0].year) * 12 + dates.dt.month - MOIS[0].month
    debut = indice(episodes["publication"]).clip(lower=0).to_numpy()
    fin = indice(episodes["fin_connue_le"]).fillna(len(MOIS) - 1).astype(int).to_numpy()
    for atc5, a, b in zip(episodes["atc5"], debut, fin):
        if atc5 in ligne and b >= a:
            matrice[ligne[atc5], a:min(b, len(MOIS) - 1) + 1] = True
    return matrice


def sur_k_mois(matrice, k):
    """Vrai à t si vrai à l'un des mois t-k+1, …, t."""
    sortie = matrice.copy()
    for decalage in range(1, k):
        sortie[:, decalage:] |= matrice[:, :-decalage]
    return sortie


def sur_3_mois(matrice):
    """Vrai à t si vrai à l'un des mois t-2, t-1, t."""
    return sur_k_mois(matrice, 3)


def autres_de_la_classe(matrice, substances):
    """Nombre d'autres substances de la même classe ATC4 (5 premiers caractères) affichées."""
    classe = np.array([s[:5] for s in substances])
    sortie = np.zeros(matrice.shape, dtype=np.int16)
    for c in sorted(set(classe)):
        membres = np.where(classe == c)[0]
        total = matrice[membres].sum(axis=0)
        sortie[membres] = total - matrice[membres]
    return sortie


def variables_page(evenements, substances):
    """Variables issues des pénuries publiées, sur les 3 derniers mois : listes affichées (France, Italie)
    et nouvelles déclarations norvégiennes (date de déclaration seule : la fin d'un épisode norvégien n'est
    connue qu'au classeur annuel suivant)."""
    pages = {pays: sur_3_mois(affichage_mensuel(evenements[evenements["pays"] == pays], substances)) for pays in PAYS}
    nouvelles = {f"{pays.lower()}_nouvelle_3m": sur_3_mois(presence_mensuelle(
        evenements[evenements["pays"] == pays].dropna(subset=["atc5"]), "publication", substances)) for pays in REGISTRES}
    return {"fr_affichee_3m": pages["FR"], "fr_autres_classe_3m": autres_de_la_classe(pages["FR"], substances),
            "it_affichee_3m": pages["IT"], **nouvelles,
            "fr_affichee_6m": sur_k_mois(affichage_mensuel(evenements[evenements["pays"] == "FR"], substances), 6),
            "fr_affichee_12m": sur_k_mois(affichage_mensuel(evenements[evenements["pays"] == "FR"], substances), 12)}


def couvert(prefixe, mois):
    """1 si le signal européen `prefixe` est couvert au mois t (bornes de COUVERTURE)."""
    premier, dernier = COUVERTURE[prefixe]
    return np.array([(premier is None or t >= premier) and (dernier is None or t <= dernier) for t in mois])


def matrice_conso(mensuelle, substances, valeur="boites"):
    """Boîtes remboursées : matrice substances × mois (Medic'AM), et substances présentes dans Medic'AM
    (les autres, souvent réservées à l'hôpital, restent vides : ni zéro, ni imputées)."""
    par_mois = mensuelle.assign(m=mensuelle["mois"].dt.to_period("M")).pivot_table(
        index="atc5", columns="m", values=valeur, aggfunc="sum")
    mois_c = pd.period_range(par_mois.columns.min(), par_mois.columns.max(), freq="M")
    presentes = np.array([s in par_mois.index for s in substances])
    boites = par_mois.reindex(index=substances, columns=mois_c).fillna(0).to_numpy(float)
    # Les régularisations (72 lignes de boîtes négatives) sont ramenées à 0 : on mesure une demande.
    return np.clip(boites, 0, None), presentes, mois_c


def dernier_mois_publie(t):
    """Dernier mois de consommation publié à la fin du mois t (semestres publiés fin août et fin février)."""
    if t.month >= 9:
        return pd.Period(f"{t.year}-06", "M")
    if t.month >= 3:
        return pd.Period(f"{t.year - 1}-12", "M")
    return pd.Period(f"{t.year - 1}-06", "M")


def variables_conso(C, presentes, mois_c, mois_t):
    """Trois variables de demande par substance et par mois t, sans lire après le dernier mois publié :
    - conso_6m_pub        boîtes du dernier semestre publié ;
    - conso_tendance_6m   ce semestre divisé par le même semestre un an avant (>1 : la demande monte) ;
    - conso_saison_cible  boîtes moyennes des mois t-11 à t-9 (un an avant la fenêtre de la cible, t+1 à t+3)
                          divisées par la moyenne des 12 mois t-20 à t-9 (>1 : trimestre de demande forte).
    Rapports lissés (LISSAGE_SEMESTRE, LISSAGE_MOIS)."""
    sortie = {nom: np.full((C.shape[0], len(mois_t)), np.nan) for nom in VARIABLES_CONSO}
    for j, t in enumerate(mois_t):
        fin = int((dernier_mois_publie(t) - mois_c[0]).n)
        ti = int((t - mois_c[0]).n)
        semestre = C[:, fin - 5:fin + 1].sum(axis=1)
        precedent = C[:, fin - 17:fin - 11].sum(axis=1)
        trimestre, annee = C[:, ti - 11:ti - 8].mean(axis=1), C[:, ti - 20:ti - 8].mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sortie["conso_6m_pub"][:, j] = semestre
            sortie["conso_tendance_6m"][:, j] = (semestre + LISSAGE_SEMESTRE) / (precedent + LISSAGE_SEMESTRE)
            sortie["conso_saison_cible"][:, j] = (trimestre + LISSAGE_MOIS) / (annee + LISSAGE_MOIS)
    for tableau in sortie.values():
        tableau[~presentes] = np.nan
    return sortie


def taille_de_classe(substances):
    """Nombre de substances (de l'univers `tout`) dans la classe ATC4 de chaque substance."""
    classe = np.array([s[:5] for s in substances])
    valeurs, comptes = np.unique(classe, return_counts=True)
    return comptes[np.searchsorted(valeurs, classe)]


def correspondance_cis_atc5():
    """CIS → ATC5 pour les médicaments d'hier comme d'aujourd'hui : `produits` (BDPM du jour), puis les CIP13 d'Open
    Medic pour les médicaments retirés depuis (sinon le marché passé paraîtrait plus petit qu'il n'était)."""
    produits = lire("produits").dropna(subset=["atc5"]).set_index("cis")["atc5"]
    cip_atc = lire("cip_atc")
    par_cip = (lire("bdpm_presentations_historique").merge(cip_atc[cip_atc["atc5"].str.len() == 7][["cip13", "atc5"]], on="cip13")
               .groupby(["cis", "atc5"]).size().rename("n").reset_index().sort_values(["cis", "n"], ascending=[True, False])
               .drop_duplicates("cis").set_index("cis")["atc5"])
    return produits.combine_first(par_cip)


INJECTABLE = "intraveineuse|sous-cutan|intramusculaire|intrath|intradermique|péridur|intravitr|injectable"
VARIABLES_MARCHE = ["mkt_nb_specialites", "mkt_nb_presentations", "mkt_nb_titulaires", "mkt_nb_generiques",
                    "mkt_part_injectable", "mkt_hhi_titulaires", "mkt_connu"]


def versions_valides(table, fin):
    """Versions de la dimension connues et valables à la fin du mois : vues au plus tard à `fin`, pas encore remplacées."""
    return table[(table["valid_from"] <= fin) & (table["valid_to"].isna() | (table["valid_to"] > fin))]


def variables_marche(substances, mois):
    """Structure du marché de chaque substance à la fin du mois t, d'après la BDPM historisée (jointure as-of).
    Valeur 0 et `mkt_connu` = 0 quand aucun médicament de la substance n'est rattaché à son ATC5 à cette date."""
    cis_atc = correspondance_cis_atc5()
    avec_atc = lambda table: table.assign(atc5=table["cis"].map(cis_atc)).dropna(subset=["atc5"])
    specialites = lire("bdpm_specialites_historique")
    specialites = avec_atc(specialites[specialites["commercialise"]])
    presentations = lire("bdpm_presentations_historique")
    presentations = avec_atc(presentations[presentations["commercialisee"]])
    generiques = lire("bdpm_generiques_historique")
    generiques = avec_atc(generiques[generiques["type_generique"] == "generique"])
    specialites["injectable"] = specialites["voies"].fillna("").str.contains(INJECTABLE, regex=True)
    ligne = {s: i for i, s in enumerate(substances)}
    sortie = {nom: np.zeros((len(substances), len(mois))) for nom in VARIABLES_MARCHE}
    for j, t in enumerate(mois):
        fin = t.end_time
        sp, pr, ge = (versions_valides(x, fin) for x in (specialites, presentations, generiques))
        compte = lambda serie, nom: serie.reindex(substances).fillna(0).to_numpy()
        nb = sp.groupby("atc5").size()
        sortie["mkt_nb_specialites"][:, j] = compte(nb, "")
        sortie["mkt_nb_presentations"][:, j] = compte(pr.groupby("atc5").size(), "")
        sortie["mkt_nb_titulaires"][:, j] = compte(sp.groupby("atc5")["titulaires"].nunique(), "")
        sortie["mkt_nb_generiques"][:, j] = compte(ge.groupby("atc5").size(), "")
        injectables = compte(sp[sp["injectable"]].groupby("atc5").size(), "")
        total = sortie["mkt_nb_specialites"][:, j]
        sortie["mkt_part_injectable"][:, j] = np.divide(injectables, total, out=np.zeros_like(total), where=total > 0)
        # Concentration (indice de Herfindahl) des médicaments entre titulaires : 1 = un seul titulaire.
        parts = sp.groupby(["atc5", "titulaires"]).size()
        parts = parts / parts.groupby(level=0).transform("sum")
        sortie["mkt_hhi_titulaires"][:, j] = compte((parts ** 2).groupby(level=0).sum(), "")
        sortie["mkt_connu"][:, j] = total > 0
    return sortie


def jours_depuis_derniere_photo(dates_photo, mois):
    """Jours entre la fin du mois t et la dernière photo connue à cette date (plafonné : 9999 s'il n'y en a pas encore)."""
    dates = np.sort(pd.to_datetime(pd.Series(dates_photo)).unique())
    fins = np.array([t.end_time.normalize().to_datetime64() for t in mois])
    indice = np.searchsorted(dates, fins, side="right") - 1
    age = (fins - dates[np.clip(indice, 0, None)]) / np.timedelta64(1, "D")
    return np.where(indice < 0, 9999, age).astype("int16")


VARIABLES_PRIX = ["prix_boite_pub", "prix_tendance_pub"]


def variables_prix(B, C, mois_c, mois_t):
    """Prix moyen d'une boîte (base de remboursement / boîtes) sur le dernier semestre publié de Medic'AM, et son
    évolution sur un an (rapport au même semestre de l'année précédente ; 1 si l'un des deux est inconnu). Un prix
    bas ou en baisse est un facteur de pénurie connu (génériques peu rentables). Même règle de publication que la
    consommation : rien après le dernier mois publié."""
    sortie = {nom: np.zeros((C.shape[0], len(mois_t))) for nom in VARIABLES_PRIX}
    for j, t in enumerate(mois_t):
        fin = int((dernier_mois_publie(t) - mois_c[0]).n)
        b, c = B[:, fin - 5:fin + 1].sum(axis=1), C[:, fin - 5:fin + 1].sum(axis=1)
        b0, c0 = B[:, fin - 17:fin - 11].sum(axis=1), C[:, fin - 17:fin - 11].sum(axis=1)
        prix = np.divide(b, c, out=np.zeros_like(b), where=c > 0)
        avant = np.divide(b0, c0, out=np.zeros_like(b0), where=c0 > 0)
        sortie["prix_boite_pub"][:, j] = prix
        sortie["prix_tendance_pub"][:, j] = np.divide(prix, avant, out=np.ones_like(prix), where=(prix > 0) & (avant > 0))
    return sortie


PLAFOND_MOIS = 99  # « jamais vu depuis le début de la source » pour les variables de récence


def mois_depuis_vrai(matrice):
    """Pour chaque case : mois écoulés depuis le dernier mois vrai (0 si vrai ce mois-ci), PLAFOND_MOIS sinon."""
    indices = np.arange(matrice.shape[1])[None, :]
    dernier = np.maximum.accumulate(np.where(matrice, indices, -1), axis=1)
    return np.where(dernier >= 0, indices - dernier, PLAFOND_MOIS)


def mois_depuis_signalement_publie(signalements, substances, mois):
    """Mois entre t et le dernier signalement ANSM d'une année déjà publiée à t (PLAFOND_MOIS s'il n'y en a pas)."""
    dates = {a: np.sort(g.dt.to_period("M").astype("int64").to_numpy()) for a, g in signalements.groupby("atc5")["date_signalement"]}
    sortie = np.full((len(substances), len(mois)), PLAFOND_MOIS, dtype="int16")
    annees = derniere_annee_publiee(np.array([t.year for t in mois]), np.array([t.month for t in mois]))
    bornes = np.array([pd.Period(f"{a}-12", "M").ordinal for a in annees])
    ordinaux = np.array([t.ordinal for t in mois])
    for i, s in enumerate(substances):
        d = dates.get(s)
        if d is None or not len(d):
            continue
        k = np.searchsorted(d, bornes, side="right") - 1
        ok = k >= 0
        sortie[i, ok] = np.minimum(ordinaux[ok] - d[k[ok]], PLAFOND_MOIS)
    return sortie


def episodes_fr_publies(evenements, substances, mois):
    """Nombre cumulé d'épisodes de pénurie publiés sur la page ANSM (depuis 2015) jusqu'à la fin du mois t."""
    fr = evenements[(evenements["pays"] == "FR")].dropna(subset=["atc5", "publication"])
    debut = min(fr["publication"].min().to_period("M"), mois[0])
    colonnes = pd.period_range(debut, mois[-1], freq="M")
    comptes = pd.crosstab(fr["atc5"], fr["publication"].dt.to_period("M")).reindex(index=substances, columns=colonnes, fill_value=0)
    return comptes.cumsum(axis=1)[list(mois)].to_numpy()


def derniere_annee_publiee(annee, mois_num):
    """Dernière année de signalements publiée à la fin du mois t : les fichiers annuels paraissent au plus tôt
    en avril de l'année suivante (data.ansm : mise à jour annuelle en avril)."""
    return annee - np.where(mois_num >= 4, 1, 2)


def construire():
    avec_total, sans = cohorte()
    mensuelle = lire("consommation_mensuelle")
    en_ville = set(mensuelle["atc5"])
    avec = [s for s in avec_total if s in en_ville]  # cohorte : MITM délivrés en ville (présents dans Medic'AM)
    signalements = lire("ansm_signalements").dropna(subset=["atc5"])
    evenements = lire("shortage_events")
    evenements = evenements[evenements["type_episode"] == "penurie"]
    # Les substances hospitalières restent dans les calculs de classe (autres substances affichées de l'ATC4).
    tout = sorted(set(avec_total) | set(sans) | set(signalements["atc5"]) | set(evenements["atc5"].dropna()))
    ligne = {s: i for i, s in enumerate(tout)}
    lignes = [ligne[s] for s in avec]
    mois = MOIS[DEBUT:T]
    nT = len(mois)

    S = presence_mensuelle(signalements, "date_signalement", tout)
    cible = (S[:, DEBUT + 1:T + 1] | S[:, DEBUT + 2:T + 2] | S[:, DEBUT + 3:T + 3])[lignes]

    table = pd.DataFrame({"atc5": np.repeat(avec, nT), "mois": np.tile(mois.to_timestamp(), len(avec)),
                          "cible": cible.ravel().astype("int8")})
    # Substances calmes (aucun signalement sur les k derniers mois, t compris) et cible des nouvelles pénuries.
    for k in CALME:
        if DEBUT - k + 1 < 0:
            raise SystemExit(f"calme_{k}m : les {k} mois avant {PREMIER_T} ne sont pas couverts par les signalements")
        recent = np.zeros((len(tout), nT), dtype=bool)
        for decalage in range(k):
            recent |= S[:, DEBUT - decalage:T - decalage]
        calme = (~recent)[lignes].ravel()
        table[f"calme_{k}m"] = calme.astype("int8")
        table[f"cible_nouvelle_{k}m"] = (table["cible"].to_numpy().astype(bool) & calme).astype("int8")
    table["annee"] = table["mois"].dt.year.astype("int16")
    table["mois_num"] = table["mois"].dt.month.astype("int8")
    table["jeu"] = pd.Series("ecart", index=table.index)
    for jeu, a, b in DECOUPAGE:
        table.loc[(table["mois"] >= pd.Period(a, "M").to_timestamp()) & (table["mois"] <= pd.Period(b, "M").to_timestamp()), "jeu"] = jeu
    table["jeu"] = table["jeu"].astype(pd.CategoricalDtype(["entrainement", "validation", "test", "ecart"]))

    # Pénuries publiées : France couverte sur tout le panel ; Italie et Norvège à 0 hors couverture,
    # avec leur indicateur de couverture.
    pages = variables_page(evenements, tout)
    for nom, matrice in pages.items():
        type_entier = "int16" if nom == "fr_autres_classe_3m" else "int8"
        valeurs = matrice[lignes][:, DEBUT:T].astype(type_entier)
        if nom[:2] in COUVERTURE:
            masque = couvert(nom[:2], mois)
            valeurs = valeurs * masque
            table[f"{nom[:2]}_couvert"] = np.tile(masque, len(avec)).astype("int8")
        table[nom] = valeurs.ravel()
    # Q1 normalisée : le compte brut des autres substances affichées croît avec la taille de la classe.
    taille = taille_de_classe(tout)
    autres = pages["fr_autres_classe_3m"].astype(float)
    part = np.divide(autres, (taille - 1)[:, None], out=np.zeros_like(autres), where=(taille > 1)[:, None])
    table["taille_classe"] = np.repeat(taille[lignes], nT).astype("int16")
    table["fr_part_classe_3m"] = part[lignes][:, DEBUT:T].clip(0, 1).ravel().astype("float32").round(4)
    # Fraîcheur des archives à la fin du mois t (la page n'est connue qu'à la dernière photo).
    photos_fr = lire("ansm_observations")["date_photo"]
    europe = lire("europe_observations")
    table["fr_age_photo_j"] = np.tile(jours_depuis_derniere_photo(photos_fr, mois), len(avec))
    table["it_age_photo_j"] = np.tile(jours_depuis_derniere_photo(europe.loc[europe["pays"] == "IT", "date_photo"], mois), len(avec))
    # Récence et historique publiés (variables « months_since_last_shortage » et « prior_episode_count » des modèles
    # de référence) : depuis la page ANSM (publiée en continu) et les signalements des années déjà publiées.
    fr_page = affichage_mensuel(evenements[evenements["pays"] == "FR"], tout)
    table["fr_mois_depuis_affichage"] = mois_depuis_vrai(fr_page)[lignes][:, DEBUT:T].ravel().astype("int16")
    table["fr_nb_episodes_publies"] = episodes_fr_publies(evenements, avec, mois).ravel().astype("int16")
    table["sig_mois_depuis_pub"] = mois_depuis_signalement_publie(signalements, avec, mois).ravel()
    table["semestre"] = pd.Categorical(table["mois"].dt.year.astype(str) + "S" + np.where(table["mois"].dt.month <= 6, "1", "2"))

    # Signalements des années déjà publiées (dernière année publiée, et cumul jusqu'à elle)
    par_annee = (signalements.groupby(["atc5", "annee"]).size().unstack(fill_value=0)
                 .reindex(index=avec, columns=range(2019, 2026), fill_value=0))
    cumul = par_annee.cumsum(axis=1)
    a_pub = derniere_annee_publiee(table["annee"].to_numpy(), table["mois_num"].to_numpy()) - 2019
    base = np.repeat(np.arange(len(avec)), nT)
    table["sig_annee_pub"] = par_annee.to_numpy()[base, a_pub].astype("int32")
    table["sig_cumul_pub"] = cumul.to_numpy()[base, a_pub].astype("int32")

    # Nouvelles pénuries sans fuite de définition : « calme » avec ce qui est publié à la fin du mois t.
    for k in CALME:
        publie = (table[f"fr_affichee_{k}m"] == 0) & (table["sig_annee_pub"] == 0)
        table[f"calme_pub_{k}m"] = publie.astype("int8")
        table[f"cible_nouvelle_pub_{k}m"] = (table["cible"].astype(bool) & publie).astype("int8")

    # Structure du marché (BDPM historisée)
    for nom, matrice in variables_marche(avec, mois).items():
        table[nom] = matrice.ravel().astype("float32" if nom in ("mkt_part_injectable", "mkt_hhi_titulaires") else "int32")

    # Demande mensuelle (Medic'AM, 2015 à 06/2026) : niveau, tendance et saison, sur les semestres déjà publiés
    C, presentes, mois_c = matrice_conso(mensuelle, tout)
    for nom, matrice in variables_conso(C, presentes, mois_c, mois).items():
        valeurs = matrice[lignes].ravel()
        table[nom] = valeurs.astype("int64") if nom == "conso_6m_pub" else valeurs.round(4).astype("float32")

    # Prix (base de remboursement par boîte, Medic'AM) sur le dernier semestre publié
    B, _, _ = matrice_conso(mensuelle, tout, "base_remboursement")
    for nom, matrice in variables_prix(B, C, mois_c, mois).items():
        table[nom] = matrice[lignes].ravel().round(4).astype("float32")

    # Criticité : substance inscrite à la liste critique de l'UE à la fin du mois t
    ue = lire("ue_liste_critique").drop_duplicates("atc5").set_index("atc5")["date_inscription"]
    inscrite = table["atc5"].map(ue)
    table["ue_critique"] = (inscrite <= table["mois"] + pd.offsets.MonthEnd(0)).fillna(False).astype("int8")

    table = table[ORDRE_COLONNES].sort_values(["atc5", "mois"], ignore_index=True)
    FINAL.mkdir(parents=True, exist_ok=True)
    enregistrer(table, "panel_fr", dossier=FINAL)
    controler(table, avec, evenements, tout, C, presentes, mois_c)
    dossier = RACINE / "manifestes"
    dossier.mkdir(exist_ok=True)
    copie = dossier / f"niveau3_{date.today().isoformat()}.json"
    shutil.copyfile(FINAL / "manifeste.json", copie)
    print(f"  manifeste copié dans {copie.relative_to(RACINE)} (à commiter)")


def controler(table, avec, evenements, tout, C, presentes, mois_c):
    """Contrôles du niveau 3 ; le premier en échec arrête la construction."""
    def verifier(nom, reussi, detail):
        if not reussi:
            raise SystemExit(f"Contrôle en échec : {nom} — {detail}")
        print(f"  ✓ {nom}")

    nT = T - DEBUT
    doublons = table.duplicated(["atc5", "mois"]).sum()
    verifier("grain panel_fr (atc5, mois)", doublons == 0, f"{doublons} doublons")
    verifier("une ligne par substance de la cohorte et par mois", len(table) == len(avec) * nT,
             f"{len(table)} lignes, attendu {len(avec) * nT}")
    vides = int(table.isna().sum().sum())
    verifier("panel complet : aucune valeur vide", vides == 0, f"{vides} valeurs vides : {table.isna().sum()[lambda x: x > 0].to_dict()}")
    verifier(f"premier mois = {PREMIER_T}", table["mois"].min() == PREMIER_T.to_timestamp(), f"{table['mois'].min()}")
    for prefixe, (premier, dernier) in COUVERTURE.items():
        hors = table[f"{prefixe}_couvert"] == 0
        signal = [c for c in table.columns if c.startswith(prefixe + "_") and c != f"{prefixe}_couvert"][0]
        verifier(f"{signal} = 0 hors couverture", (table.loc[hors, signal] == 0).all(), "valeur hors couverture")
        mois_couverts = table.loc[~hors, "mois"]
        verifier(f"{prefixe}_couvert dans ses bornes", (premier is None or mois_couverts.min() >= premier.to_timestamp())
                 and (dernier is None or mois_couverts.max() <= dernier.to_timestamp()), "bornes non respectées")
    verifier("dernière année de signalements lue ≥ 2019 (première publiée)",
             derniere_annee_publiee(table["annee"].to_numpy(), table["mois_num"].to_numpy()).min() >= 2019, "année absente lue")
    for k in CALME:
        nouvelle, calme = table[f"cible_nouvelle_{k}m"], table[f"calme_{k}m"]
        verifier(f"cible_nouvelle_{k}m = cible sur les lignes calmes, 0 ailleurs",
                 (nouvelle == table["cible"] * calme).all(), "écart")
    # Recalcul indépendant de calme_12m sur un échantillon, à partir des dates des signalements.
    sig = lire("ansm_signalements").dropna(subset=["atc5"])
    mois_sig = sig.assign(m=sig["date_signalement"].dt.to_period("M")).groupby("atc5")["m"].apply(set)
    echantillon = table.sample(2000, random_state=2)
    attendu = [int(not any((t.to_period("M") - j) in mois_sig.get(a, set()) for j in range(12)))
               for a, t in zip(echantillon["atc5"], echantillon["mois"])]
    verifier("calme_12m = aucun signalement de t-11 à t (recalcul)", list(echantillon["calme_12m"]) == attendu, "écart")
    taux = table["cible"].mean()
    verifier("taux de cas positifs plausible (15 % à 40 %)", 0.15 < taux < 0.40, f"{taux:.3f}")
    for avant, apres in [("entrainement", "validation"), ("validation", "test")]:
        fin_avant = table.loc[table["jeu"] == avant, "mois"].max()
        debut_apres = table.loc[table["jeu"] == apres, "mois"].min()
        ecart = (debut_apres.year - fin_avant.year) * 12 + debut_apres.month - fin_avant.month
        verifier(f"écart d'au moins {HORIZON} mois entre {avant} et {apres}", ecart > HORIZON, f"{ecart} mois")

    # Signalements : recalcul indépendant avec la règle de publication (année précédente à partir d'avril).
    signalements = lire("ansm_signalements").dropna(subset=["atc5"])
    par = signalements.groupby(["atc5", "annee"]).size()
    echantillon = table.sample(2000, random_state=0)
    a_pub = [int(y - (1 if m >= 4 else 2)) for y, m in zip(echantillon["annee"], echantillon["mois_num"])]
    attendu = [int(par.get((a, y), 0)) for a, y in zip(echantillon["atc5"], a_pub)]
    verifier("sig_annee_pub = signalements de la dernière année publiée", list(echantillon["sig_annee_pub"]) == attendu, "écart")
    cumul = [int(sum(par.get((a, k), 0) for k in range(2019, y + 1))) for a, y in zip(echantillon["atc5"], a_pub)]
    verifier("sig_cumul_pub = cumul jusqu'à la dernière année publiée", list(echantillon["sig_cumul_pub"]) == cumul, "écart")
    verifier("aucune année de signalements lue avant sa publication (avril)",
             all(y < a.year or (y == a.year - 1 and a.month >= 4) or y <= a.year - 2
                 for a, y in zip(echantillon["mois"], a_pub)), "année non publiée lue")

    # Aucune fuite : recalculer les variables en ne gardant que ce qui était publié à la fin de t0
    # doit redonner exactement les valeurs du mois t0 (un épisode terminé après t0 y est « en cours »).
    t0 = pd.Period("2024-06", "M")
    fin_t0 = t0.end_time
    tronque = evenements[evenements["publication"] <= fin_t0].copy()
    tronque.loc[tronque["fin_connue_le"] > fin_t0, "fin_connue_le"] = pd.NaT
    index_t0 = int((t0 - MOIS[0]).n)
    lignes = [tout.index(s) for s in avec]
    complet, partiel = variables_page(evenements, tout), variables_page(tronque, tout)
    for nom in complet:
        egal = np.array_equal(complet[nom][lignes][:, index_t0], partiel[nom][lignes][:, index_t0])
        verifier(f"{nom} au mois {t0} ne lit rien de postérieur", egal, "valeur différente après troncature")

    # Variables ajoutées : cohérence et recalcul indépendant.
    verifier("colonnes du panel = ordre documenté", list(table.columns) == ORDRE_COLONNES, "colonnes différentes")
    for k in CALME:
        publie = (table[f"fr_affichee_{k}m"] == 0) & (table["sig_annee_pub"] == 0)
        verifier(f"calme_pub_{k}m = ni affichée {k} mois, ni signalée l'année publiée",
                 (table[f"calme_pub_{k}m"] == publie.astype("int8")).all(), "écart")
        verifier(f"cible_nouvelle_pub_{k}m = cible sur les lignes calmes publiées",
                 (table[f"cible_nouvelle_pub_{k}m"] == table["cible"] * table[f"calme_pub_{k}m"]).all(), "écart")
    verifier("fenêtres d'affichage emboîtées (3 ≤ 6 ≤ 12 mois)", ((table["fr_affichee_3m"] <= table["fr_affichee_6m"])
             & (table["fr_affichee_6m"] <= table["fr_affichee_12m"])).all(), "fenêtre non emboîtée")
    classes = pd.Series(tout).str[:5].value_counts()
    taille = table["atc5"].str[:5].map(classes)
    verifier("taille_classe = substances de la classe ATC4", (table["taille_classe"] == taille).all(), "écart")
    attendu = np.where(taille > 1, table["fr_autres_classe_3m"] / (taille - 1).clip(lower=1), 0)
    verifier("fr_part_classe_3m = autres affichées / (taille - 1), dans [0, 1]",
             np.allclose(table["fr_part_classe_3m"], attendu, atol=1e-3) and table["fr_part_classe_3m"].between(0, 1).all(), "écart")
    verifier("âge de la dernière photo ≥ 0", (table[["fr_age_photo_j", "it_age_photo_j"]] >= 0).all().all(), "âge négatif")
    t0 = pd.Period("2024-06", "M")
    sp = lire("bdpm_specialites_historique")
    sp = sp[sp["commercialise"] & (sp["valid_from"] <= t0.end_time) & (sp["valid_to"].isna() | (sp["valid_to"] > t0.end_time))]
    attendu = sp["cis"].map(correspondance_cis_atc5()).value_counts()
    juin = table[table["mois"] == t0.to_timestamp()].set_index("atc5")
    verifier("mkt_nb_specialites au mois 2024-06 = recalcul sur la BDPM historisée",
             (juin["mkt_nb_specialites"] == attendu.reindex(juin.index).fillna(0).astype(int)).all(), "écart")
    verifier("mkt_nb_titulaires ≤ mkt_nb_specialites et part d'injectables dans [0, 1]",
             (table["mkt_nb_titulaires"] <= table["mkt_nb_specialites"]).all() and table["mkt_part_injectable"].between(0, 1).all(), "incohérent")
    part_connue = table["mkt_connu"].mean()
    verifier(f"substances × mois avec un marché rattaché à l'ATC5 : {part_connue:.1%} (seuil 90 %)", part_connue >= 0.90,
             f"{part_connue:.1%}")

    # Récence : recalcul indépendant sur un échantillon.
    sig_pub = lire("ansm_signalements").dropna(subset=["atc5"])
    echantillon = table.sample(1000, random_state=3)
    attendu = []
    for a, t in zip(echantillon["atc5"], echantillon["mois"]):
        t = t.to_period("M")
        limite = int(derniere_annee_publiee(np.array([t.year]), np.array([t.month]))[0])
        passees = sig_pub.loc[(sig_pub["atc5"] == a) & (sig_pub["annee"] <= limite), "date_signalement"]
        attendu.append(PLAFOND_MOIS if passees.empty else min((t - passees.max().to_period("M")).n, PLAFOND_MOIS))
    verifier("sig_mois_depuis_pub = mois depuis le dernier signalement publié (recalcul)",
             list(echantillon["sig_mois_depuis_pub"]) == attendu, "écart")
    verifier("fr_mois_depuis_affichage = 0 exactement quand la substance est affichée ce mois-ci",
             ((table["fr_mois_depuis_affichage"] == 0) <= (table["fr_affichee_3m"] == 1)).all(), "incohérent")
    verifier("mkt_hhi_titulaires dans [0, 1], égal à 1 avec un seul titulaire, entre 1/n et 1 sinon",
             table["mkt_hhi_titulaires"].between(0, 1.0001).all()
             and np.allclose(table.loc[table["mkt_nb_titulaires"] == 1, "mkt_hhi_titulaires"], 1)
             and (table.loc[table["mkt_nb_titulaires"] > 1, "mkt_hhi_titulaires"]
                  >= 1 / table.loc[table["mkt_nb_titulaires"] > 1, "mkt_nb_titulaires"] - 1e-4).all(), "incohérent")
    for t0 in (pd.Period("2024-06", "M"), pd.Period("2025-02", "M")):
        fin = dernier_mois_publie(t0)
        Bm, _, _ = matrice_conso(lire("consommation_mensuelle"), tout, "base_remboursement")
        abime_b, abime_c = Bm.copy(), C.copy()
        abime_b[:, int((fin - mois_c[0]).n) + 1:] = 1e9
        abime_c[:, int((fin - mois_c[0]).n) + 1:] = 7
        vrai, faux = variables_prix(Bm, C, mois_c, [t0]), variables_prix(abime_b, abime_c, mois_c, [t0])
        for nom in VARIABLES_PRIX:
            verifier(f"{nom} au mois {t0} ne lit rien après {fin}", np.allclose(vrai[nom], faux[nom]), "valeur modifiée")

    # Consommation mensuelle : recalcul indépendant, puis test de non-fuite par perturbation.
    mensuelle = lire("consommation_mensuelle").set_index(["atc5", "mois"])["boites"]
    echantillon = table.sample(1500, random_state=1)
    attendu = []
    for atc5, t in zip(echantillon["atc5"], echantillon["mois"]):
        fin = dernier_mois_publie(t.to_period("M"))
        attendu.append(sum(max(float(mensuelle.get((atc5, (fin - k).to_timestamp()), 0)), 0.0) for k in range(6)))
    verifier("conso_6m_pub = 6 derniers mois publiés (recalcul)", list(echantillon["conso_6m_pub"]) == attendu, "écart")
    for t0 in (pd.Period("2024-06", "M"), pd.Period("2024-10", "M"), pd.Period("2025-02", "M")):
        fin = dernier_mois_publie(t0)
        abime = C.copy()
        abime[:, int((fin - mois_c[0]).n) + 1:] = 1e9  # tout ce qui suit le dernier mois publié devient absurde
        vrai = variables_conso(C, presentes, mois_c, [t0])
        faux = variables_conso(abime, presentes, mois_c, [t0])
        for nom in VARIABLES_CONSO:
            verifier(f"{nom} au mois {t0} ne lit rien après {fin}", np.allclose(vrai[nom], faux[nom], equal_nan=True),
                     "valeur modifiée par des données postérieures")
