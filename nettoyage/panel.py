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
  la Suède (`se_nouvelle_3m`, Läkemedelsverket depuis 2018) et la Tchéquie (`cz_nouvelle_3m`, SÚKL depuis 2018)
  couvrent toute la période : pas d'indicateur de couverture ;
- l'Allemagne n'est pas dans le panel : 13 mois d'archives seulement (depuis 09/2024).

Cible `cible` : 1 si au moins un signalement de rupture est déclaré à l'ANSM pour la substance
entre t+1 et t+3.

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
REGISTRES = ("NO", "SE", "CZ")  # registres avec date de publication : signal = nouvelle déclaration sur 3 mois
# Couverture des signaux européens : (premier t couvert, dernier t couvert), bornes comprises.
COUVERTURE = {"it": (pd.Period("2021-12", "M"), None), "no": (None, pd.Period("2024-06", "M"))}
DECOUPAGE = [("entrainement", "2020-04", "2023-09"), ("ecart", "2023-10", "2023-12"),
             ("validation", "2024-01", "2024-09"), ("ecart", "2024-10", "2024-12"), ("test", "2025-01", "2025-09")]


VARIABLES_CONSO = ["conso_6m_pub", "conso_tendance_6m", "conso_saison_cible"]
# Lissage des rapports : on ajoute K boîtes aux deux termes. Sans cela, une substance qui vend quelques boîtes
# donne des rapports aberrants (jusqu'à 1,8 million) ; avec, le rapport tend vers 1 quand le volume est faible,
# et il est défini pour toutes les substances vendues en ville.
LISSAGE_SEMESTRE = 100
LISSAGE_MOIS = 10


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


def sur_3_mois(matrice):
    """Vrai à t si vrai à l'un des mois t-2, t-1, t."""
    sortie = matrice.copy()
    sortie[:, 1:] |= matrice[:, :-1]
    sortie[:, 2:] |= matrice[:, :-2]
    return sortie


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
            "it_affichee_3m": pages["IT"], **nouvelles}


def couvert(prefixe, mois):
    """1 si le signal européen `prefixe` est couvert au mois t (bornes de COUVERTURE)."""
    premier, dernier = COUVERTURE[prefixe]
    return np.array([(premier is None or t >= premier) and (dernier is None or t <= dernier) for t in mois])


def matrice_conso(mensuelle, substances):
    """Boîtes remboursées : matrice substances × mois (Medic'AM), et substances présentes dans Medic'AM
    (les autres, souvent réservées à l'hôpital, restent vides : ni zéro, ni imputées)."""
    par_mois = mensuelle.assign(m=mensuelle["mois"].dt.to_period("M")).pivot_table(
        index="atc5", columns="m", values="boites", aggfunc="sum")
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
    table["annee"] = table["mois"].dt.year.astype("int16")
    table["mois_num"] = table["mois"].dt.month.astype("int8")
    table["jeu"] = pd.Series("ecart", index=table.index)
    for jeu, a, b in DECOUPAGE:
        table.loc[(table["mois"] >= pd.Period(a, "M").to_timestamp()) & (table["mois"] <= pd.Period(b, "M").to_timestamp()), "jeu"] = jeu
    table["jeu"] = table["jeu"].astype(pd.CategoricalDtype(["entrainement", "validation", "test", "ecart"]))

    # Pénuries publiées : France couverte sur tout le panel ; Italie et Norvège à 0 hors couverture,
    # avec leur indicateur de couverture.
    for nom, matrice in variables_page(evenements, tout).items():
        type_entier = "int16" if nom == "fr_autres_classe_3m" else "int8"
        valeurs = matrice[lignes][:, DEBUT:T].astype(type_entier)
        if nom[:2] in COUVERTURE:
            masque = couvert(nom[:2], mois)
            valeurs = valeurs * masque
            table[f"{nom[:2]}_couvert"] = np.tile(masque, len(avec)).astype("int8")
        table[nom] = valeurs.ravel()

    # Signalements des années déjà publiées (dernière année publiée, et cumul jusqu'à elle)
    par_annee = (signalements.groupby(["atc5", "annee"]).size().unstack(fill_value=0)
                 .reindex(index=avec, columns=range(2019, 2026), fill_value=0))
    cumul = par_annee.cumsum(axis=1)
    a_pub = derniere_annee_publiee(table["annee"].to_numpy(), table["mois_num"].to_numpy()) - 2019
    base = np.repeat(np.arange(len(avec)), nT)
    table["sig_annee_pub"] = par_annee.to_numpy()[base, a_pub].astype("int32")
    table["sig_cumul_pub"] = cumul.to_numpy()[base, a_pub].astype("int32")

    # Demande mensuelle (Medic'AM, 2015 à 06/2026) : niveau, tendance et saison, sur les semestres déjà publiés
    C, presentes, mois_c = matrice_conso(mensuelle, tout)
    for nom, matrice in variables_conso(C, presentes, mois_c, mois).items():
        valeurs = matrice[lignes].ravel()
        table[nom] = valeurs.astype("int64") if nom == "conso_6m_pub" else valeurs.round(4).astype("float32")

    # Criticité : substance inscrite à la liste critique de l'UE à la fin du mois t
    ue = lire("ue_liste_critique").drop_duplicates("atc5").set_index("atc5")["date_inscription"]
    inscrite = table["atc5"].map(ue)
    table["ue_critique"] = (inscrite <= table["mois"] + pd.offsets.MonthEnd(0)).fillna(False).astype("int8")

    table = table.sort_values(["atc5", "mois"], ignore_index=True)
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
