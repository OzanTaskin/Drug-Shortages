"""Italie (AIFA), Allemagne (BfArM), Norvège (DMP), Suède (Läkemedelsverket), Tchéquie (SÚKL)
→ `europe_observations` et `europe_episodes`.

Ces listes servent de signaux européens (question 2). Chaque photo (archives Wayback + photo du
jour) donne les pénuries publiées à cette date.

Épisode :
- Italie : une période de présence continue d'un médicament (code AIC) sur la liste, comme pour
  la France : une absence d'une seule photo ne coupe pas l'épisode. L'AIFA révise souvent la date
  de début d'une pénurie déjà listée (2 857 médicaments, presque toujours vers plus tard) : la date
  ne peut donc pas servir à identifier l'épisode, sinon chaque révision crée un faux épisode ;
- Allemagne : une déclaration initiale (« Referenzierte Erstmeldung », ou le numéro de la
  déclaration elle-même si c'est la première, ce que le BfArM écrit vide ou « N/A »). Une déclaration couvre souvent plusieurs boîtes (PZN).

Dates :
- `publication` : première photo où l'épisode est publié. C'est la date à laquelle le signal est
  connu : c'est elle, et non le début déclaré, qu'il faut utiliser pour prédire ;
- `debut` : date de début déclarée. En Italie, la dernière déclarée (la plus à jour) ;
  `debut_declare_initial` garde la première. Elle peut être postérieure à la publication
  (pénurie annoncée à l'avance), voire à la fin si l'annonce a été retirée avant cette date ;
- `fin` : dernière photo où l'épisode est publié ; `fin_au_plus_tard` : la photo suivante, où il
  n'est plus publié. La fin réelle est entre les deux. S'il est publié dans la dernière photo, il
  est `en_cours`. Quand `debut` > `fin`, la pénurie annoncée a duré moins que l'intervalle entre
  deux photos, ou l'annonce a été retirée avant la date prévue.
ATC : seuls les codes ATC5 valides sont gardés. En Italie, la colonne ATC n'existe que depuis
fin 2024 : pour les photos plus anciennes, on reprend l'ATC connu du même code AIC, sinon
celui de la même substance.
Les arrêts définitifs de commercialisation (Italie) ne sont pas des pénuries :
`type_episode` = « arret_definitif ».

Norvège : pas de photos successives d'une liste, mais un registre annuel (2019 à 07/2024) qui garde
chaque pénurie déclarée, avec sa date de déclaration (`date_declaration`, Meldingsdato), sa période
(début, fin prévue ou constatée), sa cause et son statut à la date du classeur (31/12, 12/07/2024 pour
le dernier). `date_photo` = cette date de statut. Épisode : un numéro de boîte (Varenummer) et une date
de déclaration ; un même épisode est repris d'un classeur à l'autre tant qu'il est en cours.
- `publication` = date de déclaration : la DMP publie la pénurie à sa déclaration (hypothèse : sans délai) ;
- `fin` = fin de la période déclarée, dans le dernier classeur où l'épisode figure, quand le statut y est
  « Tilgjengelig » (disponible) ; à défaut de date lisible, la date du classeur ;
- `fin_au_plus_tard` (devient `fin_connue_le`) = date du premier classeur où l'épisode est terminé : on
  ne sait qu'il est fini qu'à cette date, au plus tard ;
- en cours au dernier classeur (12/07/2024) : `en_cours`, sans fin (la source s'arrête là) ;
- médicaments vétérinaires écartés (ATC en Q, ou « Legemiddel til dyr ») ; retraits du marché
  (cause « avregistrering ») : `type_episode` = « arret_definitif » ;
- ATC absent du classeur 2019 : repris du même numéro de boîte, sinon de la même substance.

Suède et Tchéquie : des registres complets, pas des photos ; ils ne passent pas par `europe_observations`
(pas d'observations répétées) et donnent directement leurs épisodes.
- Suède : une interruption de vente déclarée = un épisode (MedicineShortageId), dans le fichier « en cours »
  ou « historique » (terminées). `publication` = date de première publication par l'agence ; `debut` = début
  prévu le plus ancien de ses boîtes ; `fin` = fin réelle la plus tardive (vide si une boîte n'a pas de fin
  réelle : en cours) ; `fin_au_plus_tard` (→ `fin_connue_le`) = dernière mise à jour de la fiche, au moins la
  fin réelle (prudent : la fin est connue au plus tard à cette date). Arrêt définitif (CESSATION) :
  `type_episode` = « arret_definitif ». ATC : le plus fréquent parmi les produits de la fiche. Environ 1 000
  fiches ont une fin réelle avant leur début prévu : pénurie annoncée puis évitée ou annulée (gardées, comme
  en Italie).
- Tchéquie : déclarations d'interruption, de reprise et d'arrêt de chaque produit (code SÚKL), depuis 2018.
  Une interruption ouvre un épisode ; la reprise suivante du même produit le ferme (`fin` = date d'effet de la
  reprise, `fin_au_plus_tard` = la plus tardive de sa date d'effet et de sa date de déclaration) ; un arrêt
  le ferme aussi (une interruption déclarée en retard, après sa reprise, n'a sa fin connue qu'à sa
  publication). Une nouvelle interruption alors qu'un épisode est ouvert le prolonge. Sans reprise
  déclarée, l'épisode reste `en_cours` (les reprises ne sont pas toujours déclarées : durée à lire avec
  prudence). Les arrêts seuls sont des épisodes `arret_definitif`. `publication` = date de déclaration (le
  Market report est mis à jour en continu) ; `debut` = date d'effet de l'interruption.
- Slovaquie : même logique que la Tchéquie, sur l'export du portail du ŠÚKL (depuis 04/2019) : `publication` =
  date de dépôt, `debut` = date d'effet. L'export n'a ni ATC, ni substance, ni cause : l'ATC vient de la liste des
  médicaments du ŠÚKL par le code produit (actuelle, puis archives 12/2022 et 12/2019).
"""
import io
import re

import pandas as pd

from .commun import RAW, date_du_fichier, enregistrer

ATC5_VALIDE = re.compile(r"^[A-Z]\d{2}[A-Z]{2}\d{2}$")


def atc5_valide(code):
    code = (code or "").strip().upper() if isinstance(code, str) else ""
    return code if ATC5_VALIDE.match(code) else None


TRONQUE = 1024 * 1024  # la Wayback Machine coupe certaines archives à 1 Mio pile : photo incomplète


def photos(dossier_wayback, dossier_actuel):
    chemins = (sorted((RAW / "wayback" / dossier_wayback).glob("2*.csv"))
               + sorted((RAW / dossier_actuel / "actuel").glob("*.csv")))
    tronques = [c for c in chemins if c.stat().st_size == TRONQUE]
    if tronques:
        print(f"  {dossier_wayback} : {len(tronques)} archive(s) tronquée(s) à 1 Mio écartée(s) : "
              + ", ".join(c.name for c in tronques))
    return [c for c in chemins if c.stat().st_size != TRONQUE]


# En-têtes des listes AIFA de 2019-2020 (majuscules, autres noms) → en-têtes actuels.
ENTETES_AIFA_ANCIENS = {"NOME COMMERCIALE": "Nome medicinale", "AIC": "Codice AIC",
                        "PRINCIPIO ATTIVO": "Principio attivo", "DATA INIZIO": "Data inizio",
                        "DATA DI FINE PRESUNTA": "Fine presunta", "MOTIVAZIONI": "Motivazioni"}


def lire_aifa(chemin):
    # Windows-1252, avec quelques octets invalides (0x9d) dans les lignes d'avertissement : remplacés.
    texte = chemin.read_bytes().decode("cp1252", errors="replace")
    # Deux lignes d'avertissement avant l'en-tête.
    table = pd.read_csv(io.StringIO(texte), sep=";", skiprows=2, dtype=str, keep_default_na=False)
    table.columns = [re.sub(r"\s+", " ", c).strip() for c in table.columns]
    table = table.rename(columns=ENTETES_AIFA_ANCIENS)
    table = table[table["Codice AIC"].str.strip() != ""]
    return pd.DataFrame({
        "pays": "IT",
        "date_photo": pd.Timestamp(date_du_fichier(chemin)),
        "id_episode": table["Codice AIC"].str.strip() + "_" + table["Data inizio"].str.strip(),
        "code_produit": table["Codice AIC"].str.strip(),
        "medicament": table["Nome medicinale"].str.strip(),
        "substance": table["Principio attivo"].str.strip(),
        "atc5": table.get("Codice ATC", pd.Series(None, index=table.index)).map(atc5_valide),
        "debut": pd.to_datetime(table["Data inizio"], format="%d/%m/%Y", errors="coerce"),
        "fin_prevue": pd.to_datetime(table["Fine presunta"], format="%d/%m/%Y", errors="coerce"),
        "raison": table["Motivazioni"].str.strip().replace("", None),
        "criticite": None,
    })


def lire_bfarm(chemin):
    table = pd.read_csv(chemin, sep=";", encoding="latin-1", dtype=str, keep_default_na=False)
    initiale = table["Referenzierte Erstmeldung"].str.strip()
    return pd.DataFrame({
        "pays": "DE",
        "date_photo": pd.Timestamp(date_du_fichier(chemin)),
        # « N/A » (environ un quart des lignes) veut dire « pas de déclaration initiale » : la ligne est
        # elle-même la première déclaration. Sinon, toutes ces lignes formeraient un seul faux épisode.
        "id_episode": initiale.where(~initiale.isin(["", "N/A"]), table["Bearbeitungsnummer"].str.strip()),
        "code_produit": table["PZN"].str.strip(),
        "medicament": table["Arzneimittlbezeichnung"].str.strip(),
        "substance": table["Wirkstoffe"].str.strip(),
        "atc5": table["Atc Code"].map(atc5_valide),
        "debut": pd.to_datetime(table["Beginn"], format="%d.%m.%Y", errors="coerce"),
        "fin_prevue": pd.to_datetime(table["Ende"], format="%d.%m.%Y", errors="coerce"),
        "raison": table["Art des Grundes"].str.strip().replace("", None),
        # Colonne ajoutée en cours de route : absente des archives les plus anciennes.
        "criticite": table["klassifikation"].str.strip().replace("", None) if "klassifikation" in table else None,
    })


COLONNES_DMP = {  # début du nom de colonne → nom dans la table
    "Meldingsdato": "date_declaration", "ATC": "atc5", "Legemiddelnavn": "medicament",
    "Varenummer": "code_produit", "Virkestoff": "substance", "Årsak": "raison",
    "Mangelperiode fra": "debut", "Mangelperiode til": "fin_prevue", "Status pr": "statut",
    "Type legemiddel": "type_legemiddel",
}


def date_dmp(valeurs):
    """Dates du registre : cellules Excel (dates) ou textes « jj.mm.aaaa » ; le reste (« Uke 22 ») est vide."""
    texte = valeurs.astype(str).str.strip().str[:19]
    iso = pd.to_datetime(texte, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    return iso.fillna(pd.to_datetime(texte.str[:10], format="%d.%m.%Y", errors="coerce"))


def lire_dmp(chemin):
    brut = pd.read_excel(chemin, sheet_name=0, header=None, dtype=str)
    entete = next(i for i in range(20) if (brut.iloc[i].astype(str).str.strip() == "Meldingsdato").any())
    noms = [str(c).strip() for c in brut.iloc[entete]]
    table = brut.iloc[entete + 1:].set_axis(noms, axis=1)
    colonnes = {}
    for nom in noms:
        for debut, cible in COLONNES_DMP.items():
            if nom.startswith(debut) and cible not in colonnes.values():
                colonnes[nom] = cible
    statut = next(nom for nom in noms if nom.startswith("Status pr"))
    date_statut = pd.to_datetime(re.search(r"(\d{2}\.\d{2}\.\d{4})", statut).group(1), format="%d.%m.%Y")
    table = table.rename(columns=colonnes)[list(colonnes.values())]
    table = table[table["date_declaration"].notna() & table["code_produit"].notna()]
    texte = lambda c: table[c].fillna("").astype(str).str.strip() if c in table else pd.Series("", index=table.index)
    humain = ~texte("atc5").str.upper().str.startswith("Q") & ~texte("type_legemiddel").str.contains("dyr")
    table = table[humain]
    declaration = date_dmp(table["date_declaration"])
    # Faute de saisie (« 21.11.2109 ») : une déclaration postérieure au classeur est impossible ; on prend
    # alors le début de la pénurie, s'il est lui-même antérieur au classeur.
    debut = date_dmp(table["debut"])
    impossible = declaration > date_statut
    declaration = declaration.where(~impossible, debut.where(debut <= date_statut))
    code = texte("code_produit").str.replace(r"\.0$", "", regex=True).str.zfill(6)
    return pd.DataFrame({
        "pays": "NO",
        "date_photo": date_statut,
        "id_episode": code + "_" + declaration.dt.strftime("%Y%m%d"),
        "code_produit": code,
        "medicament": texte("medicament"),
        "substance": texte("substance").str.lower(),
        "atc5": texte("atc5").map(atc5_valide),
        "debut": debut,
        "fin_prevue": date_dmp(table["fin_prevue"]),
        "raison": texte("raison").replace("", None),
        "criticite": None,
        "date_declaration": declaration,
        "statut": texte("statut").replace("", None),
    }).dropna(subset=["date_declaration"])


def episodes_norvegiens(observations):
    """Un épisode par (numéro de boîte, date de déclaration), à partir des classeurs annuels."""
    no = observations[observations["pays"] == "NO"].sort_values("date_photo")
    derniere = no["date_photo"].max()
    termine = no["statut"].fillna("").str.contains("Tilgjengelig")
    premiere_fin_vue = no[termine].groupby("id_episode")["date_photo"].min()
    episodes = (no.groupby("id_episode")
                .agg(pays=("pays", "first"), medicament=("medicament", "last"), substance=("substance", "last"),
                     atc5=("atc5", lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None),
                     nb_produits=("code_produit", "nunique"),
                     debut=("debut", lambda s: s.dropna().iloc[-1] if s.notna().any() else pd.NaT),
                     debut_declare_initial=("debut", lambda s: s.dropna().iloc[0] if s.notna().any() else pd.NaT),
                     fin_prevue=("fin_prevue", "last"), premiere_photo=("date_photo", "min"),
                     derniere_photo=("date_photo", "max"), raison=("raison", "last"), criticite=("criticite", "last"),
                     publication=("date_declaration", "min"), statut=("statut", "last"))
                .reset_index())
    episodes["fin_au_plus_tard"] = episodes["id_episode"].map(premiere_fin_vue)
    fini = episodes["fin_au_plus_tard"].notna()
    episodes["en_cours"] = ~fini & (episodes["derniere_photo"] == derniere)
    # Disparu du registre sans statut « disponible » : terminé au plus tard au classeur suivant.
    disparu = ~fini & ~episodes["en_cours"]
    dates = sorted(no["date_photo"].unique())
    suivante = {d: s for d, s in zip(dates, dates[1:])}
    episodes.loc[disparu, "fin_au_plus_tard"] = episodes.loc[disparu, "derniere_photo"].map(suivante)
    fin = episodes["fin_prevue"].where(episodes["fin_prevue"] <= episodes["fin_au_plus_tard"])
    episodes["fin"] = fin.fillna(episodes["fin_au_plus_tard"]).where(~episodes["en_cours"])
    arret = episodes["raison"].fillna("").str.lower().str.contains("avregistr")
    episodes["type_episode"] = arret.map({True: "arret_definitif", False: "penurie"})
    return episodes.drop(columns="statut")


NS_SUEDE = "{http://eservices.lakemedelsverket.se/opendata/medicineshortage/v3/}"


def episodes_suedois():
    import xml.etree.ElementTree as ET

    dossier = RAW / "lakemedelsverket"
    jour = max(f.name[:10] for f in dossier.glob("*.xml"))  # photo du dernier jour de collecte
    lignes = []
    for chemin in sorted(dossier.glob(f"{jour}_*.xml")):
        for fiche in ET.parse(chemin).getroot().iter(NS_SUEDE + "MedicineShortage"):
            texte = lambda balise: (fiche.findtext(NS_SUEDE + balise) or "").strip() or None
            dates = lambda balise: [e.text for e in fiche.iter(NS_SUEDE + balise) if e.text]
            produits = fiche.findall(f".//{NS_SUEDE}MedicinalProduct")
            atcs = pd.Series([atc5_valide(p.findtext(NS_SUEDE + "ATC")) for p in produits]).dropna()
            boites = fiche.findall(f".//{NS_SUEDE}PackagedMedicinalProduct")
            fins = dates("ActualEndDate")
            lignes.append({
                "pays": "SE", "id_episode": texte("MedicineShortageId"),
                "medicament": produits[0].findtext(NS_SUEDE + "ProductName") if produits else None,
                "substance": (produits[0].findtext(NS_SUEDE + "ActiveSubstances") or "").lower() if produits else None,
                "atc5": atcs.mode().iloc[0] if len(atcs) else None, "nb_produits": max(len(boites), 1),
                "debut": min(dates("ForecastedStartDate"), default=None),
                "fin_prevue": max(dates("ForecastedEndDate"), default=None),
                "fin": max(fins) if fins and len(fins) >= len(boites) else None,
                "publication": texte("DateOfFirstPublication") or texte("DateOfFirstNotification"),
                "mise_a_jour": texte("LastUpdated"), "raison": texte("CauseOfShortageCategory"),
                "type_episode": "arret_definitif" if texte("TypeOfShortage") == "CESSATION" else "penurie",
            })
    episodes = pd.DataFrame(lignes).drop_duplicates("id_episode", keep="last")
    for colonne in ("debut", "fin_prevue", "fin", "publication", "mise_a_jour"):
        episodes[colonne] = pd.to_datetime(episodes[colonne], errors="coerce")
    episodes["en_cours"] = episodes["fin"].isna()
    episodes["fin_au_plus_tard"] = episodes[["fin", "mise_a_jour"]].max(axis=1).where(~episodes["en_cours"])
    episodes["debut_declare_initial"] = episodes["debut"]
    episodes["premiere_photo"], episodes["derniere_photo"] = episodes["publication"], episodes["mise_a_jour"]
    episodes["criticite"] = None
    return episodes.drop(columns="mise_a_jour")


def episodes_tcheques():
    import json

    dossier = RAW / "sukl"
    jour = max(d.name for d in dossier.iterdir() if d.is_dir())
    declarations = pd.DataFrame([d for chemin in sorted((dossier / jour).glob("*.json"))
                                 for d in json.loads(chemin.read_bytes())["data"]]).drop_duplicates("id")
    declarations = pd.DataFrame({
        "id": declarations["id"], "produit": declarations["kodSUKL"].astype(str),
        "type": declarations["typOmezeni"].map(lambda t: t["def"]),
        "medicament": (declarations["nazevLP"].fillna("") + " " + declarations["doplnekLP"].fillna("")).str.strip(),
        "substance": declarations["atcNazev"].str.lower(), "atc5": declarations["atc"].map(atc5_valide),
        "effet": pd.to_datetime(declarations["platOd"].str[:10], errors="coerce"),
        "declaration": pd.to_datetime(declarations["datumHlaseni"], errors="coerce"),
        "reprise_prevue": pd.to_datetime(declarations["terminObnovy"], format="%d.%m.%Y", errors="coerce"),
        "raison": declarations["duvodPreruseni"].map(lambda r: (r or {}).get("en") or None),
    }).dropna(subset=["effet", "declaration"]).sort_values(["produit", "effet", "declaration", "id"])

    types = {"preruseni": "interruption", "obnoveni": "reprise", "ukonceni": "arret"}
    return episodes_par_declarations(declarations.assign(type=declarations["type"].map(types)), "CZ")


def episodes_par_declarations(declarations, pays):
    """Tchéquie, Slovaquie : déclarations d'un produit (interruption, reprise, arrêt) → épisodes.
    `declarations` : id, produit, type, medicament, substance, atc5, effet, declaration, reprise_prevue, raison."""
    declarations = declarations.sort_values(["produit", "effet", "declaration", "id"])
    episodes, ouverts = [], {}

    def nouveau(d, type_episode):
        return {"pays": pays, "id_episode": f"{d.produit}_{d.id}", "medicament": d.medicament,
                "substance": d.substance, "atc5": d.atc5, "nb_produits": 1, "debut": d.effet,
                "debut_declare_initial": d.effet, "fin_prevue": d.reprise_prevue, "premiere_photo": d.declaration,
                "derniere_photo": d.declaration, "publication": d.declaration, "raison": d.raison, "criticite": None,
                "type_episode": type_episode, "fin": pd.NaT, "fin_au_plus_tard": pd.NaT, "en_cours": True}

    for d in declarations.itertuples():
        if d.type == "interruption":
            if d.produit in ouverts:  # nouvelle interruption pendant un épisode ouvert : il continue
                ouverts[d.produit].update(fin_prevue=d.reprise_prevue, derniere_photo=d.declaration)
            else:
                ouverts[d.produit] = nouveau(d, "penurie")
        elif d.produit in ouverts:  # reprise ou arrêt : l'épisode ouvert se ferme
            e = ouverts.pop(d.produit)
            # Interruption déclarée en retard, après sa reprise : la fin n'est connue qu'à la publication.
            e.update(fin=d.effet, fin_au_plus_tard=max(d.effet, d.declaration, e["publication"]), en_cours=False)
            episodes.append(e)
        elif d.type == "arret":
            e = nouveau(d, "arret_definitif")
            e.update(fin_prevue=pd.NaT, fin=d.effet, fin_au_plus_tard=max(d.effet, d.declaration), en_cours=False)
            episodes.append(e)
    return pd.DataFrame(episodes + list(ouverts.values()))


def atc_slovaques():
    """Code ŠÚKL → ATC5, d'après la liste des médicaments actuelle puis les archives (la plus récente prime)."""
    import io
    import zipfile

    def classeur(archive, chemin=()):
        """Ouvre `archive` (zip), descend dans les zips imbriqués `chemin`, lit le classeur zoznam_liekov.*"""
        z = zipfile.ZipFile(archive)
        for interieur in chemin:
            z = zipfile.ZipFile(io.BytesIO(z.read(interieur)))
        nom = next(n for n in z.namelist() if n.lower().startswith("zoznam_liekov."))
        table = pd.read_excel(io.BytesIO(z.read(nom)), dtype=str)
        return table.set_index(table["ŠÚKL kód"].str.strip())["ATC kód"].map(atc5_valide).dropna()

    dossier = RAW / "sukl_sk"
    actuelle = sorted(dossier.glob("*_zoznam_liekov_aktualny.zip"))[-1]
    sources = [classeur(actuelle), classeur(dossier / "Lieky2022.zip", ["zoznam_liekov_2022_12.zip"]),
               classeur(dossier / "Lieky2019.zip", ["zoznam_liekov_2019_12.zip"])]
    atc = pd.concat(sources)
    return atc[~atc.index.duplicated(keep="first")]


def episodes_slovaques():
    chemin = sorted((RAW / "sukl_sk").glob("*_hlasenia_prerusenia.csv"))[-1]
    brut = pd.read_csv(chemin, sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    types = {"R": "interruption", "O": "reprise", "Z": "arret"}  # U (première mise sur le marché) : ignoré
    brut = brut[brut["Predmet"].isin(types)]
    produit = brut["Kód"].str.strip()
    declarations = pd.DataFrame({
        # Pas d'identifiant de déclaration : l'ordre dans le fichier (le plus récent en premier) en tient lieu.
        "id": range(len(brut), 0, -1), "produit": produit, "type": brut["Predmet"].map(types),
        "medicament": brut["Liek"].str.strip(), "substance": None, "atc5": produit.map(atc_slovaques()),
        "effet": pd.to_datetime(brut["Účinnosť"], format="%Y-%m-%d", errors="coerce"),
        "declaration": pd.to_datetime(brut["Podanie"].str[:10], format="%Y-%m-%d", errors="coerce"),
        "reprise_prevue": pd.NaT, "raison": None,
    }).dropna(subset=["effet", "declaration"])
    return episodes_par_declarations(declarations, "SK")


def episodes_italiens(observations):
    """Identifiant d'épisode italien : AIC + mois de la première photo d'une présence continue.

    Un nouvel épisode commence quand le médicament était absent de plus d'une photo, ou quand il
    passe de pénurie à arrêt définitif (ou l'inverse).
    """
    italie = observations["pays"] == "IT"
    it = observations[italie].copy()
    it["arret"] = it["raison"].fillna("").str.contains("Cessata commercializzazione definitiva")
    rang = {photo: i for i, photo in enumerate(sorted(it["date_photo"].unique()))}
    it["rang"] = it["date_photo"].map(rang)

    presences = (it.groupby(["code_produit", "rang"], as_index=False).agg(arret=("arret", "max"))
                 .sort_values(["code_produit", "rang"]))
    par_produit = presences.groupby("code_produit")
    presences["nouveau"] = (par_produit["rang"].diff().isna() | (par_produit["rang"].diff() > 2)
                            | (par_produit["arret"].shift() != presences["arret"]))
    presences["numero"] = presences.groupby("code_produit")["nouveau"].cumsum()
    premiere = presences.groupby(["code_produit", "numero"])["rang"].transform("min")
    photos = sorted(rang)
    presences["id_episode"] = (presences["code_produit"] + "_"
                               + premiere.map(lambda r: f"{pd.Timestamp(photos[r]):%Y%m}"))

    # Affectation en place : l'ordre des lignes ne change pas.
    ids = presences.set_index(["code_produit", "rang"])["id_episode"]
    observations = observations.copy()
    observations.loc[italie, "id_episode"] = ids.reindex(
        pd.MultiIndex.from_frame(it[["code_produit", "rang"]])).to_numpy()
    return observations


def construire():
    observations = pd.concat(
        [lire_aifa(c) for c in photos("aifa_carenti", "aifa")]
        + [lire_bfarm(c) for c in photos("bfarm_lieferengpaesse", "bfarm")]
        + [lire_dmp(c) for c in sorted((RAW / "dmp").glob("legemiddelmangel_*.xls*"))], ignore_index=True)

    # Italie : ATC repris du même code AIC quand la photo n'a pas encore la colonne ATC.
    atc_par_produit = (observations.dropna(subset=["atc5"]).sort_values("date_photo")
                       .drop_duplicates(["pays", "code_produit"], keep="last")
                       .set_index(["pays", "code_produit"])["atc5"])
    manquant = observations["atc5"].isna()
    cles = pd.MultiIndex.from_frame(observations.loc[manquant, ["pays", "code_produit"]])
    observations.loc[manquant, "atc5"] = atc_par_produit.reindex(cles).to_numpy()
    # Sinon, ATC le plus fréquent pour la même substance dans le même pays.
    atc_par_substance = (observations.dropna(subset=["atc5"])
                         .groupby(["pays", "substance"])["atc5"].agg(lambda s: s.mode().iloc[0]))
    manquant = observations["atc5"].isna()
    cles = pd.MultiIndex.from_frame(observations.loc[manquant, ["pays", "substance"]])
    observations.loc[manquant, "atc5"] = atc_par_substance.reindex(cles).to_numpy()
    observations = episodes_italiens(observations.drop_duplicates())
    enregistrer(observations, "europe_observations")

    norvege = episodes_norvegiens(observations)
    observations = observations[observations["pays"] != "NO"]  # Italie, Allemagne : listes en photos
    derniere_photo = observations.groupby("pays")["date_photo"].max()
    episodes = (observations.sort_values("date_photo").groupby(["pays", "id_episode"])
                .agg(medicament=("medicament", "last"), substance=("substance", "last"),
                     atc5=("atc5", lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None),
                     nb_produits=("code_produit", "nunique"),
                     debut_min=("debut", "min"), debut_declare_initial=("debut", "first"),
                     debut_dernier=("debut", "last"), fin_prevue=("fin_prevue", "last"),
                     premiere_photo=("date_photo", "min"), derniere_photo=("date_photo", "max"),
                     raison=("raison", "last"), criticite=("criticite", "last"))
                .reset_index())
    # Italie : la date la plus à jour ; Allemagne : la plus ancienne déclarée pour la déclaration initiale.
    episodes.insert(episodes.columns.get_loc("debut_min"), "debut",
                    episodes["debut_dernier"].where(episodes["pays"] == "IT", episodes["debut_min"]))
    episodes = episodes.drop(columns=["debut_min", "debut_dernier"])
    episodes["publication"] = episodes["premiere_photo"]
    # Un arrêt définitif de commercialisation n'est pas une pénurie : on le distingue.
    arret = episodes["raison"].fillna("").str.contains("Cessata commercializzazione definitiva")
    episodes["type_episode"] = arret.map({True: "arret_definitif", False: "penurie"})
    episodes["en_cours"] = episodes["derniere_photo"] == episodes["pays"].map(derniere_photo)
    episodes["fin"] = episodes["derniere_photo"].where(~episodes["en_cours"])
    photos_par_pays = observations.groupby("pays")["date_photo"].unique().map(sorted)
    photo_suivante = {(pays, photo): suivante for pays, photos in photos_par_pays.items()
                      for photo, suivante in zip(photos, photos[1:])}
    episodes["fin_au_plus_tard"] = [photo_suivante.get((pays, fin)) if pd.notna(fin) else pd.NaT
                                    for pays, fin in zip(episodes["pays"], episodes["fin"])]
    episodes["fin_au_plus_tard"] = pd.to_datetime(episodes["fin_au_plus_tard"])
    episodes = pd.concat([episodes, norvege[episodes.columns], episodes_suedois()[episodes.columns],
                          episodes_tcheques()[episodes.columns], episodes_slovaques()[episodes.columns]],
                         ignore_index=True)
    enregistrer(episodes, "europe_episodes")
    for pays, groupe in episodes.groupby("pays"):
        print(f"  {pays} : {len(groupe)} épisodes, ATC5 connu pour {groupe['atc5'].notna().mean():.0%}")
