"""France — page ANSM « Disponibilité des produits de santé » → `ansm_observations`.

Une ligne = une fiche ANSM vue dans une photo de la page (archive Wayback ou photo du jour).
Le passage aux épisodes de pénurie (début, fin) se fait dans `episodes.py`.

Structure de la page, stable de 2021 à 2026 : un tableau, une ligne <tr class="product-item">
par fiche, avec les colonnes
    statut | date | spécialité | remise à disposition | (domaines médicaux, depuis 2025)
Points d'attention :
- la colonne de date s'appelle « Publication » en 2021, puis « Mise à jour » : ce n'est pas
  la date de début de la pénurie ;
- la spécialité s'écrit « NOM dose, forme (substance) » en 2021,
  puis « Nom dose, forme – [substance] » ;
- dans la colonne « remise à disposition », l'attribut data-value est rempli même quand la
  cellule est vide : seule la date affichée compte.

Ancien site (archives de 02/2015 à 03/2021, page « S'informer › Ruptures de stock des médicaments ») :
une liste alphabétique, un bloc <div class="links"> par fiche, avec le statut dans
<span class="date_oos"> et un lien « JJ/MM/AAAA - NOM dose, forme [– statut] ». La date est celle de
la dernière mise à jour de la fiche, comme la colonne « Mise à jour » du nouveau site. Il n'y a ni
colonne de remise à disposition ni domaine médical. La fiche est identifiée par la fin de son adresse,
préfixée de « ancien: » : les fiches de l'ancien site ne sont pas reliées à celles du nouveau.
Colonne `site` : « ancien » ou « actuel ».
"""
import re
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup

from .commun import RAW, date_du_fichier, enregistrer, normaliser_texte, source_du_fichier

STATUTS = {
    "Rupture de stock": "rupture",
    "Tension d'approvisionnement": "tension",
    "Risque de rupture de stock": "risque_rupture",
    "Remise à disposition": "remise_a_disposition",
    "Arrêt de commercialisation": "arret_commercialisation",
}
MOIS = {"janv": 1, "févr": 2, "mars": 3, "avr": 4, "mai": 5, "juin": 6, "juil": 7,
        "août": 8, "sept": 9, "oct": 10, "nov": 11, "déc": 12}


def lire_date(texte):
    """« 22 sept. 2022 », « 05/10/2026 » ou « 2022/10/06 » → date (None si vide)."""
    texte = (texte or "").strip().rstrip(".")
    if not texte:
        return None
    for format_ in ("%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(texte, format_).date()
        except ValueError:
            pass
    # Quelques dates sont saisies avec l'année sur deux chiffres (« 3 oct. 22 »).
    jour, mois, annee = re.match(r"(\d{1,2})\s+([^\s.]+)\.?\s+(\d{4}|\d{2})$", texte).groups()
    annee = int(annee) + (2000 if len(annee) == 2 else 0)
    return datetime(annee, MOIS[mois], int(jour)).date()


def decouper_specialite(texte):
    """Sépare le nom du médicament et la substance active."""
    texte = re.sub(r"\s+", " ", texte).strip()
    crochets = re.match(r"^(.*?)\s*[–-]?\s*\[(.+)\]\s*$", texte)
    if crochets:
        return crochets.group(1).strip(" –-"), crochets.group(2).strip()
    # Format 2021 : substance dans la dernière parenthèse de premier niveau.
    if texte.endswith(")"):
        profondeur = 0
        for i in range(len(texte) - 1, -1, -1):
            profondeur += {")": 1, "(": -1}.get(texte[i], 0)
            if profondeur == 0:
                return texte[:i].strip(), texte[i + 1:-1].strip()
    return texte, None


# Statuts de l'ancien site : libellés plus variés, ramenés aux statuts du nouveau site.
STATUTS_ANCIENS = [
    ("remise", "remise_a_disposition"), ("arret", "arret_commercialisation"),
    ("risque", "risque_rupture"), ("tension", "tension"), ("rupture", "rupture"),
]
MIN_FICHES_PAR_PHOTO = 20
# Liens de la liste qui ne sont pas des fiches de médicament.
HORS_FICHES = ("guide", "bilans-trimestriels", "decret")


def statut_ancien(texte):
    """Statut normalisé, ou None si le texte n'en contient aucun."""
    cle = normaliser_texte(texte)
    for mot, statut in STATUTS_ANCIENS:
        if mot in cle:
            return statut
    return None


def lire_page_ancienne(chemin):
    soupe = BeautifulSoup(chemin.read_text(encoding="utf-8", errors="replace"), "lxml")
    lignes = []
    for bloc in soupe.select("div.links"):
        lien, statut = bloc.find("a", href=True), bloc.find("span", class_="date_oos")
        if lien is None or statut is None or "Ruptures-de-stock-des-medicaments/" not in lien["href"]:
            continue
        adresse = lien["href"].rstrip("/").split("/")[-1]
        if any(mot in adresse.lower() for mot in HORS_FICHES):
            continue
        texte = re.sub(r"\s+", " ", lien.get_text(" ", strip=True))
        date = re.match(r"^(\d{2}/\d{2}/\d{4})\s*-\s*", texte)
        titre = texte[date.end():] if date else texte
        # Le statut est souvent répété à la fin du titre (« … – Rupture de stock ») : retiré du nom.
        titre = re.sub(r"\s*[–-]\s*(rupture|tension|risque|remise|arr[êe]t)[^–-]*$", "", titre, flags=re.IGNORECASE)
        medicament, substance = decouper_specialite(titre.strip(" –-"))
        # Statut parfois absent de l'étiquette (47 cas) : il est alors lu dans le titre ; sinon la fiche
        # n'est pas une pénurie (information générale) et n'est pas gardée.
        statut = statut_ancien(statut.get_text(" ", strip=True)) or statut_ancien(texte)
        if statut is None:
            continue
        lignes.append({
            "date_photo": date_du_fichier(chemin),
            "origine_photo": source_du_fichier(chemin),
            "site": "ancien",
            "fiche": "ancien:" + adresse,
            "statut": statut,
            "date_mise_a_jour": lire_date(date.group(1)) if date else None,
            "medicament": medicament,
            "substance": substance,
            "date_remise_a_disposition": None,
            "domaine_medical": None,
        })
    return lignes


def lire_page(chemin):
    soupe = BeautifulSoup(chemin.read_text(encoding="utf-8", errors="replace"), "lxml")
    lignes = []
    for tr in soupe.select("tr.product-item"):
        cellules = tr.find_all("td")
        statut_brut = cellules[0].get_text(strip=True)
        medicament, substance = decouper_specialite(cellules[2].get_text(" ", strip=True))
        lignes.append({
            "date_photo": date_du_fichier(chemin),
            "origine_photo": source_du_fichier(chemin),
            "site": "actuel",
            "fiche": tr["data-href"].rstrip("/").split("/")[-1],
            "statut": STATUTS.get(statut_brut, statut_brut),
            "date_mise_a_jour": lire_date(cellules[1].get("data-value")),
            "medicament": medicament,
            "substance": substance,
            "date_remise_a_disposition": lire_date(cellules[3].get_text(strip=True)),
            "domaine_medical": cellules[4].get_text(" ", strip=True) if len(cellules) > 4 else None,
        })
    return lignes


def construire():
    anciennes = sorted((RAW / "wayback" / "ansm_ancien").glob("2*.html"))
    pages = sorted((RAW / "wayback" / "ansm_disponibilites").glob("2*.html"))
    pages += sorted((RAW / "ansm" / "actuel").glob("*.html"))
    premiere_actuelle = date_du_fichier(pages[0])
    # Une photo de l'ancien site prise après le passage au nouveau (redirection archivée) est ignorée.
    anciennes = [page for page in anciennes if date_du_fichier(page) < premiere_actuelle]
    lignes, ecartees = [], []
    for page, lecteur in [(page, lire_page_ancienne) for page in anciennes] + [(page, lire_page) for page in pages]:
        lues = lecteur(page)
        # Une capture presque vide est une page d'erreur ou une redirection archivée : on l'écarte
        # (elle ferait croire que toutes les pénuries ont disparu ce mois-là).
        if len(lues) < MIN_FICHES_PAR_PHOTO:
            ecartees.append(f"{page.parent.name}/{page.name} ({len(lues)} fiches)")
            continue
        lignes += lues
    if ecartees:
        print(f"  {len(ecartees)} photo(s) ANSM écartée(s), presque vides : {', '.join(ecartees)}")
    table = pd.DataFrame(lignes)
    for colonne in ("date_photo", "date_mise_a_jour", "date_remise_a_disposition"):
        table[colonne] = pd.to_datetime(table[colonne])
    # Quelques fiches de l'ancien site n'affichent pas de date : celle de la photo la remplace.
    table["date_mise_a_jour"] = table["date_mise_a_jour"].fillna(table["date_photo"])

    inconnus = set(table["statut"]) - set(STATUTS.values())
    if inconnus:
        raise ValueError(f"Statuts ANSM inconnus : {inconnus}")
    enregistrer(table, "ansm_observations")
    return table
