"""Vérifie que la couche brute est intacte : chaque fichier de data/raw est tracé dans le
manifeste avec la même empreinte SHA-256, et aucun téléchargement interrompu (.part) ne traîne.

Lancée à la fin de `python -m collecte` et au début de `python -m nettoyage` : un fichier brut
modifié à la main ou abîmé est détecté avant de construire les tables.
"""
from .commun import MANIFESTE, RAW, lire_manifeste, sha256

# .DS_Store : fichiers créés par le Finder de macOS à l'ouverture d'un dossier.
IGNORES = {MANIFESTE.name, "_captures.json", ".DS_Store"}
# Fichiers tracés dans le manifeste mais volontairement absents du disque : essais de collecte remplacés ou
# hors périmètre. Toute autre absence arrête la vérification.
OBSOLETES = {
    "dmp/legemiddelmangel_2017.xls", "dmp/legemiddelmangel_2018.xls",  # classeurs avant 2019, hors panel (début 2020)
    "medicam/MEDICAM_2019_annuel.zip", "medicam/MEDICAM_2022_s1.csv",  # premières tentatives, renommées en .xls / .zip
    # Historique suédois du 09/11/2026, effacé du disque le 21/11 ; retéléchargé le 23/11/2026 (le nettoyage lit
    # la photo du dernier jour de collecte, donc les fichiers du 23/11).
    "lakemedelsverket/2026-11-09_opendata-medicine-shortages-historic-3-0.xml",
}


def verifier():
    # En cas de retéléchargement, la dernière ligne d'un fichier fait foi.
    empreintes = {ligne["fichier"]: ligne["sha256"] for ligne in lire_manifeste()}
    problemes, nb_fichiers = [], 0
    for chemin in sorted(RAW.rglob("*")):
        if not chemin.is_file() or chemin.name in IGNORES:
            continue
        nb_fichiers += 1
        relatif = str(chemin.relative_to(RAW))
        if chemin.suffix == ".part":
            problemes.append(f"téléchargement interrompu : {relatif}")
        elif relatif not in empreintes:
            problemes.append(f"absent du manifeste : {relatif}")
        elif sha256(chemin.read_bytes()) != empreintes[relatif]:
            problemes.append(f"empreinte différente du manifeste : {relatif}")
    # Un fichier tracé dans le manifeste mais absent du disque (effacé, déplacé) fausse les tables sans erreur :
    # par exemple, sans l'historique suédois, la Suède passe de 15 886 à 2 422 épisodes.
    for relatif in sorted(empreintes):
        if relatif not in OBSOLETES and not (RAW / relatif).is_file():
            problemes.append(f"tracé dans le manifeste mais absent du disque : {relatif}")
    if problemes:
        raise SystemExit(f"Couche brute non conforme ({len(problemes)} problème(s)) :\n  "
                         + "\n  ".join(problemes[:20]))
    print(f"  couche brute intacte : {nb_fichiers} fichiers, empreintes conformes au manifeste")
