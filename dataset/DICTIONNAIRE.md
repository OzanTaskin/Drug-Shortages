# Dictionnaire des données

Ce que contient chaque table livrée (`data/clean/` et `data/final/`, versionnées), colonne par colonne, et comment
s'en servir. Les tables sont écrites en Parquet ; le manifeste (`manifeste.json` de chaque dossier, copie datée dans
`manifestes/`) donne leur nombre de lignes, leurs sommes et leur empreinte. Les choix et leurs raisons sont dans
[EXPLICATIONS.md](EXPLICATIONS.md), les changements dans [CHANGELOG_DATA.md](CHANGELOG_DATA.md).

Règle générale : **aucune ligne n'est supprimée pour cause de qualité** ; une colonne dit si elle est fiable, et c'est
à celui qui trace une courbe ou entraîne un modèle de filtrer.

## `shortage_events` : un épisode de pénurie ou d'arrêt, dans un pays

Clé : `(pays, id_episode)`. Pays : FR, IT, DE, NO, SE, CZ, SK.

| Colonne | Sens |
|---|---|
| `type_episode` | `penurie` ou `arret_definitif` |
| `atc5`, `atc4` | substance et classe (vide pour 3 à 11 % des épisodes selon le pays) |
| `debut`, `fin`, `en_cours` | dates de l'épisode ; `fin` vide si en cours |
| `publication` | première photo où l'épisode est publié : **la date à laquelle on le sait** (variables de prédiction) |
| `fin_connue_le` | photo où l'on voit que l'épisode est terminé |
| `censure_gauche` / `censure_droite` | début avant la première observation / fin inconnue (changement de site ANSM) |
| `debut_connu` | faux si le début n'est connu qu'approximativement |
| `raison` | cause brute dans la langue de la source |
| `cause` | famille de cause comparable entre pays (voir ci-dessous) |
| `cause_connue` | la source donne une cause (faux : France, Slovaquie, 60 % de la Suède) |
| `anomalie_date` | `aucune`, `fin_avant_debut`, ou `debut_futur` (début après la dernière observation du pays : date prévue) |
| `duree_j` | durée en jours ; vide sauf épisode terminé, au début connu, sans anomalie, non censuré |
| `age_j` | ancienneté d'un épisode encore en cours à la dernière observation |
| `ouvert_longue_duree` | en cours depuis plus de 2 ans : peut-être jamais clos (reprise non déclarée) |
| `doublon` | même pays, substance, dates et type qu'une ligne précédente ; à écarter pour compter des substances |

Familles de `cause` : Production / qualité, Demande en hausse, Distribution / capacité, Arrêt / raison commerciale,
Réglementaire / sécurité, Autre (cause donnée mais non classée, par exemple le « Sonstige » allemand), Non renseignée.

À faire : durées → filtrer `duree_j` non vide ; compter des substances touchées → `~doublon & anomalie_date == "aucune"` ;
courbe de stock → utiliser `couverture_mensuelle`, pas un recomptage.

## `couverture_mensuelle` : ce que chaque source observe, mois par mois

Clé : `(pays, mois)`. Une ligne par mois entre la première observation du pays et son dernier mois complet.

| Colonne | Sens |
|---|---|
| `type_source` | `photos` (page web archivée), `fichiers_annuels` (Norvège) ou `registre` (registre complet lu à la collecte) |
| `nb_photos`, `jours_depuis_photo` | photos du mois ; jours depuis la dernière photo connue (photos seulement) |
| `trou_archive` | dernière photo à plus de 60 jours : l'état du mois est déduit d'une photo ancienne |
| `demarrage_registre` | 12 premiers mois d'un registre : stock sous-estimé |
| `fiable` | ni trou d'archive ni démarrage de registre |
| `stock_fiable` | `fiable` et fins d'épisode connues : faux pour la Norvège (seules ses nouvelles déclarations sont fiables) |
| `stock_episodes`, `stock_substances` | pénuries en cours à la fin du mois (sans anomalie de date) ; substances distinctes |
| `stock_recents` | part du stock commencée depuis moins de 2 ans |
| `nouvelles_substances` | substances dont une pénurie commence dans le mois (hors épisodes déjà publiés à la première observation) |

**Une courbe de stock ne se trace que sur `fiable`**, sinon elle montre des artefacts : l'Italie perd la moitié de
son stock en 2021 et début 2026 faute de photos, la Slovaquie et la Tchéquie démarrent à zéro.

## `ansm_signalements` : un signalement d'un laboratoire à l'ANSM

| Colonne | Sens |
|---|---|
| `date_signalement`, `annee` | date exacte du signalement ; latence de publication de 3 mois à plus d'un an |
| `laboratoire`, `laboratoire_norm` | libellé brut ; libellé sans casse ni forme juridique (« Sandoz SAS » = « SANDOZ ») |
| `atc5`, `cis`, `methode_rapprochement` | substance, médicament et voie de rattachement |
| `ligne_repetee` | identique à une ligne précédente (date, laboratoire, CIP, nom, DCI) : gardée, repérée |

Le fichier officiel ne donne **pas** le type du signalement (rupture, risque de rupture, arrêt) : la cible
« au moins un signalement » mélange ces cas.

## `panel_fr` : substance (ATC5) × mois t

58 410 lignes, aucune valeur vide. Une variable d'une ligne n'utilise que ce qui était **publié à la fin du mois t**.

**Clés et découpage** : `atc5`, `mois`, `annee`, `mois_num`, `semestre` (bloc de 6 mois, validation glissante, voir
`analyse/decoupage.py`), `jeu` (entraînement, validation, test, écart).

**Cibles — jamais des variables d'entrée** :

| Colonne | Définition |
|---|---|
| `cible` | au moins un signalement ANSM entre t+1 et t+3 ; trop facile (la persistance suffit) |
| `calme_6m`, `calme_12m` | aucun signalement de t-5 (t-11) à t : lit les **vraies** dates, **inconnues à t** : sous-population d'évaluation uniquement |
| `cible_nouvelle_6m`, `cible_nouvelle_12m` | `cible` sur les lignes calmes (définition ci-dessus) |
| `calme_pub_6m`, `calme_pub_12m` | **sans fuite** : ni affichée sur la page ANSM depuis k mois, ni signalée l'année publiée la plus récente |
| `cible_nouvelle_pub_6m`, `cible_nouvelle_pub_12m` | `cible` sur les lignes `calme_pub` : **la cible à utiliser pour un modèle utilisable en production** |

**Variables** :

| Famille | Colonnes |
|---|---|
| Page ANSM | `fr_affichee_3m`, `fr_affichee_6m`, `fr_affichee_12m` ; `fr_age_photo_j` (jours depuis la dernière photo) |
| Cascade Q1 | `fr_autres_classe_3m` (compte brut), `fr_part_classe_3m` (part de la classe, à préférer), `taille_classe` |
| Europe Q2 | `it_affichee_3m` + `it_couvert` + `it_age_photo_j` ; `no_nouvelle_3m` + `no_couvert` ; `se_`, `cz_`, `sk_nouvelle_3m` |
| Historique ANSM | `sig_annee_pub` (année publiée la plus récente), `sig_cumul_pub` (cumul jusqu'à elle) |
| Demande | `conso_6m_pub`, `conso_tendance_6m`, `conso_saison_cible` (Medic'AM, semestres publiés) |
| Marché | `mkt_nb_specialites`, `mkt_nb_presentations`, `mkt_nb_titulaires`, `mkt_nb_generiques`, `mkt_part_injectable`, `mkt_hhi_titulaires` (concentration entre titulaires, 1 = un seul), `mkt_connu` |
| Prix | `prix_boite_pub` (base de remboursement par boîte, dernier semestre publié), `prix_tendance_pub` (évolution sur un an) |
| Récence | `fr_mois_depuis_affichage` (99 = jamais affichée depuis 01/2019), `fr_nb_episodes_publies` (épisodes publiés sur la page ANSM depuis 2015), `sig_mois_depuis_pub` (mois depuis le dernier signalement d'une année publiée, 99 = aucun) |
| Criticité | `ue_critique` |

Un signal européen vaut 0 hors couverture : `*_couvert` dit si la source observait le mois (« 0 et couvert » = pas de
pénurie publiée ; « 0 et non couvert » = on ne sait pas). `mkt_connu` = 0 : aucun médicament de la substance n'a pu être
rattaché à son ATC5 à cette date.

**Limites à connaître avant de modéliser** :
- `ue_critique` vaut 0 pendant tout l'entraînement (la liste de l'UE date de 12/2023) : **ne pas l'utiliser** tant que
  l'entraînement ne contient pas de lignes à 1.
- `it_couvert` vaut 0 en 2020 et 8 % en 2021 : l'Italie ne s'apprend que depuis 12/2021 ; la Norvège s'arrête en 06/2024.
- Les dates de publication des signalements et de la consommation sont des hypothèses documentées (data.gouv.fr, data.ansm).
- La cohorte (MITM commercialisés, délivrés en ville) est fixée à l'état de 2026 : les substances retirées depuis manquent.
- Le taux de positifs de `cible` monte de 21 % (2020) à 37 % (2023) : en partie un effet de déclaration, pas de pénurie.
- Environ 120 nouvelles ruptures distinctes dans le test 2025 : une AUC a un intervalle d'environ ± 0,05.
