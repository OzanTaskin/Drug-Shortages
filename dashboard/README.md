# Dashboard d'exploration (livrable 2)

Application web locale (Plotly Dash) pour explorer le dataset du projet DALAS : les pénuries de médicaments
dans 7 pays européens, les signalements de rupture à l'ANSM, la consommation de l'Assurance Maladie et le
panel construit pour prévoir les ruptures en France. Le dashboard **explore** les données ; il ne contient
aucun modèle et ne répond pas encore aux questions de recherche.

## Lancer le dashboard

Il faut Python 3.11 ou plus récent. Depuis la racine du projet :

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python dashboard/app.py
```

Puis ouvrir <http://127.0.0.1:8050>. Le démarrage prend quelques secondes.

Le dashboard ne lit que `dashboard/donnees/` (environ 4 Mo, versionné avec le code) : **pas besoin de
télécharger ni de reconstruire les 870 Mo de données brutes**. La carte de l'Europe charge le fond de carte
de Plotly depuis Internet ; sans connexion, seule cette carte reste vide.

Pour reconstruire les tables résumées après une mise à jour du dataset (il faut alors `dataset/data/`) :

```bash
.venv/bin/python dashboard/construire_donnees.py
```

## Les sept vues

| Onglet | Ce qu'il montre | Pourquoi cette forme |
|---|---|---|
| **Carte de l'Europe** | pénuries en cours par pays (par million d'habitants ou en effectif), au choix de l'année ; un clic sur un pays affiche son profil ; frise de ce que chaque source observe (période, dates des archives) | données géographiques ; la frise rend visibles les trous des archives Wayback |
| **Dans le temps** | pénuries en cours ou nouvelles par mois, par pays (effectif, par million d'habitants, ou indice 2022 = 100) ; saisonnalité des signalements ANSM (année × mois) ; concentration des signalements par laboratoire | séries temporelles ; l'indice compare les dynamiques sans l'effet des définitions nationales ; axe log pour les 557 laboratoires |
| **Classes thérapeutiques** | arbre ATC (classe → sous-classe → substance), lié à une carte de chaleur sous-classe × mois (clic sur l'arbre = filtre, clic sur une case = liste des substances) | hiérarchie ; carte de chaleur lisible : 45 sous-classes d'au moins 5 substances, lignes ordonnées par clustering, couleur log quand la mesure est asymétrique |
| **Durées et causes** | durée des pénuries terminées par pays (boîtes à moustaches ou courbe cumulée, axe log) ; familles de causes par pays | durées très asymétriques (de 1 jour à plusieurs années) ; causes de 5 langues regroupées en 5 familles |
| **Pays comparés** | part des pénuries simultanées entre deux pays (même substance, même mois) ; corrélations de Spearman entre 12 variables par substance | prépare Q2 (pénuries importées) ; Spearman car les variables sont asymétriques |
| **Fiche substance** | pour une substance : ses épisodes dans les 7 pays (diagramme de Gantt), ses signalements ANSM, sa consommation, ses voisines de même classe ATC4 | les données sont des épisodes avec un début et une fin ; les voisines préparent Q1 (substituts) |
| **Panel et qualité** | les deux cibles mois par mois avec le découpage entraînement / validation / test ; les signaux disponibles chaque mois (gris = pays non couvert) ; tableau de qualité des sources | montre la table d'entraînement et ses limites : absences de données en gris, jamais des zéros |

## Fichiers

- `app.py` : l'application (mise en page, graphiques, interactions).
- `construire_donnees.py` : lit `dataset/data/clean/` et `dataset/data/final/`, écrit `donnees/*.parquet`.
- `donnees/` : les tables résumées (épisodes, substances, séries mensuelles, signalements, consommation,
  co-occurrences, résumé du panel) et `meta.json`.
- `assets/style.css` : la mise en forme (chargée automatiquement par Dash).

## Limites à garder en tête

- **Chaque agence a sa propre définition de la pénurie.** L'Italie recense chaque présentation ; la France
  seulement les médicaments d'intérêt thérapeutique majeur. Il faut comparer les évolutions (indice), pas
  les niveaux.
- **La France, l'Italie et l'Allemagne ne sont connues que par des photos archivées** : entre deux photos,
  une pénurie courte peut passer inaperçue. L'Allemagne n'est observée que depuis 07/2024, et la Norvège
  seulement jusqu'en 06/2024.
- **Familles de causes** : le regroupement est fait par mots-clés. La France et la Slovaquie ne publient
  pas de cause, et la Suède n'en donne que pour 40 % des épisodes.
- **Les substances suivies** (885) sont celles de la liste des médicaments d'intérêt thérapeutique majeur
  délivrés en ville. Les vues ATC et le panel ne couvrent que ces substances.
