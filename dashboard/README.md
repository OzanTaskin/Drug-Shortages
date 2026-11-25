# Dashboard (TME6)

Application web locale (Plotly Dash) pour explorer le dataset : les pénuries publiées en France, en Italie et en
Allemagne, les signalements de rupture à l'ANSM, la consommation de l'Assurance Maladie et le panel `panel_fr`. Le
dashboard montre les données ; il ne contient aucun modèle.

## Lancer

Python 3.11 ou plus récent, depuis la racine du projet :

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python dashboard/app.py
```

Puis ouvrir <http://127.0.0.1:8050>. Le dashboard ne lit que `dashboard/donnees/` (environ 1 Mo, versionné) :
**pas besoin de télécharger ni de reconstruire les 500 Mo de données**.

Pour régénérer ces tables après une mise à jour du dataset (il faut alors `data/`) :

```bash
.venv/bin/python -m nettoyage                    # data/clean et data/final
.venv/bin/python dashboard/construire_donnees.py # dashboard/donnees/
```

## Les six vues

| Onglet | Ce qu'il montre | Pourquoi cette forme |
|---|---|---|
| **Dans le temps** | pénuries en cours ou nouvelles par mois et par pays (effectif ou indice 2025 = 100) ; frise des photos archivées ; saisonnalité des signalements ANSM (année × mois) ; concentration des signalements par laboratoire | séries temporelles ; l'indice compare des agences aux définitions différentes ; les mois dans un trou de plus de 3 mois entre deux photos ne sont pas tracés (Italie 12/2025 → 08/2026) ; axe log pour les 557 laboratoires |
| **Classes thérapeutiques** | arbre ATC (classe → sous-classe → substance) coloré par signalements ou nombre de pays touchés, lié à une carte de chaleur sous-classe ATC2 × mois (clic sur l'arbre = filtre, clic sur une case = liste des substances) | hiérarchie ; 54 sous-classes d'au moins 5 substances, lignes ordonnées par clustering de Ward, couleur log pour une mesure asymétrique |
| **Durées et causes** | durée des pénuries terminées par pays (boîtes ou courbe cumulée, axe log) ; causes publiées (Italie, Allemagne) et gravité (France) | durées très asymétriques ; libellés regroupés par mots-clés |
| **Pays comparés** | part des pénuries simultanées entre deux pays (même substance, même mois) ; corrélations de Spearman entre variables par substance | prépare Q2 ; Spearman car les variables sont asymétriques |
| **Fiche substance** | épisodes dans les 3 pays (Gantt), signalements, consommation, voisines de même ATC4 | épisodes avec début et fin ; les voisines préparent Q1 |
| **Panel et qualité** | la cible mois par mois avec les jeux entraînement / validation / test ; les signaux disponibles chaque mois (gris = non couvert) ; valeurs vides et leur raison ; qualité des sources | une absence de données est grise, jamais un zéro |

## Fichiers

- `app.py` : l'application (mise en page, graphiques, interactions).
- `construire_donnees.py` : lit `data/clean/` et `data/final/`, écrit `donnees/`.
- `donnees/` : tables résumées (épisodes, substances, séries mensuelles, signalements, consommation, co-occurrences,
  résumé du panel) et `meta.json` (couverture, valeurs vides, ordres de clustering).
- `assets/style.css` : la mise en forme (chargée automatiquement par Dash).

## Limites à garder en tête

- Chaque agence a sa propre définition de la pénurie : l'Italie recense chaque présentation, la France seulement les
  médicaments d'intérêt thérapeutique majeur. Comparer les évolutions (indice), pas les niveaux.
- Les trois pays ne sont connus que par des photos archivées : entre deux photos, une pénurie courte peut passer
  inaperçue, et les dates de fin sont approximatives.
- La mise en forme est claire uniquement (pas de thème sombre).
