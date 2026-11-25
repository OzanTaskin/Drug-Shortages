"""Dashboard du projet DALAS : les pénuries de médicaments en France, et leurs signaux en Italie et en Allemagne.

Lancement (depuis la racine du projet) :
    .venv/bin/pip install -r requirements.txt
    .venv/bin/python dashboard/app.py          puis ouvrir http://127.0.0.1:8050

Le dashboard ne lit que les petites tables de dashboard/donnees/ (versionnées, environ 1 Mo) ; il n'a pas besoin
des données brutes. Ces tables se reconstruisent avec dashboard/construire_donnees.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

DONNEES = Path(__file__).resolve().parent / "donnees"

# ── Chargement (une fois, au démarrage) ─────────────────────────────────────────────────────
EP = pd.read_parquet(DONNEES / "episodes.parquet")
SUB = pd.read_parquet(DONNEES / "substances.parquet")
SERIE = pd.read_parquet(DONNEES / "serie_pays.parquet")
SIG = pd.read_parquet(DONNEES / "signalements.parquet")
CONSO = pd.read_parquet(DONNEES / "consommation.parquet")
CLASSES = pd.read_parquet(DONNEES / "classes_mois.parquet")
COOC = pd.read_parquet(DONNEES / "cooccurrence.parquet")
PHOTOS = pd.read_parquet(DONNEES / "photos.parquet")
PANEL = pd.read_parquet(DONNEES / "panel_mensuel.parquet")
SIGNAUX = pd.read_parquet(DONNEES / "panel_signaux.parquet")
EVAL = pd.read_parquet(DONNEES / "predictions_evaluation.parquet")
PREVISION = pd.read_parquet(DONNEES / "prediction_courante.parquet")
META = json.loads((DONNEES / "meta.json").read_text())
METRIQUES = META["metriques"]

# ── Référentiels ────────────────────────────────────────────────────────────────────────────
PAYS = ["FR", "IT", "DE"]
NOMS = {"FR": "France", "IT": "Italie", "DE": "Allemagne"}
SOURCES = {"FR": "ANSM, page des disponibilités", "IT": "AIFA, liste des médicaments manquants",
           "DE": "BfArM, Lieferengpässe"}
# Palette catégorielle validée (validate_palette.js, ordre fixe) : couleur = pays, jamais son rang.
COULEUR = {"FR": "#2a78d6", "IT": "#eb6834", "DE": "#1baf7a"}
GRIS_REF = "#898781"
ATC1 = {"A": "Digestif et métabolisme", "B": "Sang", "C": "Cardio-vasculaire", "D": "Dermatologie",
        "G": "Génito-urinaire", "H": "Hormones", "J": "Anti-infectieux", "L": "Cancérologie, immunité",
        "M": "Muscles et squelette", "N": "Système nerveux", "P": "Antiparasitaires", "R": "Respiratoire",
        "S": "Organes sensoriels", "V": "Divers"}
CAUSES = ["Production / qualité", "Demande en hausse", "Distribution", "Arrêt de commercialisation",
          "Réglementaire", "Autre"]
COULEUR_CAUSE = dict(zip(CAUSES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#c3c2b7"]))
JEUX = {"entrainement": ("Entraînement", "rgba(42,120,214,0.07)"), "validation": ("Validation", "rgba(235,104,52,0.09)"),
        "test": ("Test", "rgba(27,175,122,0.10)"), "ecart": ("", "rgba(137,135,129,0.10)")}
BLEUS = [[0, "#f3f7fd"], [0.15, "#cde2fb"], [0.35, "#86b6ef"], [0.6, "#3987e5"], [0.8, "#1c5cab"], [1, "#0d366b"]]
DIVERGENT = [[0, "#1c5cab"], [0.25, "#86b6ef"], [0.5, "#f0efec"], [0.75, "#f09a9a"], [1, "#b42828"]]
SURFACE, GRILLE, AXE, ENCRE, ENCRE2, MUET = "#fcfcfb", "#e1e0d9", "#c3c2b7", "#0b0b0b", "#52514e", "#898781"
STYLE_CELLULE = {"padding": "4px 8px", "textAlign": "left", "border": f"1px solid {GRILLE}", "whiteSpace": "normal"}
STYLE_ENTETE = {"fontWeight": 650, "backgroundColor": "#f3f2ee"}
FIN_VUE = pd.Timestamp(META["dernier_mois"] + "-01") + pd.offsets.MonthEnd(0)


def fmt(n: float) -> str:
    return f"{n:,.0f}".replace(",", " ")


def virgule(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def mise_en_page(fig: go.Figure, hauteur: int = 380, **kw) -> go.Figure:
    base = dict(height=hauteur, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, margin=dict(l=8, r=8, t=10, b=8),
                font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=12.5, color=ENCRE2),
                hoverlabel=dict(bgcolor="white", font_size=12.5, bordercolor=GRILLE), separators=", ",
                legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title_text=""))
    fig.update_layout(**{**base, **kw})
    fig.update_xaxes(gridcolor=GRILLE, linecolor=AXE, zeroline=False, ticks="")
    fig.update_yaxes(gridcolor=GRILLE, linecolor=AXE, zeroline=False, ticks="")
    return fig


def vide(fig: go.Figure, texte: str) -> None:
    fig.add_annotation(text=texte, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font_color=MUET)


def carte(titre: str, sous_titre: str, contenu, lecture: str | None = None) -> html.Div:
    enfants = [html.H3(titre), html.P(sous_titre, className="sous-titre"), contenu]
    if lecture:
        enfants.append(html.Details([html.Summary("Comment lire ce graphique"), html.P(lecture)], className="lecture"))
    return html.Div(enfants, className="carte")


def filtre(titre: str, composant, large: bool = False) -> html.Div:
    return html.Div([html.Label(titre, className="titre"), composant], className="filtre filtre-large" if large else "filtre")


def graphe(id_: str) -> dcc.Graph:
    return dcc.Graph(id=id_, config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def tableau(df: pd.DataFrame, **kw) -> dash_table.DataTable:
    return dash_table.DataTable(data=df.to_dict("records"), columns=[{"name": c, "id": c} for c in df.columns],
                                style_table={"overflowX": "auto"}, style_cell=STYLE_CELLULE, style_header=STYLE_ENTETE, **kw)


def couverture(pays: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(META["couverture"][pays] + "-01"), pd.Timestamp(META["dernier_mois"] + "-01")


# ── Pré-calculs ─────────────────────────────────────────────────────────────────────────────
PEN = EP[EP["type_episode"] == "penurie"]
LIB_ATC2 = SUB.drop_duplicates("atc2").set_index("atc2")["libelle_atc2"].fillna("")
MITM = SUB[SUB["mitm"]].copy()
OPTIONS_SUBSTANCES = [{"label": f"{r.nom} ({r.atc5})", "value": r.atc5} for r in SUB.sort_values("nom").itertuples()]
ANNEE_CONSO = META["annee_conso"]


def tableau_qualite() -> pd.DataFrame:
    lignes = []
    for p in PAYS:
        e = EP[EP["pays"] == p]
        d, f = couverture(p)
        lignes.append({
            "Pays": NOMS[p], "Source": SOURCES[p], "Période observée": f"{d:%m/%Y} – {f:%m/%Y}",
            "Photos archivées": fmt(PHOTOS[PHOTOS["pays"] == p]["date_photo"].nunique()),
            "Épisodes": fmt(len(e)), "Dont arrêts définitifs": fmt((e["type_episode"] == "arret_definitif").sum()),
            "ATC inconnu": f"{e['atc5'].isna().mean():.1%}", "En cours": f"{e['en_cours'].mean():.1%}",
            "Début avant la 1re photo": f"{e['censure_gauche'].mean():.1%}",
            "Cause publiée": "non (gravité seulement)" if p == "FR" else "oui"})
    return pd.DataFrame(lignes)


def spearman_substances() -> tuple[pd.DataFrame, int]:
    v = pd.DataFrame({
        f"Consommation {ANNEE_CONSO} (log)": np.log10(MITM["boites"].fillna(0) + 1),
        "Présentations": MITM["nb_presentations"].fillna(0), "Signalements ANSM": MITM["nb_signalements"],
        "Durée médiane": MITM["duree_mediane_j"], "Liste critique UE": MITM["ue_critique"].astype(int),
        **{f"Épisodes {p}": MITM[f"ep_{p}"] for p in PAYS}, "Risque prédit": MITM["score"]})
    r = v.corr(method="spearman")
    ordre = r.index[leaves_list(linkage(squareform((1 - r.abs()).clip(lower=0).to_numpy(), checks=False), "average"))]
    return r.loc[ordre, ordre], len(v)


QUALITE = tableau_qualite()
SPEARMAN, N_SPEARMAN = spearman_substances()
RETENU = METRIQUES["retenu"]["nom"]

# ── Mise en page ────────────────────────────────────────────────────────────────────────────
TUILES = [(META["episodes"], "épisodes de pénurie ou d'arrêt (FR, IT, DE)"), (len(SIG), "signalements ANSM (2019–2025)"),
          (META["panel_substances"], "substances suivies (MITM)"), (META["panel_lignes"], "lignes du panel substance × mois"),
          (virgule(METRIQUES["test"][RETENU]["pr_auc"]), "PR-AUC du modèle sur le test 2025"),
          (f"{(PREVISION['score'] >= 0.5).sum()}", f"substances à risque > 50 % ({META['mois_prediction']})")]


def vue_temps() -> html.Div:
    return html.Div([
        html.P(["Trois agences publient leurs pénuries, chacune avec sa définition. On ne les connaît que par des ",
                html.B("photos archivées"), " de leur page : avant la première photo, la courbe n'existe pas (ce n'est pas un zéro)."],
               className="intro"),
        html.Div([
            filtre("Pays", dcc.Checklist(id="temps-pays", value=PAYS, inline=True,
                                         options=[{"label": NOMS[p], "value": p} for p in PAYS])),
            filtre("Mesure", dcc.RadioItems(id="temps-mesure", value="en_cours", inline=True, options=[
                {"label": "en cours", "value": "en_cours"}, {"label": "nouvelles par mois", "value": "nouvelles"}])),
            filtre("Échelle", dcc.RadioItems(id="temps-echelle", value="indice", inline=True, options=[
                {"label": "effectif", "value": "effectif"}, {"label": "indice (moyenne 2025 = 100)", "value": "indice"}])),
            filtre("Lissage", dcc.Checklist(id="temps-lissage", value=["oui"],
                                            options=[{"label": "moyenne mobile 3 mois", "value": "oui"}])),
        ], className="filtres"),
        carte("Pénuries dans le temps, par pays", "Une courbe par agence, sur sa seule période d'observation",
              graphe("temps-courbes"),
              "« Indice » ramène chaque pays à sa moyenne de 2025 (= 100), année observée dans les trois pays : on compare "
              "les dynamiques sans l'effet des définitions nationales (l'Italie recense chaque présentation, la France "
              "seulement les médicaments d'intérêt thérapeutique majeur). « Nouvelles » exclut les épisodes déjà ouverts "
              "à la première photo, dont le début est inconnu. Les trous de plus de 3 mois entre deux photos (Italie, 11/2025 → "
              "09/2026) ne sont pas tracés : on n'y voit rien."),
        carte("Ce que chaque source observe, et quand", "Barre = période couverte ; trait = une photo de la page récupérée",
              graphe("couverture"),
              "Entre deux photos on ne voit rien : une pénurie plus courte que l'intervalle peut passer inaperçue. "
              "Les signalements ANSM, eux, sont des classeurs annuels complets (2019-2025)."),
        html.Div([
            carte("Signalements ANSM : saisonnalité", "Déclarations de risque de rupture, par mois et par année", graphe("saison"),
                  "Chaque case compte les signalements envoyés à l'ANSM par les laboratoires. Une ligne plus foncée = une "
                  "année plus chargée. Une partie de la hausse est un effet de déclaration, pas de pénurie."),
            carte("Signalements ANSM : qui déclare ?", "Part cumulée des signalements selon le nombre de laboratoires", graphe("labos"),
                  "Les laboratoires sont triés du plus au moins déclarant (axe log). Une courbe qui monte vite = quelques "
                  "laboratoires font l'essentiel des déclarations."),
        ], className="grille-2"),
    ], className="vue")


def vue_classes() -> html.Div:
    return html.Div([
        html.P([f"Les {len(MITM)} substances suivies sont rangées dans l'arbre ATC (classe → sous-classe → substance). ",
                html.B("Cliquez sur une classe"), " de l'arbre pour filtrer la carte de chaleur, puis ",
                html.B("sur une case"), " de la carte pour lister ses substances."], className="intro"),
        html.Div([
            filtre("Taille des secteurs", dcc.RadioItems(id="arbre-taille", value="nb", inline=True, options=[
                {"label": "nombre de substances", "value": "nb"},
                {"label": f"consommation {ANNEE_CONSO} (boîtes)", "value": "conso"}])),
            filtre("Couleur", dcc.RadioItems(id="arbre-couleur", value="sig", inline=True, options=[
                {"label": "signalements par substance", "value": "sig"},
                {"label": "risque prédit", "value": "score"}])),
        ], className="filtres"),
        html.Div([
            carte("L'arbre thérapeutique", "Classe ATC1 → ATC2 → substance", graphe("arbre"),
                  "La surface d'un secteur est proportionnelle au nombre de substances (ou à la consommation). Sa couleur "
                  "est la moyenne de la mesure choisie : plus c'est foncé, plus la classe est touchée."),
            carte("Quand chaque classe est touchée", "Part des substances de la sous-classe ATC2 touchées ce mois-là",
                  html.Div([
                      html.Div([
                          filtre("Source", dcc.RadioItems(id="chaleur-source", value="signalements", inline=True, options=[
                              {"label": "signalement ANSM", "value": "signalements"},
                              {"label": "pénurie affichée (FR)", "value": "episodes_fr"}])),
                          filtre("Ordre des lignes", dcc.RadioItems(id="chaleur-ordre", value="cluster", inline=True, options=[
                              {"label": "profils similaires", "value": "cluster"}, {"label": "code ATC", "value": "code"}])),
                      ], className="filtres", style={"padding": "6px 0", "border": "none"}),
                      html.Div(id="chaleur-filtre", className="sous-titre"), graphe("chaleur")]),
                  f"Une ligne par sous-classe ATC2 d'au moins {META['min_taille_atc2']} substances suivies (les plus petites "
                  "passeraient de 0 à 100 % en un signalement), une colonne par mois. « Profils similaires » regroupe les "
                  "lignes au même calendrier (clustering hiérarchique de Ward). La page ANSM n'est observée que depuis 04/2021."),
        ], className="grille-2"),
        carte("Substances de la sous-classe sélectionnée", "Cliquez sur une case de la carte de chaleur", html.Div(id="liste-substances")),
    ], className="vue")


def vue_durees() -> html.Div:
    return html.Div([
        html.P(["Les durées sont très asymétriques (quelques jours à plusieurs années) : l'axe est ",
                html.B("logarithmique"), ". Seules les pénuries terminées et commencées après la première photo ont une "
                "durée connue."], className="intro"),
        html.Div([
            filtre("Pays", dcc.Checklist(id="duree-pays", value=PAYS, inline=True,
                                         options=[{"label": NOMS[p], "value": p} for p in PAYS])),
            filtre("Forme", dcc.RadioItems(id="duree-forme", value="boite", inline=True, options=[
                {"label": "boîtes à moustaches", "value": "boite"}, {"label": "courbe cumulée", "value": "ecdf"}])),
            filtre("Début de la pénurie", dcc.RangeSlider(id="duree-annees", min=2021, max=2026, step=1, value=[2021, 2026],
                                                          marks={a: str(a) for a in range(2021, 2027)}), large=True),
        ], className="filtres"),
        html.Div([
            carte("Combien de temps dure une pénurie ?", "Durée en jours des pénuries terminées (échelle log)", graphe("durees"),
                  "La boîte va du 1er au 3e quartile, le trait central est la médiane. En courbe cumulée, lisez la part des "
                  "pénuries terminées avant une durée donnée. Les fins sont datées à la photo près."),
            carte("Pourquoi ? Causes et gravité", "Italie et Allemagne : cause publiée ; France : gravité affichée", graphe("causes"),
                  "Les libellés italiens et allemands sont regroupés par mots-clés. L'Allemagne ne distingue que « problème "
                  "de production » et « autre ». La France ne publie pas de cause, seulement « tension » ou « rupture »."),
        ], className="grille-2"),
    ], className="vue")


def vue_pays() -> html.Div:
    return html.Div([
        html.P(["Q2 se demande si une pénurie en Italie ou en Allemagne annonce une pénurie en France. On regarde d'abord ",
                html.B("si les mêmes substances manquent en même temps"), " dans plusieurs pays."], className="intro"),
        html.Div([
            carte("Pénuries simultanées", "Si une substance manque dans le pays en ligne, part des cas où elle manque aussi "
                  "dans le pays en colonne (même mois)", graphe("cooc"),
                  "Calculé sur les seuls mois observés dans les deux pays. Lecture : « quand la France manque d'une substance, "
                  "l'Italie en manque aussi dans x % des cas ». La matrice n'est pas symétrique : l'Italie recense beaucoup "
                  "plus de pénuries, donc elle « couvre » souvent les autres. Simultané ne veut pas dire « annonce » : "
                  "l'onglet Modèle mesure ce que l'Italie apporte vraiment à la prévision."),
            carte("Ce qui va ensemble, substance par substance",
                  f"Corrélation de Spearman entre {len(SPEARMAN)} variables, sur les {N_SPEARMAN} substances suivies",
                  graphe("spearman"),
                  "Spearman (sur les rangs) plutôt que Pearson, car les variables sont très asymétriques. Rouge = les deux "
                  "montent ensemble, bleu = l'une monte quand l'autre baisse. Variables rangées par similarité. Une "
                  "corrélation n'est pas une causalité : la consommation et le nombre de présentations jouent sur presque tout."),
        ], className="grille-2"),
    ], className="vue")


def vue_substance() -> html.Div:
    return html.Div([
        html.P("Toutes les sources réunies pour une substance : ses pénuries dans chaque pays, ses signalements à l'ANSM, "
               "sa consommation en France, son risque prédit et ses voisines de même classe.", className="intro"),
        html.Div([filtre("Substance (nom ou code ATC)", dcc.Dropdown(id="substance", options=OPTIONS_SUBSTANCES,
                                                                        value="J01CA04", clearable=False), large=True)],
                 className="filtres"),
        html.Div(id="substance-badges", style={"marginBottom": "10px"}),
        carte("Ses pénuries, pays par pays", "Une barre par épisode, du début à la fin (en cours : jusqu'à la dernière photo)",
              graphe("gantt"),
              "Une ligne par pays. Une barre = un épisode pour un produit ou une présentation contenant la substance ; "
              "plusieurs barres superposées = plusieurs produits en pénurie en même temps. Barres pâles : arrêts définitifs."),
        html.Div([
            carte("Signalements ANSM", "Déclarations des laboratoires, par trimestre", graphe("sub-signalements")),
            carte("Risque prédit et réalité", "Score du modèle chaque mois (validation et test) ; points = signalement dans les 3 mois",
                  graphe("sub-score"),
                  "Le score est la probabilité, estimée à la fin du mois, d'au moins un signalement dans les 3 mois suivants. "
                  "Un point en haut = la substance a bien été signalée ensuite."),
        ], className="grille-2"),
        html.Div([
            carte("Consommation en France", "Boîtes remboursées par l'Assurance Maladie, par an", graphe("sub-conso")),
            carte("Substances voisines (même classe ATC4)", "Les substituts possibles, au cœur de la question Q1",
                  html.Div(id="voisines")),
        ], className="grille-2"),
    ], className="vue")


def vue_panel() -> html.Div:
    m = META["manquants_panel"]
    noms = {"fr_affichee_3m": "Affichée sur la page ANSM (3 mois)", "fr_autres_classe_3m": "Autres de la classe ATC4 affichées (Q1)",
            "it_affichee_3m": "En pénurie en Italie (Q2)", "de_affichee_3m": "En pénurie en Allemagne (Q2)",
            "sig_an_prec": "Signalements de l'année précédente", "sig_cumul_avant": "Signalements des années précédentes",
            "conso_boites_an_prec": "Boîtes remboursées l'année précédente",
            "conso_presentations_an_prec": "Présentations remboursées l'année précédente", "ue_critique": "Liste critique de l'UE"}
    manquants = pd.DataFrame({"Variable": [noms[k] for k in m], "Colonne": list(m), "Vide": [f"{v:.0%}" for v in m.values()],
                              "Pourquoi": ["3 mois de page ANSM observés à partir de 06/2021"] * 2 + [
                                          "3 mois de page italienne observés à partir de 12/2021",
                                          "3 mois de page allemande observés à partir de 09/2024", "2019 : pas d'année précédente",
                                          "2019 : pas d'année précédente", "2019 et substances hors Open Medic (hôpital)",
                                          "2019 et substances hors Open Medic (hôpital)", "—"]})
    return html.Div([
        html.P(["La table d'entraînement : ", html.B(f"{META['panel_substances']} substances × "
                f"{len(PANEL)} mois"), f" ({META['panel_mois'][0]} → {META['panel_mois'][1]}). Chaque variable n'utilise que ce "
                "qui était publié à la fin du mois ; le découpage entraînement / validation / test est chronologique."],
               className="intro"),
        carte("La cible au fil des mois", "Part des substances signalées à l'ANSM dans les 3 mois suivants ; fonds = jeux",
              graphe("panel-cible"),
              "Les trois mois d'écart (gris) entre les jeux évitent qu'une cible de l'entraînement, qui regarde trois mois en "
              "avant, chevauche la validation. La hausse de 2019 à 2023 vient en partie de l'obligation de déclarer."),
        html.Div([filtre("Échelle des couleurs", dcc.RadioItems(id="signaux-echelle", value="ligne", inline=True, options=[
            {"label": "normalisée par variable", "value": "ligne"}, {"label": "part brute", "value": "brute"}]))],
            className="filtres"),
        carte("Les signaux disponibles chaque mois", "Part des substances pour lesquelles le signal vaut 1 ; gris = source non couverte",
              graphe("panel-signaux"),
              "En « normalisée par variable », chaque ligne va de son minimum (clair) à son maximum (foncé) : on voit quand un "
              "signal s'allume, même rare. Les cases grises sont des absences de données, jamais des zéros. L'Allemagne "
              "n'apparaît qu'en validation et test : elle ne peut pas être apprise."),
        html.Div([
            carte("Valeurs vides du panel", "Part des lignes sans valeur, et pourquoi", tableau(manquants)),
            carte("Qualité et couverture des sources", "Ce qui est fiable, et ce qui ne l'est pas", tableau(QUALITE)),
        ], className="grille-2"),
    ], className="vue")


def vue_modele() -> html.Div:
    return html.Div([
        html.P(["Peut-on anticiper les pénuries ? Le modèle retenu (", html.B("régression logistique"), ", choisi sur la "
                "validation) est comparé à des règles simples sur le test 2025, lu une seule fois. Puis Q1 et Q2 : le même "
                "modèle avec et sans la variable testée."], className="intro"),
        html.Div([
            carte("Le modèle face aux références", "Test 2025 : PR-AUC (précision moyenne) ; ligne pointillée = hasard",
                  graphe("modele-metriques"),
                  "La PR-AUC résume la précision à tous les seuils ; un classement au hasard vaut le taux de positifs (0,28). "
                  "La règle « signalée l'année précédente » fait presque aussi bien que le modèle : l'historique de la "
                  "substance porte l'essentiel du signal."),
            carte("Q1 et Q2 : ce que la variable ajoute", "Écart de PR-AUC sur le test (avec − sans), intervalle à 95 % par bootstrap sur les substances",
                  graphe("modele-questions"),
                  "Un intervalle qui touche zéro = pas d'apport démontré. La cascade (Q1) n'apporte rien ; l'Italie (Q2) est à "
                  "la limite en logistique et nulle en boosting. L'Allemagne n'est pas testable (13 mois, absents de l'entraînement)."),
        ], className="grille-2"),
        carte("Mois par mois sur le test : les 100 substances jugées les plus à risque", "Part réellement signalée dans les 3 mois",
              graphe("modele-topk"),
              "Chaque mois, on prend les 100 substances au score le plus élevé et on regarde combien ont été signalées. "
              "C'est l'usage visé : une liste courte à surveiller."),
        html.Div([
            filtre("Classe ATC", dcc.Dropdown(id="prev-classe", value="tout", clearable=False, options=[
                {"label": "toutes", "value": "tout"}, *[{"label": f"{k} · {v}", "value": k} for k, v in ATC1.items()]])),
            filtre("Pénurie affichée en France", dcc.RadioItems(id="prev-affichee", value="tout", inline=True, options=[
                {"label": "toutes", "value": "tout"}, {"label": "pas encore affichées", "value": "non"}])),
        ], className="filtres"),
        carte(f"Prévision : risque pour {pd.Period(META['mois_prediction'], 'M') + 1} → {pd.Period(META['mois_prediction'], 'M') + 3}",
              f"Données connues à la fin de {META['mois_prediction']} ; les substances au risque le plus élevé en premier",
              html.Div(id="prev-tableau"),
              "Le score est calculé par analyse/prediction.py avec les mêmes variables que le panel (signalements jusqu'en "
              "2025, consommation 2025). « Pas encore affichées » garde les substances absentes de la page ANSM ces 3 "
              "derniers mois : celles qu'une alerte apprendrait vraiment."),
    ], className="vue")


app = Dash(__name__, title="Pénuries de médicaments — DALAS", suppress_callback_exceptions=True)
ONGLETS = [("temps", "Dans le temps", vue_temps), ("classes", "Classes thérapeutiques", vue_classes),
           ("durees", "Durées et causes", vue_durees), ("pays", "Pays comparés", vue_pays),
           ("substance", "Fiche substance", vue_substance), ("panel", "Panel et qualité", vue_panel),
           ("modele", "Modèle et prévisions", vue_modele)]
app.layout = html.Div([
    html.Header([html.H1("Pénuries de médicaments en France : prévisibles, contagieuses, importées ?"),
                 html.P("Explorer le dataset du projet DALAS : les pénuries publiées par l'ANSM, l'AIFA et le BfArM, les "
                        "signalements de rupture à l'ANSM, la consommation de l'Assurance Maladie, le panel de modélisation "
                        "et ce que le modèle en tire.")]),
    html.Div([html.Div([html.Div(v if isinstance(v, str) else fmt(v), className="valeur"), html.Div(l, className="libelle")],
                       className="tuile") for v, l in TUILES], className="tuiles"),
    dcc.Tabs(id="onglets", value="temps", className="onglets", children=[
        dcc.Tab(label=lib, value=v, className="onglet", selected_className="onglet--actif", children=f())
        for v, lib, f in ONGLETS]),
    html.Footer("Ozan Taskin et Rahim Belkacemi — Sorbonne Université, cours DALAS 2026-2027. Sources : ANSM (page des "
                "disponibilités, signalements), AIFA, BfArM, Assurance Maladie (Open Medic), BDPM, liste critique de l'UE."),
], className="page")


# ── Vue 1 : dans le temps ───────────────────────────────────────────────────────────────────
def mois_sans_photo(pays: str) -> pd.DatetimeIndex:
    """Mois entièrement compris dans un trou de plus de 3 mois entre deux photos : rien n'y est observé, et les
    dates de fin des épisodes y sont inventées par interpolation. On ne trace pas ces mois."""
    dates = PHOTOS[PHOTOS["pays"] == pays]["date_photo"].drop_duplicates().sort_values()
    trous = [pd.period_range(pd.Period(a, "M") + 1, pd.Period(b, "M") - 1, freq="M").to_timestamp()
             for a, b in zip(dates[:-1], dates[1:]) if (b - a).days > 92]
    return trous[0].append(trous[1:]) if trous else pd.DatetimeIndex([])


@app.callback(Output("temps-courbes", "figure"), Input("temps-pays", "value"), Input("temps-mesure", "value"),
              Input("temps-echelle", "value"), Input("temps-lissage", "value"))
def fig_temps(pays: list, mesure: str, echelle: str, lissage: list) -> go.Figure:
    fig = go.Figure()
    fins = []
    for p in [p for p in PAYS if p in pays]:
        d, f = couverture(p)
        s = SERIE[SERIE["pays"] == p].set_index("mois")[mesure].sort_index().reindex(pd.date_range(d, f, freq="MS"), fill_value=0)
        s = s.astype("float64")
        s[s.index.isin(mois_sans_photo(p))] = np.nan
        if "oui" in (lissage or []):
            s = s.rolling(3, min_periods=1).mean().where(s.notna())
        if echelle == "indice":
            s = s / s[s.index.year == 2025].mean() * 100
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=NOMS[p], mode="lines", line=dict(color=COULEUR[p], width=2),
                                 hovertemplate=f"{NOMS[p]}<br>%{{x|%m/%Y}} : %{{y:.1f}}<extra></extra>"))
        isoles = s[s.notna() & s.shift(1).isna() & s.shift(-1).isna()]  # mois observé entre deux trous : un point
        fig.add_trace(go.Scatter(x=isoles.index, y=isoles.values, mode="markers", showlegend=False,
                                 marker=dict(size=8, color=COULEUR[p], line=dict(color=SURFACE, width=2)),
                                 hovertemplate=f"{NOMS[p]}<br>%{{x|%m/%Y}} : %{{y:.1f}}<extra></extra>"))
        fins.append((s.dropna().iat[-1], s.dropna().index[-1], NOMS[p]))
    # Étiquettes de fin de courbe, écartées verticalement quand elles se chevauchent
    haut = max([v for v, _, _ in fins] + [1])
    decalage, precedent = 0, None
    for v, x, nom in sorted(fins):
        decalage = decalage + 13 if precedent is not None and v - precedent < 0.05 * haut else 0
        precedent = v
        fig.add_annotation(x=x, y=v, text=nom, showarrow=False, xanchor="left", xshift=4, yshift=decalage,
                           font=dict(size=11, color=ENCRE2))
    titre = {"effectif": "nombre d'épisodes", "indice": "indice, 2025 = 100"}[echelle]
    return mise_en_page(fig, 420, yaxis=dict(title=titre, rangemode="tozero"), hovermode="x unified",
                        xaxis_range=["2021-01-01", "2027-03-31"])


@app.callback(Output("couverture", "figure"), Input("onglets", "value"))
def fig_couverture(_) -> go.Figure:
    fig = go.Figure()
    for p in PAYS[::-1]:
        d, f = couverture(p)
        fig.add_trace(go.Scatter(x=[d, f + pd.offsets.MonthEnd(0)], y=[NOMS[p]] * 2, mode="lines", opacity=0.35,
                                 line=dict(color=COULEUR[p], width=12), showlegend=False,
                                 hovertemplate=f"{NOMS[p]} : {d:%m/%Y} – {f:%m/%Y}<extra></extra>"))
        dates = PHOTOS[PHOTOS["pays"] == p]["date_photo"].drop_duplicates()
        fig.add_trace(go.Scatter(x=dates, y=[NOMS[p]] * len(dates), mode="markers", showlegend=False,
                                 marker=dict(symbol="line-ns-open", size=14, color=COULEUR[p], line_width=2),
                                 hovertemplate=f"{NOMS[p]} : photo du %{{x|%d/%m/%Y}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=["2019-01-01", "2025-12-31"], y=["Signalements ANSM"] * 2, mode="lines", opacity=0.35,
                             line=dict(color=COULEUR["FR"], width=12), showlegend=False,
                             hovertemplate="Signalements ANSM : classeurs annuels 2019-2025<extra></extra>"))
    return mise_en_page(fig, 230, xaxis_range=["2018-10-01", "2026-12-31"])


@app.callback(Output("saison", "figure"), Input("onglets", "value"))
def fig_saison(_) -> go.Figure:
    t = SIG.assign(mois=SIG["date_signalement"].dt.month).groupby(["annee", "mois"]).size().unstack(fill_value=0)
    noms_mois = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]
    fig = go.Figure(go.Heatmap(z=t.values, x=noms_mois, y=[str(a) for a in t.index], colorscale=BLEUS, xgap=2, ygap=2,
                               colorbar=dict(thickness=10, title=dict(text="signalements", side="right")),
                               hovertemplate="%{x} %{y} : %{z} signalements<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return mise_en_page(fig, 330)


@app.callback(Output("labos", "figure"), Input("onglets", "value"))
def fig_labos(_) -> go.Figure:
    n = SIG["laboratoire"].value_counts()
    part = n.cumsum() / n.sum() * 100
    fig = go.Figure(go.Scatter(x=np.arange(1, len(n) + 1), y=part.values, mode="lines", line=dict(color=COULEUR["FR"], width=2),
                               customdata=n.index, hovertemplate="les %{x} premiers labos : %{y:.0f} %<br>dernier ajouté : "
                                                                 "%{customdata}<extra></extra>"))
    for k in (10, 50):
        fig.add_trace(go.Scatter(x=[k], y=[part.iloc[k - 1]], mode="markers+text", marker=dict(size=8, color=COULEUR["FR"]),
                                 text=[f"{k} labos : {part.iloc[k - 1]:.0f} %"], textposition="middle right",
                                 textfont_color=ENCRE2, showlegend=False, hoverinfo="skip"))
    return mise_en_page(fig, 330, showlegend=False, xaxis=dict(type="log", title=f"nombre de laboratoires (sur {len(n)}, log)"),
                        yaxis=dict(title="part des signalements (%)", range=[0, 102]))


# ── Vue 2 : classes ATC ─────────────────────────────────────────────────────────────────────
@app.callback(Output("arbre", "figure"), Input("arbre-taille", "value"), Input("arbre-couleur", "value"))
def fig_arbre(taille: str, couleur: str) -> go.Figure:
    m = MITM.assign(l1=MITM["atc1"] + " · " + MITM["atc1"].map(ATC1),
                    l2=MITM["atc2"] + " · " + MITM["libelle_atc2"].fillna("").str.slice(0, 32),
                    poids=1.0 if taille == "nb" else MITM["boites"].fillna(0).clip(lower=1),
                    mesure=MITM["nb_signalements"] if couleur == "sig" else MITM["score"].fillna(0) * 100)
    m["pm"] = m["poids"] * m["mesure"]
    m["id1"], m["id2"] = m["l1"], m["l1"] + "/" + m["l2"]
    m["id3"] = m["id2"] + "/" + m["atc5"]
    niveaux = []
    for id_, parent, label in [("id1", None, "l1"), ("id2", "id1", "l2"), ("id3", "id2", "nom")]:
        g = m.groupby(id_).agg(poids=("poids", "sum"), pm=("pm", "sum"), label=(label, "first"),
                               parent=(parent, "first") if parent else (id_, lambda s: ""))
        niveaux.append(g.reset_index().rename(columns={id_: "id"}))
    n = pd.concat(niveaux, ignore_index=True)
    mesures = n["pm"] / n["poids"]
    if couleur == "sig":  # très asymétrique : couleur en log
        unite, couleurs = "signalements par substance (log)", np.log10(mesures + 1)
        barre = dict(tickvals=np.log10(np.array([0, 3, 10, 30, 100, 300]) + 1), ticktext=["0", "3", "10", "30", "100", "300"])
    else:
        unite, couleurs, barre = "risque prédit moyen (%)", mesures, {}
    fig = go.Figure(go.Sunburst(ids=n["id"], labels=n["label"], parents=n["parent"], values=n["poids"], branchvalues="total",
                                customdata=mesures, maxdepth=2, insidetextorientation="radial",
                                marker=dict(colors=couleurs, colorscale=BLEUS, line=dict(color="white", width=1),
                                            colorbar=dict(thickness=10, title=dict(text=unite, side="right"), **barre)),
                                hovertemplate="<b>%{label}</b><br>" + unite.replace(" (log)", "") + " : %{customdata:.1f}<extra></extra>"))
    return mise_en_page(fig, 560)


@app.callback(Output("chaleur", "figure"), Output("chaleur-filtre", "children"), Input("chaleur-source", "value"),
              Input("chaleur-ordre", "value"), Input("arbre", "clickData"))
def fig_chaleur(source: str, ordre: str, click) -> tuple[go.Figure, str]:
    prefixe = ""
    if click:
        point = click["points"][0]
        # Un clic sur le centre (entry == id) remonte d'un niveau : on filtre alors sur le parent.
        cible = point.get("parent", "") if point.get("entry") == point["id"] else point["id"]
        segments = cible.split("/") if cible else []
        prefixe = segments[min(len(segments), 2) - 1].split(" ")[0] if segments else ""
    t = CLASSES[CLASSES["source"] == source]
    lignes = META["ordres_atc2"][source]
    lignes = [l for l in (sorted(lignes) if ordre == "code" else lignes) if l.startswith(prefixe)]
    if not lignes:
        fig = go.Figure()
        vide(fig, f"Aucune sous-classe de {prefixe} ne compte {META['min_taille_atc2']} substances suivies ou plus")
        return mise_en_page(fig, 200, xaxis_visible=False, yaxis_visible=False), f"Filtré sur {prefixe}"
    large = t.pivot(index="atc2", columns="mois", values="part").reindex(index=lignes).fillna(0)
    if source == "episodes_fr":
        large = large.loc[:, large.columns >= pd.Timestamp(META["couverture"]["FR"] + "-01")]
    taille = MITM.groupby("atc2").size()
    noms = [f"{c} · {LIB_ATC2.get(c, '')[:30]} ({taille.get(c, 0)})" for c in large.index]
    fig = go.Figure(go.Heatmap(z=large.values * 100, x=large.columns, y=noms, colorscale=BLEUS, zmin=0, ygap=1,
                               zmax=max(5, np.nanpercentile(large.values * 100, 99)),
                               customdata=np.repeat(np.array(list(large.index))[:, None], large.shape[1], axis=1),
                               colorbar=dict(thickness=10, title=dict(text="% touchées", side="right")),
                               hovertemplate="%{y}<br>%{x|%m/%Y} : %{z:.0f} % des substances<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickfont_size=10.5, dtick=1)
    texte = (f"Filtré sur la classe {prefixe} (cliquez au centre de l'arbre pour tout réafficher)" if prefixe
             else f"Les {len(lignes)} sous-classes ATC2 d'au moins {META['min_taille_atc2']} substances suivies (nombre entre parenthèses)")
    return mise_en_page(fig, max(320, 26 + 13 * len(lignes))), texte


@app.callback(Output("liste-substances", "children"), Input("chaleur", "clickData"))
def liste_substances(click):
    if not click:
        return html.P("Aucune case sélectionnée.", className="sous-titre")
    atc2 = click["points"][0]["customdata"]
    m = MITM[MITM["atc2"] == atc2].sort_values("nb_signalements", ascending=False)
    df = pd.DataFrame({"Substance": m["nom"], "ATC": m["atc5"], "Signalements ANSM": m["nb_signalements"],
                       **{f"Épisodes {p}": m[f"ep_{p}"] for p in PAYS},
                       "Risque prédit": (m["score"] * 100).round(0).astype("Int64").astype(str) + " %",
                       "Liste critique UE": np.where(m["ue_critique"], "oui", "")})
    return [html.P(f"{atc2} · {LIB_ATC2.get(atc2, '')} — {len(df)} substances", className="sous-titre"),
            tableau(df, sort_action="native", page_size=12)]


# ── Vue 3 : durées et causes ────────────────────────────────────────────────────────────────
@app.callback(Output("durees", "figure"), Input("duree-pays", "value"), Input("duree-forme", "value"),
              Input("duree-annees", "value"))
def fig_durees(pays: list, forme: str, annees: list) -> go.Figure:
    d = PEN[(~PEN["censure_gauche"]) & PEN["duree_j"].notna() & PEN["debut"].dt.year.between(*annees)]
    d = d.assign(duree=d["duree_j"].clip(lower=1))
    fig = go.Figure()
    for p in [p for p in PAYS if p in pays]:
        x = d.loc[d["pays"] == p, "duree"]
        if x.empty:
            continue
        nom = f"{NOMS[p]} (n = {fmt(len(x))})"
        if forme == "boite":
            fig.add_trace(go.Box(y=x, name=nom, marker_color=COULEUR[p], boxpoints=False, line_width=1.5,
                                 hovertemplate=f"{NOMS[p]}<br>%{{y:.0f}} jours<extra></extra>"))
            fig.add_annotation(x=nom, y=np.log10(x.median()), text=f"{x.median():.0f} j", showarrow=False, yshift=9,
                               font=dict(size=11, color=ENCRE))
        else:
            xs = np.sort(x.to_numpy())
            fig.add_trace(go.Scatter(x=xs, y=np.arange(1, len(xs) + 1) / len(xs) * 100, name=NOMS[p], mode="lines",
                                     line=dict(color=COULEUR[p], width=2),
                                     hovertemplate=f"{NOMS[p]} : %{{y:.0f}} % durent moins de %{{x:.0f}} j<extra></extra>"))
    reperes = dict(tickvals=[1, 7, 30, 90, 365, 1095], ticktext=["1 j", "1 sem.", "1 mois", "3 mois", "1 an", "3 ans"])
    if not fig.data:
        vide(fig, "Aucune pénurie terminée pour ce choix")
    if forme == "boite":
        return mise_en_page(fig, 440, showlegend=False, yaxis=dict(type="log", title="durée (échelle log)", **reperes))
    return mise_en_page(fig, 440, xaxis=dict(type="log", title="durée (échelle log)", **reperes),
                        yaxis=dict(title="part des pénuries terminées (%)", range=[0, 101]))


@app.callback(Output("causes", "figure"), Input("duree-pays", "value"), Input("duree-annees", "value"))
def fig_causes(pays: list, annees: list) -> go.Figure:
    d = EP[EP["debut"].dt.year.between(*annees) & EP["pays"].isin(pays)]
    fig = go.Figure()
    eu = d[d["pays"] != "FR"]
    t = pd.crosstab(eu["pays"], eu["cause"], normalize="index").reindex(columns=CAUSES, fill_value=0) * 100
    lignes = [NOMS[p] for p in t.index]
    for c in CAUSES:
        fig.add_trace(go.Bar(y=lignes, x=t[c], name=c, orientation="h", marker_color=COULEUR_CAUSE[c],
                             marker_line=dict(color=SURFACE, width=2), legendgroup="cause",
                             hovertemplate="%{y} : %{x:.0f} % " + c.lower() + "<extra></extra>"))
    fr = d[(d["pays"] == "FR") & (d["type_episode"] == "penurie")]
    if len(fr):
        g = fr["gravite"].fillna("non précisée").value_counts(normalize=True) * 100
        for nom, coul in (("rupture", "#1c5cab"), ("tension", "#86b6ef"), ("non précisée", "#e1e0d9")):
            fig.add_trace(go.Bar(y=["France (gravité)"], x=[g.get(nom, 0)], name=f"France : {nom}", orientation="h",
                                 marker_color=coul, marker_line=dict(color=SURFACE, width=2),
                                 hovertemplate=f"France : %{{x:.0f}} % {nom}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return mise_en_page(fig, 440, barmode="stack", xaxis=dict(title="part des épisodes (%)", range=[0, 100]),
                        legend=dict(orientation="h", y=-0.18, yanchor="top", x=0, traceorder="normal"))


# ── Vue 4 : pays comparés ───────────────────────────────────────────────────────────────────
@app.callback(Output("cooc", "figure"), Input("onglets", "value"))
def fig_cooc(_) -> go.Figure:
    m = COOC.pivot(index="a", columns="b", values="p").reindex(index=PAYS, columns=PAYS).to_numpy(copy=True) * 100
    np.fill_diagonal(m, np.nan)
    texte = [["" if np.isnan(v) else f"{v:.0f} %" for v in ligne] for ligne in m]
    fig = go.Figure(go.Heatmap(z=m, x=[NOMS[p] for p in PAYS], y=[NOMS[p] for p in PAYS], colorscale=BLEUS, zmin=0, zmax=60,
                               text=texte, texttemplate="%{text}", xgap=2, ygap=2,
                               colorbar=dict(thickness=10, title=dict(text="%", side="right")),
                               hovertemplate="Quand %{y} manque d'une substance,<br>%{x} en manque aussi : %{z:.0f} %<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="si pénurie en…")
    fig.update_xaxes(title="… aussi en pénurie en", side="bottom")
    return mise_en_page(fig, 470)


@app.callback(Output("spearman", "figure"), Input("onglets", "value"))
def fig_spearman(_) -> go.Figure:
    r = SPEARMAN
    fig = go.Figure(go.Heatmap(z=r.values, x=r.columns, y=r.index, colorscale=DIVERGENT, zmin=-1, zmax=1, xgap=2, ygap=2,
                               text=[[f"{v:.2f}" for v in ligne] for ligne in r.values], texttemplate="%{text}",
                               textfont_size=9.5, colorbar=dict(thickness=10, title=dict(text="ρ", side="right")),
                               hovertemplate="%{y} × %{x}<br>ρ = %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-40)
    return mise_en_page(fig, 470)


# ── Vue 5 : fiche substance ─────────────────────────────────────────────────────────────────
@app.callback(Output("substance-badges", "children"), Output("gantt", "figure"), Output("sub-signalements", "figure"),
              Output("sub-score", "figure"), Output("sub-conso", "figure"), Output("voisines", "children"),
              Input("substance", "value"))
def fiche(atc5: str):
    s = SUB.set_index("atc5").loc[atc5]
    badges = [html.Span(f"{s['nom']} · {atc5}", className="badge", style={"fontWeight": 650}),
              html.Span(f"Classe {s['atc1']} · {ATC1.get(s['atc1'], '')}", className="badge"),
              html.Span(f"{s['atc4']} · {s['libelle_atc4'] or ''}", className="badge")]
    if s["ue_critique"]:
        badges.append(html.Span("Liste critique de l'UE", className="badge fort"))
    if s["mitm"]:
        badges.append(html.Span("Suivie dans le panel (MITM)", className="badge"))
    if pd.notna(s["score"]):
        badges.append(html.Span(f"Risque prédit ({META['mois_prediction']}) : {s['score']:.0%} · rang {s['rang']} / "
                                f"{len(PREVISION)}", className="badge fort" if s["score"] >= 0.5 else "badge"))

    e = EP[EP["atc5"] == atc5].copy()
    e["fin_aff"] = e["fin"].fillna(FIN_VUE).clip(upper=FIN_VUE)
    e["debut_aff"] = e["debut"].clip(lower=pd.Timestamp("2020-01-01"))
    e = e[e["fin_aff"] >= e["debut_aff"]]
    gantt = go.Figure()
    for p in PAYS:
        ep = e[e["pays"] == p]
        for type_, opacite in (("penurie", 0.85), ("arret_definitif", 0.35)):
            x = ep[ep["type_episode"] == type_]
            if x.empty:
                continue
            duree_ms = ((x["fin_aff"] - x["debut_aff"]).dt.total_seconds() * 1000).clip(lower=86400000 * 5)
            gantt.add_trace(go.Bar(base=x["debut_aff"], x=duree_ms, y=[NOMS[p]] * len(x), orientation="h", width=0.6,
                                   marker=dict(color=COULEUR[p], opacity=opacite), showlegend=False,
                                   customdata=np.stack([x["medicament"].fillna("").str.slice(0, 60),
                                                        x["debut"].dt.strftime("%d/%m/%Y"),
                                                        np.where(x["en_cours"], "en cours", x["fin"].dt.strftime("%d/%m/%Y")),
                                                        x["cause"].where(x["pays"] != "FR", x["gravite"].fillna(""))], axis=1),
                                   hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} → %{customdata[2]}<br>"
                                                 + ("Pénurie" if type_ == "penurie" else "Arrêt définitif")
                                                 + " · %{customdata[3]}<extra></extra>"))
    gantt.update_yaxes(categoryorder="array", categoryarray=[NOMS[p] for p in PAYS[::-1]])
    mise_en_page(gantt, 260, barmode="overlay", xaxis=dict(type="date", range=["2020-01-01", str(FIN_VUE.date())]))
    if e.empty:
        vide(gantt, "Aucun épisode connu pour cette substance")

    trim = SIG[SIG["atc5"] == atc5]["date_signalement"].dt.to_period("Q").value_counts()
    trim = trim.reindex(pd.period_range("2019Q1", "2025Q4", freq="Q"), fill_value=0)
    fig_sig = go.Figure(go.Bar(x=trim.index.to_timestamp(), y=trim.values, marker_color=COULEUR["FR"],
                               marker_line=dict(color=SURFACE, width=2), xperiod="M3", xperiodalignment="start",
                               customdata=trim.index.quarter, hovertemplate="%{x|%Y} T%{customdata} : %{y} signalements<extra></extra>"))
    mise_en_page(fig_sig, 280, yaxis_title="signalements", bargap=0.1)

    ev = EVAL[EVAL["atc5"] == atc5].sort_values("mois")
    fig_score = go.Figure()
    if ev.empty:
        vide(fig_score, "Substance hors du panel (pas de médicament MITM commercialisé)")
    else:
        for jeu in ("validation", "test"):
            g = ev[ev["jeu"] == jeu]
            fig_score.add_trace(go.Scatter(x=g["mois"], y=g["score"] * 100, mode="lines", name=f"score ({jeu})",
                                           line=dict(color=COULEUR["FR"], width=2), showlegend=False,
                                           hovertemplate="%{x|%m/%Y} : risque %{y:.0f} %<extra></extra>"))
        fig_score.add_trace(go.Scatter(x=ev["mois"], y=ev["cible"] * 100, mode="markers", name="signalée dans les 3 mois",
                                       marker=dict(size=8, color=ENCRE2, line=dict(color=SURFACE, width=2)),
                                       hovertemplate="%{x|%m/%Y} : " + "%{customdata}<extra></extra>",
                                       customdata=np.where(ev["cible"] == 1, "signalée ensuite", "pas de signalement")))
        if pd.notna(s["score"]):
            mois = pd.Timestamp(META["mois_prediction"] + "-01")
            fig_score.add_trace(go.Scatter(x=[mois], y=[s["score"] * 100], mode="markers+text", name="prévision",
                                           marker=dict(size=10, color=COULEUR["FR"], symbol="diamond"),
                                           text=["prévision"], textposition="middle left", textfont_color=ENCRE2,
                                           hovertemplate="%{x|%m/%Y} : prévision %{y:.0f} %<extra></extra>"))
    mise_en_page(fig_score, 280, yaxis=dict(title="%", range=[-5, 105]), showlegend=False)

    c = CONSO[CONSO["atc5"] == atc5].sort_values("annee")
    fig_conso = go.Figure(go.Bar(x=c["annee"], y=c["boites"], marker_color=COULEUR["FR"], marker_line=dict(color=SURFACE, width=2),
                                 hovertemplate="%{x} : %{y:,.0f} boîtes<extra></extra>"))
    mise_en_page(fig_conso, 280, yaxis_title="boîtes remboursées")
    if c.empty:
        vide(fig_conso, "Pas de consommation en ville connue")

    v = SUB[SUB["atc4"] == s["atc4"]].sort_values("nb_signalements", ascending=False)
    df = pd.DataFrame({"Substance": v["nom"], "ATC": v["atc5"], "Signalements ANSM": v["nb_signalements"],
                       **{f"Épisodes {p}": v[f"ep_{p}"] for p in PAYS},
                       "Risque prédit": v["score"].map(lambda x: "" if pd.isna(x) else f"{x:.0%}")})
    voisines = tableau(df, sort_action="native", page_size=8, style_data_conditional=[
        {"if": {"filter_query": f'{{ATC}} = "{atc5}"'}, "fontWeight": 700, "backgroundColor": "#eef4fc"}])
    return badges, gantt, fig_sig, fig_score, fig_conso, voisines


# ── Vue 6 : panel et qualité ────────────────────────────────────────────────────────────────
def bandes_jeux(fig: go.Figure) -> None:
    p = PANEL.sort_values("mois")
    for _, g in p.groupby((p["jeu"] != p["jeu"].shift()).cumsum()):
        jeu = g["jeu"].iat[0]
        fig.add_vrect(x0=g["mois"].min() - pd.Timedelta(days=15), x1=g["mois"].max() + pd.Timedelta(days=15),
                      fillcolor=JEUX[jeu][1], line_width=0, layer="below", annotation_text=JEUX[jeu][0],
                      annotation_position="top left", annotation_font=dict(size=11, color=ENCRE2))


@app.callback(Output("panel-cible", "figure"), Input("onglets", "value"))
def fig_panel(_) -> go.Figure:
    p = PANEL.sort_values("mois")
    fig = go.Figure(go.Scatter(x=p["mois"], y=p["cible"] * 100, mode="lines", line=dict(color=COULEUR["FR"], width=2),
                               customdata=p["lignes"], hovertemplate="%{x|%m/%Y} : %{y:.1f} % de %{customdata} substances<extra></extra>"))
    bandes_jeux(fig)
    return mise_en_page(fig, 340, yaxis=dict(title="part de positifs (%)", rangemode="tozero"), showlegend=False)


@app.callback(Output("panel-signaux", "figure"), Input("signaux-echelle", "value"))
def fig_signaux(echelle: str) -> go.Figure:
    noms = {"deja_signale": "Déjà signalée à l'ANSM (années précédentes)", "fr_affichee_3m": "Affichée en pénurie sur la page ANSM",
            "fr_autres": "Autre substance de sa classe ATC4 affichée (Q1)", "it_affichee_3m": "En pénurie en Italie (Q2)",
            "de_affichee_3m": "En pénurie en Allemagne (Q2)", "ue_critique": "Sur la liste critique de l'UE"}
    s = SIGNAUX.set_index("mois")[list(noms)].T
    z = s.to_numpy(dtype="float64", na_value=np.nan) * 100
    if echelle == "ligne":
        mini, maxi = np.nanmin(z, axis=1, keepdims=True), np.nanmax(z, axis=1, keepdims=True)
        couleur = (z - mini) / np.where(maxi - mini > 0, maxi - mini, 1)
    else:
        couleur = z
    fig = go.Figure(go.Heatmap(z=couleur, x=s.columns, y=[noms[c] for c in s.index], customdata=z, colorscale=BLEUS,
                               xgap=1, ygap=2, showscale=echelle == "brute",
                               colorbar=dict(thickness=10, title=dict(text="%", side="right")),
                               hovertemplate="%{y}<br>%{x|%m/%Y} : %{customdata:.1f} % des substances<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return mise_en_page(fig, 300, plot_bgcolor="#d9d8d2")


# ── Vue 7 : modèle et prévisions ────────────────────────────────────────────────────────────
@app.callback(Output("modele-metriques", "figure"), Input("onglets", "value"))
def fig_metriques(_) -> go.Figure:
    t = METRIQUES["test"]
    noms = {"ref_affichee_3m": "Règle : affichée sur la page ANSM (3 mois)",
            "ref_signalee_annee_prec": "Règle : signalée l'année précédente", RETENU: "Régression logistique (retenue)"}
    cles = list(noms)
    fig = go.Figure(go.Bar(x=[t[k]["pr_auc"] for k in cles], y=[noms[k] for k in cles], orientation="h", width=0.55,
                           marker_color=[GRIS_REF, GRIS_REF, COULEUR["FR"]], marker_line=dict(color=SURFACE, width=2),
                           text=[virgule(t[k]["pr_auc"]) for k in cles], textposition="outside", textfont_color=ENCRE2,
                           customdata=[[virgule(t[k]["roc_auc"]), f"{t[k]['precision_top100']:.0%}"] for k in cles],
                           hovertemplate="%{y}<br>PR-AUC %{x:.3f} · ROC-AUC %{customdata[0]}<br>"
                                         "précision top 100 : %{customdata[1]}<extra></extra>"))
    taux = t[RETENU]["taux_positifs"]
    fig.add_vline(x=taux, line_dash="dot", line_color=MUET, annotation_text=f"hasard {virgule(taux, 2)}",
                  annotation_position="bottom right", annotation_font=dict(size=11, color=MUET))
    fig.update_yaxes(autorange="reversed")
    return mise_en_page(fig, 300, xaxis=dict(title="PR-AUC (test 2025)", range=[0, 1]), showlegend=False)


@app.callback(Output("modele-questions", "figure"), Input("onglets", "value"))
def fig_questions(_) -> go.Figure:
    q = {k: v for k, v in METRIQUES["questions"].items() if isinstance(v, dict)}
    fig = go.Figure()
    for i, (nom, r) in enumerate(q.items()):
        e = r["test"]["ecart_pr_auc"]
        question, famille = nom.split(" | ")
        y = f"{question} · {famille}"
        fig.add_trace(go.Scatter(x=[e["ecart"]], y=[y], mode="markers", showlegend=False,
                                 marker=dict(size=10, color=COULEUR["FR"] if "Q1" in question else COULEUR["IT"],
                                             line=dict(color=SURFACE, width=2)),
                                 error_x=dict(type="data", symmetric=False, array=[e["ic95"][1] - e["ecart"]],
                                              arrayminus=[e["ecart"] - e["ic95"][0]], color=ENCRE2, thickness=1.5, width=6),
                                 hovertemplate=f"{y}<br>écart {e['ecart']:+.4f}<br>IC 95 % [{e['ic95'][0]:+.4f} ; "
                                               f"{e['ic95'][1]:+.4f}]<extra></extra>"))
    fig.add_vline(x=0, line_color=AXE, line_width=1.5)
    fig.update_yaxes(autorange="reversed")
    return mise_en_page(fig, 300, xaxis=dict(title="écart de PR-AUC (avec − sans)", tickformat="+.3f"))


@app.callback(Output("modele-topk", "figure"), Input("onglets", "value"))
def fig_topk(_) -> go.Figure:
    test = EVAL[EVAL["jeu"] == "test"]
    lignes = []
    for mois, g in test.groupby("mois"):
        lignes.append({"mois": mois, "modele": g.nlargest(100, "score")["cible"].mean() * 100,
                       "reference": g.nlargest(100, "score_reference")["cible"].mean() * 100,
                       "hasard": g["cible"].mean() * 100})
    t = pd.DataFrame(lignes)
    fig = go.Figure()
    for col, nom, coul, tiret in (("modele", "Régression logistique", COULEUR["FR"], None),
                                  ("reference", "Règle : signalée l'année précédente", GRIS_REF, None),
                                  ("hasard", "Hasard (taux de positifs du mois)", AXE, "dot")):
        fig.add_trace(go.Scatter(x=t["mois"], y=t[col], name=nom, mode="lines+markers", line=dict(color=coul, width=2, dash=tiret),
                                 marker=dict(size=8, line=dict(color=SURFACE, width=2)),
                                 hovertemplate=nom + " : %{y:.0f} %<extra></extra>"))
    return mise_en_page(fig, 340, hovermode="x unified", yaxis=dict(title="% des 100 signalées ensuite", range=[0, 102]))


@app.callback(Output("prev-tableau", "children"), Input("prev-classe", "value"), Input("prev-affichee", "value"))
def tableau_prevision(classe: str, affichee: str):
    p = PREVISION.merge(SUB[["atc5", "nom", "nb_signalements"]], on="atc5", how="left")
    if classe != "tout":
        p = p[p["atc5"].str[0] == classe]
    if affichee == "non":
        p = p[p["fr_affichee_3m"].fillna(0) == 0]
    df = pd.DataFrame({"Rang": p["rang"], "Substance": p["nom"].fillna(p["atc5"]), "ATC": p["atc5"],
                       "Risque (%)": (p["score"] * 100).round(0).astype(int),
                       "Signalements 2025": p["sig_an_prec"].astype("Int64"),
                       "Affichée FR (3 mois)": np.where(p["fr_affichee_3m"].fillna(0) == 1, "oui", ""),
                       "Italie (3 mois)": np.where(p["it_affichee_3m"].fillna(0) == 1, "oui", ""),
                       "Allemagne (3 mois)": np.where(p["de_affichee_3m"].fillna(0) == 1, "oui", ""),
                       "Liste critique UE": np.where(p["ue_critique"] == 1, "oui", "")})
    return [html.P(f"{len(df)} substances ; {(p['score'] >= 0.5).sum()} à plus de 50 %.", className="sous-titre"),
            tableau(df, sort_action="native", filter_action="native", page_size=15)]


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
