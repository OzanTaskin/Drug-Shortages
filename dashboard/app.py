"""Dashboard d'exploration du dataset DALAS (livrable 2) : les pénuries de médicaments en France et en Europe.

Lancement (depuis la racine du projet) :
    pip install -r requirements.txt
    python dashboard/app.py          puis ouvrir http://127.0.0.1:8050

Le dashboard ne lit que les petites tables de dashboard/donnees/ (versionnées) ; il n'a pas besoin des
données brutes. Ces tables se reconstruisent avec dashboard/construire_donnees.py.
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
META = json.loads((DONNEES / "meta.json").read_text())

# ── Référentiels ────────────────────────────────────────────────────────────────────────────
PAYS = ["FR", "IT", "DE", "NO", "SE", "CZ", "SK"]
NOMS = {"FR": "France", "IT": "Italie", "DE": "Allemagne", "NO": "Norvège", "SE": "Suède",
        "CZ": "Tchéquie", "SK": "Slovaquie"}
ISO3 = {"FR": "FRA", "IT": "ITA", "DE": "DEU", "NO": "NOR", "SE": "SWE", "CZ": "CZE", "SK": "SVK"}
POPULATION = {"FR": 68.4, "IT": 59.0, "DE": 83.5, "NO": 5.55, "SE": 10.55, "CZ": 10.9, "SK": 5.42}  # millions, 2024
SOURCES = {
    "FR": ("ANSM, page des disponibilités", "archives Wayback (photos)"),
    "IT": ("AIFA, liste des médicaments manquants", "archives Wayback (photos)"),
    "DE": ("BfArM, Lieferengpässe", "archives Wayback (photos)"),
    "NO": ("DMP, registres annuels", "fichiers annuels"),
    "SE": ("Läkemedelsverket, open data XML", "registre complet"),
    "CZ": ("SÚKL, API des déclarations", "registre complet"),
    "SK": ("ŠÚKL, export des déclarations", "registre complet"),
}
# Ordre fixe de la palette catégorielle validée (couleur = pays, jamais son rang).
COULEUR = {"FR": "#2a78d6", "IT": "#eb6834", "DE": "#1baf7a", "NO": "#eda100", "SE": "#e87ba4",
           "CZ": "#008300", "SK": "#6250d6"}
ATC1 = {"A": "Digestif et métabolisme", "B": "Sang", "C": "Cardio-vasculaire", "D": "Dermatologie",
        "G": "Génito-urinaire", "H": "Hormones", "J": "Anti-infectieux", "L": "Cancérologie, immunité",
        "M": "Muscles et squelette", "N": "Système nerveux", "P": "Antiparasitaires", "R": "Respiratoire",
        "S": "Organes sensoriels", "V": "Divers"}
CAUSES = ["Production / qualité", "Demande en hausse", "Distribution / capacité", "Arrêt / raison commerciale",
          "Réglementaire", "Autre", "Non renseignée"]
COULEUR_CAUSE = dict(zip(CAUSES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#898781", "#d6d5cf"]))
JEUX = {"entrainement": ("Entraînement", "rgba(42,120,214,0.07)"), "validation": ("Validation", "rgba(235,104,52,0.09)"),
        "test": ("Test", "rgba(27,175,122,0.10)"), "ecart": ("Écart", "rgba(137,135,129,0.10)")}
BLEUS = [[0, "#f3f7fd"], [0.15, "#cde2fb"], [0.35, "#86b6ef"], [0.6, "#3987e5"], [0.8, "#1c5cab"], [1, "#0d366b"]]
DIVERGENT = [[0, "#1c5cab"], [0.25, "#86b6ef"], [0.5, "#f0efec"], [0.75, "#f09a9a"], [1, "#b42828"]]

SURFACE, GRILLE, AXE, ENCRE, ENCRE2, MUET = "#fcfcfb", "#e1e0d9", "#c3c2b7", "#0b0b0b", "#52514e", "#898781"


def fmt(n: float) -> str:
    return f"{n:,.0f}".replace(",", " ")


def mise_en_page(fig: go.Figure, hauteur: int = 380, **kw) -> go.Figure:
    base = dict(
        height=hauteur, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, margin=dict(l=8, r=8, t=10, b=8),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=12.5, color=ENCRE2),
        hoverlabel=dict(bgcolor="white", font_size=12.5, bordercolor=GRILLE),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title_text=""))
    fig.update_layout(**{**base, **kw})
    fig.update_xaxes(gridcolor=GRILLE, linecolor=AXE, zeroline=False, ticks="")
    fig.update_yaxes(gridcolor=GRILLE, linecolor=AXE, zeroline=False, ticks="")
    return fig


def carte(titre: str, sous_titre: str, contenu, lecture: str | None = None) -> html.Div:
    enfants = [html.H3(titre), html.P(sous_titre, className="sous-titre"), contenu]
    if lecture:
        enfants.append(html.Details([html.Summary("Comment lire ce graphique"), html.P(lecture)], className="lecture"))
    return html.Div(enfants, className="carte")


def filtre(titre: str, composant, large: bool = False) -> html.Div:
    return html.Div([html.Label(titre, className="titre"), composant],
                    className="filtre filtre-large" if large else "filtre")


def graphe(id_: str) -> dcc.Graph:
    return dcc.Graph(id=id_, config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


# ── Pré-calculs ─────────────────────────────────────────────────────────────────────────────
MOIS_SERIE = sorted(SERIE["mois"].unique())
ANNEES_CARTE = list(range(2019, 2027))
PEN = EP[EP["type_episode"] == "penurie"]
LIB_ATC2 = SUB.drop_duplicates("atc2").set_index("atc2")["libelle_atc2"].fillna("")
MITM = SUB[SUB["mitm"]].copy()
OPTIONS_SUBSTANCES = [{"label": f"{r.nom} ({r.atc5})", "value": r.atc5}
                      for r in SUB.sort_values("nom").itertuples()]


def couverture(pays: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    debut = pd.Timestamp(META["couverture"][pays] + "-01")
    fin = pd.Timestamp(META["fin_couverture"].get(pays, META["dernier_mois"]) + "-01")
    return debut, fin


def tableau_qualite() -> pd.DataFrame:
    lignes = []
    for p in PAYS:
        e = EP[EP["pays"] == p]
        d, f = couverture(p)
        nb_photos = PHOTOS[PHOTOS["pays"] == p]["date_photo"].nunique()
        lignes.append({
            "Pays": NOMS[p], "Source": SOURCES[p][0], "Mode de collecte": SOURCES[p][1],
            "Période observée": f"{d:%m/%Y} – {f:%m/%Y}",
            "Photos": fmt(nb_photos) if nb_photos else "—",
            "Épisodes": fmt(len(e)), "Dont arrêts définitifs": fmt((e["type_episode"] == "arret_definitif").sum()),
            "ATC inconnu": f"{e['atc5'].isna().mean():.1%}", "En cours": f"{e['en_cours'].mean():.1%}",
            "Début avant l'archive": f"{e['censure_gauche'].mean():.1%}",
            "Cause renseignée": f"{(e['cause'] != 'Non renseignée').mean():.0%}",
        })
    return pd.DataFrame(lignes)


def spearman_substances() -> tuple[pd.DataFrame, int]:
    v = pd.DataFrame({
        "Consommation 2024 (log)": np.log10(MITM["boites_2024"].fillna(0) + 1),
        "Présentations": MITM["nb_presentations"].fillna(0),
        "Signalements ANSM": MITM["nb_signalements"],
        "Durée médiane": MITM["duree_mediane_j"],
        "Liste critique UE": MITM["ue_critique"].astype(int),
        **{f"Épisodes {p}": MITM[f"ep_{p}"] for p in PAYS},
    })
    r = v.corr(method="spearman")
    dist = squareform((1 - r.abs()).clip(lower=0).to_numpy(), checks=False)
    ordre = r.index[leaves_list(linkage(dist, "average"))]
    return r.loc[ordre, ordre], len(v)


QUALITE = tableau_qualite()
SPEARMAN, N_SPEARMAN = spearman_substances()

# ── Mise en page ────────────────────────────────────────────────────────────────────────────
TUILES = [
    (len(PAYS), "pays européens"), (META["episodes"], "épisodes de pénurie ou d'arrêt"),
    (len(SIG), "signalements ANSM (2019–2025)"), (META["panel_substances"], "substances suivies (MITM)"),
    (META["panel_lignes"], "lignes du panel substance × mois"), ("2015–2026", "période couverte"),
]


def vue_europe() -> html.Div:
    return html.Div([
        html.P(["Sept agences publient leurs pénuries, chacune avec sa définition et sur sa propre période. ",
                html.B("Cliquez sur un pays"), " pour afficher son profil. Les pays gris ne sont pas observés à la date choisie."],
               className="intro"),
        html.Div([
            filtre("Mesure", dcc.RadioItems(id="carte-mesure", value="million", inline=True, options=[
                {"label": "pénuries en cours par million d'habitants", "value": "million"},
                {"label": "pénuries en cours (effectif)", "value": "effectif"}])),
            filtre("Date (janvier de chaque année, octobre pour 2026)", dcc.Slider(
                id="carte-annee", min=2019, max=2026, step=1, value=2024,
                marks={a: str(a) for a in ANNEES_CARTE}), large=True),
        ], className="filtres"),
        html.Div([
            carte("Pénuries de médicaments en cours", "Présentations ou produits en pénurie, selon la définition de chaque agence",
                  graphe("carte-europe"),
                  "Le nombre affiché compte les épisodes de pénurie ouverts au mois choisi (les arrêts définitifs sont exclus). "
                  "Les agences ne comptent pas la même chose : l'Italie recense chaque présentation, la France seulement les "
                  "médicaments d'intérêt thérapeutique majeur. Comparez les évolutions plutôt que les niveaux."),
            html.Div(id="profil-pays", className="carte profil"),
        ], className="grille-3-2"),
        carte("Ce que chaque source observe, et quand", "Barre = période couverte ; trait = une archive (photo) récupérée",
              graphe("couverture"),
              "La France, l'Italie et l'Allemagne n'existent que par des photos de leur page web archivées par la "
              "Wayback Machine : entre deux photos, on ne voit rien. Suède, Tchéquie et Slovaquie publient un registre "
              "complet ; la Norvège publie un fichier par an jusqu'en juin 2024."),
    ], className="vue")


def vue_temps() -> html.Div:
    return html.Div([
        html.P(["Les registres européens montrent une hausse nette des pénuries depuis 2018. ",
                "Les pointillés de couverture sont des ", html.B("absences de données"), ", pas des zéros."],
               className="intro"),
        html.Div([
            filtre("Pays", dcc.Checklist(id="temps-pays", value=PAYS, inline=True,
                                         options=[{"label": NOMS[p], "value": p} for p in PAYS])),
            filtre("Mesure", dcc.RadioItems(id="temps-mesure", value="en_cours", inline=True, options=[
                {"label": "en cours", "value": "en_cours"}, {"label": "nouvelles par mois", "value": "nouvelles"}])),
            filtre("Échelle", dcc.RadioItems(id="temps-echelle", value="indice", inline=True, options=[
                {"label": "par million d'habitants", "value": "million"}, {"label": "effectif", "value": "effectif"},
                {"label": "indice (2022 = 100)", "value": "indice"}])),
            filtre("Lissage", dcc.Checklist(id="temps-lissage", value=["oui"],
                                            options=[{"label": "moyenne mobile 3 mois", "value": "oui"}])),
        ], className="filtres"),
        carte("Pénuries dans le temps, par pays", "Une courbe par agence, sur sa seule période d'observation",
              graphe("temps-courbes"),
              "« Indice » ramène chaque pays à sa moyenne de 2022 (= 100) : on compare alors la dynamique, sans "
              "l'effet des définitions nationales. L'Allemagne n'est observée que depuis juillet 2024."),
        html.Div([
            carte("Signalements ANSM : saisonnalité", "Nombre de déclarations de risque de rupture, par mois et par année",
                  graphe("saison"),
                  "Chaque case compte les signalements envoyés à l'ANSM par les laboratoires. Une ligne plus foncée "
                  "= une année plus chargée : 2023 compte plus de trois fois 2019."),
            carte("Signalements ANSM : qui déclare ?", "Part cumulée des signalements selon le nombre de laboratoires",
                  graphe("labos"),
                  "Les laboratoires sont triés du plus au moins déclarant. Une courbe qui monte très vite signifie "
                  "que quelques laboratoires (souvent des génériqueurs) font l'essentiel des déclarations."),
        ], className="grille-2"),
    ], className="vue")


def vue_classes() -> html.Div:
    return html.Div([
        html.P(["Les 885 substances suivies sont rangées dans l'arbre ATC (classe → sous-classe → substance). ",
                html.B("Cliquez sur une classe"), " de l'arbre pour filtrer la carte de chaleur, puis ",
                html.B("sur une case"), " de la carte pour lister les substances."], className="intro"),
        html.Div([
            filtre("Taille des secteurs", dcc.RadioItems(id="arbre-taille", value="nb", inline=True, options=[
                {"label": "nombre de substances", "value": "nb"}, {"label": "consommation 2024 (boîtes)", "value": "conso"}])),
            filtre("Couleur", dcc.RadioItems(id="arbre-couleur", value="sig", inline=True, options=[
                {"label": "signalements par substance", "value": "sig"},
                {"label": "pays européens touchés", "value": "pays"}])),
        ], className="filtres"),
        html.Div([
            carte("L'arbre thérapeutique", "Classe ATC1 → ATC2 → substance", graphe("arbre"),
                  "La surface d'un secteur est proportionnelle au nombre de substances (ou à la consommation). "
                  "Sa couleur indique la moyenne de la mesure choisie : plus c'est foncé, plus la classe est touchée."),
            html.Div([
                carte("Quand chaque classe est touchée", "Part des substances de la sous-classe ATC2 touchées ce mois-là",
                      html.Div([
                          html.Div([
                              filtre("Source", dcc.RadioItems(id="chaleur-source", value="signalements", inline=True, options=[
                                  {"label": "signalement ANSM", "value": "signalements"},
                                  {"label": "pénurie en cours (FR)", "value": "episodes_fr"}])),
                              filtre("Ordre des lignes", dcc.RadioItems(id="chaleur-ordre", value="cluster", inline=True, options=[
                                  {"label": "profils similaires", "value": "cluster"}, {"label": "code ATC", "value": "code"}])),
                          ], className="filtres", style={"padding": "6px 0", "border": "none"}),
                          html.Div(id="chaleur-filtre", className="sous-titre"),
                          graphe("chaleur"),
                      ]),
                      "Une ligne par sous-classe ATC2 (seulement les substances suivies), une colonne par mois. "
                      "« Profils similaires » regroupe les lignes qui ont le même calendrier (clustering hiérarchique de Ward). "
                      "Les épisodes français ne sont fiables qu'à partir de 04/2021 (nouveau site de l'ANSM)."),
            ]),
        ], className="grille-2"),
        carte("Substances de la classe sélectionnée", "Cliquez sur une case de la carte de chaleur",
              html.Div(id="liste-substances")),
    ], className="vue")


def vue_durees() -> html.Div:
    return html.Div([
        html.P(["Les durées sont très asymétriques (quelques jours à plusieurs années) : l'axe est ",
                html.B("logarithmique"), ". Seules les pénuries terminées ont une durée connue."], className="intro"),
        html.Div([
            filtre("Pays", dcc.Checklist(id="duree-pays", value=PAYS, inline=True,
                                         options=[{"label": NOMS[p], "value": p} for p in PAYS])),
            filtre("Forme", dcc.RadioItems(id="duree-forme", value="boite", inline=True, options=[
                {"label": "boîtes à moustaches", "value": "boite"}, {"label": "courbe cumulée", "value": "ecdf"}])),
            filtre("Début de la pénurie", dcc.RangeSlider(id="duree-annees", min=2015, max=2026, step=1, value=[2019, 2026],
                                                          marks={a: str(a) for a in range(2015, 2027, 2)}), large=True),
        ], className="filtres"),
        html.Div([
            carte("Combien de temps dure une pénurie ?", "Durée en jours des pénuries terminées (échelle log)",
                  graphe("durees"),
                  "La boîte va du 1er au 3e quartile, le trait central est la médiane. En courbe cumulée, lisez "
                  "la part des pénuries terminées avant une durée donnée. Les épisodes ouverts avant la première "
                  "archive sont exclus, car leur début est inconnu."),
            carte("Pourquoi ? Les causes déclarées", "Familles de causes regroupées à partir des libellés de chaque agence",
                  graphe("causes"),
                  "Les libellés (italien, norvégien, suédois, tchèque, allemand) sont regroupés par mots-clés en cinq "
                  "familles. La France et la Slovaquie ne publient pas de cause."),
        ], className="grille-2"),
    ], className="vue")


def vue_pays() -> html.Div:
    return html.Div([
        html.P(["Q2 se demande si une pénurie à l'étranger annonce une pénurie en France. Avant de le tester, on regarde "
                "à quel point ", html.B("les mêmes substances manquent en même temps"), " dans plusieurs pays."],
               className="intro"),
        html.Div([
            carte("Pénuries simultanées", "Si une substance manque dans le pays en ligne, part des cas où elle manque aussi dans le pays en colonne (même mois)",
                  graphe("cooc"),
                  "Calculé sur les seuls mois observés dans les deux pays. Lecture : « quand la France manque d'une "
                  "substance, l'Italie en manque aussi dans x % des cas ». La matrice n'est pas symétrique : l'Italie "
                  "recense beaucoup plus de pénuries, donc elle « couvre » souvent les autres."),
            carte("Ce qui va ensemble, substance par substance", f"Corrélation de Spearman entre {len(SPEARMAN)} variables, sur les {N_SPEARMAN} substances suivies",
                  graphe("spearman"),
                  "Spearman (sur les rangs) plutôt que Pearson, car les variables sont très asymétriques. Rouge = les deux "
                  "variables montent ensemble, bleu = l'une monte quand l'autre baisse. Les variables sont rangées par "
                  "similarité. Une corrélation n'est pas une causalité : la consommation et le nombre de présentations "
                  "jouent sur presque tout."),
        ], className="grille-2"),
    ], className="vue")


def vue_substance() -> html.Div:
    return html.Div([
        html.P(["Toutes les sources réunies pour une substance : ses pénuries dans chaque pays, ses signalements "
                "à l'ANSM et sa consommation en France."], className="intro"),
        html.Div([filtre("Substance (nom ou code ATC)", dcc.Dropdown(id="substance", options=OPTIONS_SUBSTANCES,
                                                                        value="J01CA04", clearable=False), large=True)],
                 className="filtres"),
        html.Div(id="substance-badges", style={"marginBottom": "10px"}),
        carte("Ses pénuries, pays par pays", "Une barre par épisode, du début à la fin (en cours : jusqu'à la dernière observation)",
              graphe("gantt"),
              "Chaque ligne est un pays. Une barre = un épisode pour un produit ou une présentation contenant la "
              "substance ; plusieurs barres superposées = plusieurs produits en pénurie en même temps."),
        html.Div([
            carte("Signalements ANSM", "Déclarations des laboratoires, par trimestre", graphe("sub-signalements")),
            carte("Consommation en France", "Boîtes remboursées par l'Assurance Maladie, par an", graphe("sub-conso")),
        ], className="grille-2"),
        carte("Substances voisines (même classe ATC4)", "Les substituts possibles, au cœur de la question Q1",
              html.Div(id="voisines")),
    ], className="vue")


def vue_panel() -> html.Div:
    return html.Div([
        html.P(["Le panel de modélisation : ", html.B(f"{META['panel_substances']} substances × 66 mois"),
                " (04/2020–09/2025). Chaque variable n'utilise que ce qui était publié à la fin du mois, "
                "et le découpage entraînement / validation / test est chronologique."], className="intro"),
        carte("Les deux cibles au fil des mois", "Part de lignes positives ; fonds colorés = jeux d'entraînement, de validation et de test",
              graphe("panel-cibles"),
              "« Signalement dans les 3 mois » est facile à prévoir (une substance déjà signalée le sera encore). "
              "« Nouvelle pénurie » ne regarde que les substances sans signalement depuis 12 mois : c'est la cible "
              "difficile et utile. Les mois d'écart évitent qu'une cible du train chevauche la validation."),
        html.Div([
            filtre("Échelle des couleurs", dcc.RadioItems(id="signaux-echelle", value="ligne", inline=True, options=[
                {"label": "normalisée par variable", "value": "ligne"}, {"label": "part brute", "value": "brute"}])),
        ], className="filtres"),
        carte("Les signaux disponibles chaque mois", "Part des substances pour lesquelles le signal vaut 1 ; gris = pays non couvert ce mois-là",
              graphe("panel-signaux"),
              "En « normalisée par variable », chaque ligne va de son minimum (clair) à son maximum (foncé) : on voit "
              "quand un signal s'allume, même s'il est rare. Les cases grises sont des absences de données, jamais des zéros."),
        carte("Qualité et couverture des sources", "Ce qui est fiable, et ce qui ne l'est pas",
              dash_table.DataTable(
                  data=QUALITE.to_dict("records"), columns=[{"name": c, "id": c} for c in QUALITE.columns],
                  style_table={"overflowX": "auto"}, style_cell={"padding": "6px 8px", "textAlign": "left",
                                                                 "border": f"1px solid {GRILLE}", "whiteSpace": "normal"},
                  style_header={"fontWeight": 650, "backgroundColor": "#f3f2ee"})),
    ], className="vue")


app = Dash(__name__, title="Pénuries de médicaments — DALAS", suppress_callback_exceptions=True)
ONGLETS = [("europe", "Carte de l'Europe", vue_europe), ("temps", "Dans le temps", vue_temps),
           ("classes", "Classes thérapeutiques", vue_classes), ("durees", "Durées et causes", vue_durees),
           ("pays", "Pays comparés", vue_pays), ("substance", "Fiche substance", vue_substance),
           ("panel", "Panel et qualité", vue_panel)]
app.layout = html.Div([
    html.Header([
        html.H1("Pénuries de médicaments en France et en Europe"),
        html.P("Explorer le dataset du projet DALAS : 7 agences du médicament, les signalements ANSM, la consommation "
               "de l'Assurance Maladie et le panel construit pour prévoir les ruptures en France."),
    ]),
    html.Div([html.Div([html.Div(v if isinstance(v, str) else fmt(v), className="valeur"),
                        html.Div(l, className="libelle")], className="tuile") for v, l in TUILES], className="tuiles"),
    dcc.Tabs(id="onglets", value="europe", className="onglets", children=[
        dcc.Tab(label=lib, value=v, className="onglet", selected_className="onglet--actif", children=f())
        for v, lib, f in ONGLETS]),
    html.Footer("Ozan Taskin et Rahim Belkacemi — Master 1, Sorbonne Université, cours DALAS. Sources : ANSM, AIFA, "
                "BfArM, DMP, Läkemedelsverket, SÚKL (CZ), ŠÚKL (SK), Assurance Maladie (Open Medic), BDPM, liste critique de l'UE."),
], className="page")


# ── Vue 1 : carte et couverture ─────────────────────────────────────────────────────────────
def mois_carte(annee: int) -> pd.Timestamp:
    return pd.Timestamp(META["dernier_mois"] + "-01") if annee == 2026 else pd.Timestamp(f"{annee}-01-01")


@app.callback(Output("carte-europe", "figure"), Input("carte-mesure", "value"), Input("carte-annee", "value"))
def fig_carte(mesure: str, annee: int) -> go.Figure:
    mois = mois_carte(annee)
    valeurs = SERIE[SERIE["mois"] == mois].set_index("pays")["en_cours"]
    lignes = []
    for p in PAYS:
        d, f = couverture(p)
        n = valeurs.get(p, np.nan) if d <= mois <= f else np.nan
        lignes.append((p, n, n / POPULATION[p] if mesure == "million" else n))
    df = pd.DataFrame(lignes, columns=["pays", "n", "v"])
    vus, absents = df.dropna(), df[df["n"].isna()]
    fig = go.Figure()
    fig.add_choropleth(locations=[ISO3[p] for p in absents["pays"]], z=[0] * len(absents),
                       colorscale=[[0, "#c9c8c1"], [1, "#c9c8c1"]], showscale=False, marker_line_color="white",
                       text=[f"{NOMS[p]} : non observé en {mois:%m/%Y}" for p in absents["pays"]], hoverinfo="text")
    unite = "par million d'hab." if mesure == "million" else "épisodes"
    fig.add_choropleth(locations=[ISO3[p] for p in vus["pays"]], z=vus["v"], customdata=vus["pays"],
                       colorscale=[[0, "#cde2fb"], [0.3, "#86b6ef"], [0.6, "#3987e5"], [0.8, "#1c5cab"], [1, "#0d366b"]], zmin=0, marker_line_color="white", marker_line_width=1,
                       colorbar=dict(title=dict(text=unite, side="right"), thickness=10, len=0.7),
                       text=[f"<b>{NOMS[p]}</b><br>{fmt(n)} pénuries en cours<br>{n / POPULATION[p]:.1f} par million d'hab."
                             for p, n in zip(vus["pays"], vus["n"])], hoverinfo="text")
    fig.update_geos(scope="europe", projection_type="natural earth", showcountries=True, countrycolor="white",
                    showland=True, landcolor="#f3f2ee", showframe=False, bgcolor=SURFACE, showcoastlines=False,
                    lataxis_range=[36, 71], lonaxis_range=[-12, 32])
    return mise_en_page(fig, 470, clickmode="event")


@app.callback(Output("profil-pays", "children"), Input("carte-europe", "clickData"))
def profil(click) -> list:
    p = "FR"
    if click and click["points"][0].get("customdata"):
        p = click["points"][0]["customdata"]
    e = EP[EP["pays"] == p]
    pen = e[e["type_episode"] == "penurie"]
    d, f = couverture(p)
    top = (pen.dropna(subset=["atc5"]).groupby("atc5").size().nlargest(8).rename("Épisodes").reset_index()
           .merge(SUB[["atc5", "nom"]], on="atc5", how="left"))
    top = top.assign(Substance=top["nom"].fillna(top["atc5"]).str.slice(0, 34))[["Substance", "atc5", "Épisodes"]]
    return [
        html.H3(NOMS[p]), html.P(f"{SOURCES[p][0]} — {SOURCES[p][1]}", className="sous-titre"),
        html.Dl([html.Dt("Période observée"), html.Dd(f"{d:%m/%Y} – {f:%m/%Y}"),
                 html.Dt("Épisodes de pénurie"), html.Dd(fmt(len(pen))),
                 html.Dt("Arrêts définitifs"), html.Dd(fmt(len(e) - len(pen))),
                 html.Dt("Durée médiane (terminées)"), html.Dd(f"{pen['duree_j'].median():.0f} jours"),
                 html.Dt("Substances touchées"), html.Dd(fmt(pen["atc5"].nunique()))]),
        html.P("Substances les plus souvent en pénurie", className="sous-titre"),
        dash_table.DataTable(data=top.to_dict("records"), columns=[{"name": c if c != "atc5" else "ATC", "id": c} for c in top.columns],
                             style_cell={"padding": "4px 8px", "textAlign": "left", "border": f"1px solid {GRILLE}"},
                             style_header={"fontWeight": 650, "backgroundColor": "#f3f2ee"}),
        html.P("Cliquez sur un autre pays de la carte.", className="lecture"),
    ]


@app.callback(Output("couverture", "figure"), Input("onglets", "value"))
def fig_couverture(_) -> go.Figure:
    fig = go.Figure()
    for i, p in enumerate(PAYS[::-1]):
        d, f = couverture(p)
        fig.add_trace(go.Scatter(x=[d, f + pd.offsets.MonthEnd(0)], y=[NOMS[p]] * 2, mode="lines",
                                 line=dict(color=COULEUR[p], width=12), opacity=0.35, showlegend=False,
                                 hovertemplate=f"{NOMS[p]} : {d:%m/%Y} – {f:%m/%Y}<extra></extra>"))
        dates = PHOTOS[PHOTOS["pays"] == p]["date_photo"].drop_duplicates()
        if len(dates):
            fig.add_trace(go.Scatter(x=dates, y=[NOMS[p]] * len(dates), mode="markers", showlegend=False,
                                     marker=dict(symbol="line-ns-open", size=14, color=COULEUR[p], line_width=2),
                                     hovertemplate=f"{NOMS[p]} : photo du %{{x|%d/%m/%Y}}<extra></extra>"))
    fig.add_vline(x=pd.Timestamp("2021-04-05"), line_color=MUET, line_width=1)
    fig.add_annotation(x=pd.Timestamp("2021-04-05"), y=1.02, yref="paper", text="nouveau site ANSM", showarrow=False,
                       font_size=11, font_color=MUET, xanchor="left")
    return mise_en_page(fig, 300, xaxis_range=["2015-01-01", "2026-12-31"])


# ── Vue 2 : séries temporelles ──────────────────────────────────────────────────────────────
@app.callback(Output("temps-courbes", "figure"), Input("temps-pays", "value"), Input("temps-mesure", "value"),
              Input("temps-echelle", "value"), Input("temps-lissage", "value"))
def fig_temps(pays: list, mesure: str, echelle: str, lissage: list) -> go.Figure:
    fig = go.Figure()
    for p in [p for p in PAYS if p in pays]:
        s = SERIE[SERIE["pays"] == p].set_index("mois")[mesure].sort_index()
        d, f = couverture(p)
        s = s.reindex(pd.date_range(d, f, freq="MS"), fill_value=0)
        if "oui" in (lissage or []):
            s = s.rolling(3, min_periods=1).mean()
        if echelle == "million":
            s = s / POPULATION[p]
        elif echelle == "indice":
            base = s[s.index.year == 2022].mean()
            s = s / base * 100 if base and not np.isnan(base) else s * np.nan
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=NOMS[p], mode="lines", line=dict(color=COULEUR[p], width=2),
                                 hovertemplate=f"{NOMS[p]}<br>%{{x|%m/%Y}} : %{{y:.1f}}<extra></extra>"))
        if s.notna().any():
            fig.add_annotation(x=s.index[-1], y=s.values[-1], text=p, showarrow=False, xanchor="left", xshift=4,
                               font=dict(size=11, color=ENCRE2))
    titre = {"million": "par million d'habitants", "effectif": "nombre d'épisodes", "indice": "indice, 2022 = 100"}[echelle]
    return mise_en_page(fig, 420, yaxis_title=titre, hovermode="x unified", xaxis_range=["2015-01-01", "2027-01-31"])


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
    rang = np.arange(1, len(n) + 1)
    fig = go.Figure(go.Scatter(x=rang, y=part.values, mode="lines", line=dict(color=COULEUR["FR"], width=2),
                               customdata=n.index, hovertemplate="les %{x} premiers labos : %{y:.0f} %<br>dernier ajouté : %{customdata}<extra></extra>"))
    for k in (10, 50):
        fig.add_trace(go.Scatter(x=[k], y=[part.iloc[k - 1]], mode="markers+text", marker=dict(size=8, color=COULEUR["FR"]),
                                 text=[f"{k} labos : {part.iloc[k - 1]:.0f} %"], textposition="middle right",
                                 showlegend=False, hoverinfo="skip"))
    return mise_en_page(fig, 330, showlegend=False, xaxis=dict(type="log", title=f"nombre de laboratoires (sur {len(n)}, log)"),
                        yaxis=dict(title="part des signalements (%)", range=[0, 102]))


# ── Vue 3 : classes ATC ─────────────────────────────────────────────────────────────────────
@app.callback(Output("arbre", "figure"), Input("arbre-taille", "value"), Input("arbre-couleur", "value"))
def fig_arbre(taille: str, couleur: str) -> go.Figure:
    m = MITM.assign(l1=MITM["atc1"] + " · " + MITM["atc1"].map(ATC1),
                    l2=MITM["atc2"] + " · " + MITM["libelle_atc2"].fillna("").str.slice(0, 32),
                    poids=1.0 if taille == "nb" else MITM["boites_2024"].fillna(0).clip(lower=1),
                    mesure=MITM["nb_signalements"] if couleur == "sig" else MITM["nb_pays_touches"] - (MITM["ep_FR"] > 0))
    m["pm"] = m["poids"] * m["mesure"]
    m["id1"] = m["l1"]
    m["id2"] = m["l1"] + "/" + m["l2"]
    m["id3"] = m["id2"] + "/" + m["atc5"]
    niveaux = []
    for id_, parent, label in [("id1", None, "l1"), ("id2", "id1", "l2"), ("id3", "id2", "nom")]:
        g = m.groupby(id_).agg(poids=("poids", "sum"), pm=("pm", "sum"), label=(label, "first"),
                               parent=(parent, "first") if parent else (id_, lambda s: ""))
        niveaux.append(g.reset_index().rename(columns={id_: "id"}))
    n = pd.concat(niveaux, ignore_index=True)
    ids, labels, parents, poids = n["id"], n["label"], n["parent"], n["poids"]
    mesures = n["pm"] / n["poids"]
    if couleur == "sig":  # très asymétrique (médiane 6, maximum 343) : couleur en log
        unite, couleurs = "signalements par substance (log)", np.log10(mesures + 1)
        barre = dict(tickvals=np.log10(np.array([0, 3, 10, 30, 100, 300]) + 1), ticktext=["0", "3", "10", "30", "100", "300"])
    else:
        unite, couleurs, barre = "pays étrangers touchés (sur 6)", mesures, {}
    fig = go.Figure(go.Sunburst(ids=ids, labels=labels, parents=parents, values=poids, branchvalues="total",
                                customdata=mesures,
                                marker=dict(colors=couleurs, colorscale=BLEUS, line=dict(color="white", width=1),
                                            colorbar=dict(thickness=10, title=dict(text=unite, side="right"), **barre)),
                                maxdepth=2, insidetextorientation="radial",
                                hovertemplate="<b>%{label}</b><br>" + unite.replace(" (log)", "")
                                              + " : %{customdata:.1f}<extra></extra>"))
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
    if ordre == "code":
        lignes = sorted(lignes)
    lignes = [l for l in lignes if l.startswith(prefixe)]
    large = t.pivot(index="atc2", columns="mois", values="part").reindex(index=lignes).fillna(0)
    taille = MITM.groupby("atc2").size()
    if not lignes:
        fig = go.Figure()
        fig.add_annotation(text=f"Aucune sous-classe de {prefixe} ne compte {META['min_taille_atc2']} substances suivies ou plus",
                           x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font_color=MUET)
        return mise_en_page(fig, 200, xaxis_visible=False, yaxis_visible=False), f"Filtré sur {prefixe}"
    noms = [f"{c} · {LIB_ATC2.get(c, '')[:30]} ({taille.get(c, 0)})" for c in large.index]
    fig = go.Figure(go.Heatmap(z=large.values * 100, x=large.columns, y=noms, colorscale=BLEUS, zmin=0,
                               zmax=max(5, np.nanpercentile(large.values * 100, 99)) if large.size else 5, ygap=1,
                               customdata=np.repeat(np.array(list(large.index))[:, None], large.shape[1], axis=1),
                               colorbar=dict(thickness=10, title=dict(text="% touchées", side="right")),
                               hovertemplate="%{y}<br>%{x|%m/%Y} : %{z:.0f} % des substances<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickfont_size=10.5)
    texte = (f"Filtré sur la classe {prefixe} (cliquez au centre de l'arbre pour tout réafficher)" if prefixe
             else f"Les {len(lignes)} sous-classes ATC2 d'au moins {META['min_taille_atc2']} substances suivies "
                  "(nombre entre parenthèses)")
    return mise_en_page(fig, max(320, 26 + 13 * len(lignes))), texte


@app.callback(Output("liste-substances", "children"), Input("chaleur", "clickData"))
def liste_substances(click):
    if not click:
        return html.P("Aucune case sélectionnée.", className="sous-titre")
    atc2 = click["points"][0]["customdata"]
    m = MITM[MITM["atc2"] == atc2].sort_values("nb_signalements", ascending=False)
    df = pd.DataFrame({"Substance": m["nom"], "ATC": m["atc5"], "Signalements ANSM": m["nb_signalements"],
                       **{f"Épisodes {p}": m[f"ep_{p}"] for p in PAYS},
                       "Liste critique UE": np.where(m["ue_critique"], "oui", "")})
    return [html.P(f"{atc2} · {LIB_ATC2.get(atc2, '')} — {len(df)} substances", className="sous-titre"),
            dash_table.DataTable(data=df.to_dict("records"), columns=[{"name": c, "id": c} for c in df.columns],
                                 sort_action="native", page_size=12, style_table={"overflowX": "auto"},
                                 style_cell={"padding": "4px 8px", "textAlign": "left", "border": f"1px solid {GRILLE}"},
                                 style_header={"fontWeight": 650, "backgroundColor": "#f3f2ee"})]


# ── Vue 4 : durées et causes ────────────────────────────────────────────────────────────────
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
            fig.add_annotation(x=nom, y=np.log10(x.median()), text=f"{x.median():.0f} j", showarrow=False,
                               yshift=9, font=dict(size=11, color=ENCRE))
        else:
            xs = np.sort(x.to_numpy())
            fig.add_trace(go.Scatter(x=xs, y=np.arange(1, len(xs) + 1) / len(xs) * 100, name=NOMS[p], mode="lines",
                                     line=dict(color=COULEUR[p], width=2),
                                     hovertemplate=f"{NOMS[p]} : %{{y:.0f}} % durent moins de %{{x:.0f}} j<extra></extra>"))
    reperes = dict(tickvals=[1, 7, 30, 90, 365, 1095], ticktext=["1 j", "1 sem.", "1 mois", "3 mois", "1 an", "3 ans"])
    if forme == "boite":
        return mise_en_page(fig, 440, showlegend=False, yaxis=dict(type="log", title="durée (échelle log)", **reperes))
    for v in (90, 365):
        fig.add_vline(x=v, line_color=GRILLE, line_width=1)
    return mise_en_page(fig, 440, xaxis=dict(type="log", title="durée (échelle log)", **reperes),
                        yaxis=dict(title="part des pénuries terminées (%)", range=[0, 101]))


@app.callback(Output("causes", "figure"), Input("duree-pays", "value"), Input("duree-annees", "value"))
def fig_causes(pays: list, annees: list) -> go.Figure:
    d = EP[EP["debut"].dt.year.between(*annees) & EP["pays"].isin(pays)]
    t = pd.crosstab(d["pays"], d["cause"], normalize="index").reindex(columns=CAUSES, fill_value=0) * 100
    t = t.reindex([p for p in PAYS if p in t.index])
    fig = go.Figure()
    for c in CAUSES:
        fig.add_trace(go.Bar(y=[NOMS[p] for p in t.index], x=t[c], name=c, orientation="h", marker_color=COULEUR_CAUSE[c],
                             marker_line=dict(color=SURFACE, width=2),
                             hovertemplate="%{y} : %{x:.0f} % " + c.lower() + "<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return mise_en_page(fig, 440, barmode="stack", xaxis=dict(title="part des épisodes (%)", range=[0, 100]),
                        legend=dict(orientation="h", y=-0.18, yanchor="top", x=0))


# ── Vue 5 : pays comparés ───────────────────────────────────────────────────────────────────
@app.callback(Output("cooc", "figure"), Input("onglets", "value"))
def fig_cooc(_) -> go.Figure:
    m = COOC.pivot(index="a", columns="b", values="p").reindex(index=PAYS, columns=PAYS).to_numpy(copy=True) * 100
    np.fill_diagonal(m, np.nan)
    texte = [[("" if np.isnan(v) else f"{v:.0f} %") for v in ligne] for ligne in m]
    fig = go.Figure(go.Heatmap(z=m, x=[NOMS[p] for p in PAYS], y=[NOMS[p] for p in PAYS], colorscale=BLEUS,
                               zmin=0, zmax=60, text=texte, texttemplate="%{text}", xgap=2, ygap=2,
                               colorbar=dict(thickness=10, title=dict(text="%", side="right")),
                               hovertemplate="Quand %{y} manque d'une substance,<br>%{x} en manque aussi : %{z:.0f} %<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="si pénurie en…")
    fig.update_xaxes(title="… aussi en pénurie en", side="bottom")
    fig.add_annotation(x=NOMS["DE"], y=NOMS["NO"], text="jamais<br>observés<br>ensemble", showarrow=False, font_size=9.5, font_color=MUET)
    fig.add_annotation(x=NOMS["NO"], y=NOMS["DE"], text="jamais<br>observés<br>ensemble", showarrow=False, font_size=9.5, font_color=MUET)
    return mise_en_page(fig, 470)


@app.callback(Output("spearman", "figure"), Input("onglets", "value"))
def fig_spearman(_) -> go.Figure:
    r = SPEARMAN
    texte = [[f"{v:.2f}" for v in ligne] for ligne in r.values]
    fig = go.Figure(go.Heatmap(z=r.values, x=r.columns, y=r.index, colorscale=DIVERGENT, zmin=-1, zmax=1, xgap=2, ygap=2,
                               text=texte, texttemplate="%{text}", textfont_size=9.5,
                               colorbar=dict(thickness=10, title=dict(text="ρ", side="right")),
                               hovertemplate="%{y} × %{x}<br>ρ = %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-40)
    return mise_en_page(fig, 470)


# ── Vue 6 : fiche substance ─────────────────────────────────────────────────────────────────
@app.callback(Output("substance-badges", "children"), Output("gantt", "figure"), Output("sub-signalements", "figure"),
              Output("sub-conso", "figure"), Output("voisines", "children"), Input("substance", "value"))
def fiche(atc5: str):
    s = SUB.set_index("atc5").loc[atc5]
    badges = [html.Span(f"{s['nom']} · {atc5}", className="badge", style={"fontWeight": 650}),
              html.Span(f"Classe {s['atc1']} · {ATC1.get(s['atc1'], '')}", className="badge"),
              html.Span(f"{s['atc4']} · {s['libelle_atc4'] or ''}", className="badge")]
    if s["ue_critique"]:
        badges.append(html.Span("Liste critique de l'UE", className="badge fort"))
    if s["mitm"]:
        badges.append(html.Span("Suivie dans le panel (MITM)", className="badge"))
    badges.append(html.Span(f"Pays touchés : {s['nb_pays_touches']} / 7", className="badge"))

    e = EP[EP["atc5"] == atc5].copy()
    fin_vue = pd.Timestamp(META["dernier_mois"] + "-28")
    e["fin_aff"] = e["fin"].fillna(fin_vue).clip(upper=fin_vue)
    e["debut_aff"] = e["debut"].clip(lower=pd.Timestamp("2014-01-01"))
    e = e[e["fin_aff"] >= e["debut_aff"]]
    gantt = go.Figure()
    for p in PAYS:
        ep = e[e["pays"] == p]
        for type_, opacite in (("penurie", 0.85), ("arret_definitif", 0.35)):
            x = ep[ep["type_episode"] == type_]
            if x.empty:
                continue
            duree_ms = ((x["fin_aff"] - x["debut_aff"]).dt.total_seconds() * 1000).clip(lower=86400000 * 5)
            gantt.add_trace(go.Bar(base=x["debut_aff"], x=duree_ms, y=[NOMS[p]] * len(x), orientation="h",
                                   marker=dict(color=COULEUR[p], opacity=opacite), showlegend=False, width=0.6,
                                   customdata=np.stack([x["medicament"].fillna("").str.slice(0, 60),
                                                        x["debut"].dt.strftime("%d/%m/%Y"),
                                                        np.where(x["en_cours"], "en cours", x["fin"].dt.strftime("%d/%m/%Y")),
                                                        x["cause"]], axis=1),
                                   hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} → %{customdata[2]}<br>"
                                                 + ("Pénurie" if type_ == "penurie" else "Arrêt définitif")
                                                 + " · %{customdata[3]}<extra></extra>"))
    gantt.update_yaxes(categoryorder="array", categoryarray=[NOMS[p] for p in PAYS[::-1]])
    mise_en_page(gantt, 340, barmode="overlay", xaxis=dict(type="date", range=["2015-01-01", "2026-12-31"]))
    if e.empty:
        gantt.add_annotation(text="Aucun épisode connu pour cette substance", x=0.5, y=0.5, xref="paper", yref="paper",
                             showarrow=False, font_color=MUET)

    sg = SIG[SIG["atc5"] == atc5]
    trim = sg["date_signalement"].dt.to_period("Q").value_counts().sort_index()
    trim = trim.reindex(pd.period_range("2019Q1", "2025Q4", freq="Q"), fill_value=0)
    fig_sig = go.Figure(go.Bar(x=trim.index.to_timestamp(), y=trim.values, marker_color=COULEUR["FR"],
                               marker_line=dict(color=SURFACE, width=2), xperiod="M3", xperiodalignment="start",
                               hovertemplate="%{x|%Y} T%{customdata} : %{y} signalements<extra></extra>",
                               customdata=trim.index.quarter))
    mise_en_page(fig_sig, 280, yaxis_title="signalements", bargap=0.1)

    c = CONSO[CONSO["atc5"] == atc5].sort_values("annee")
    fig_conso = go.Figure(go.Bar(x=c["annee"], y=c["boites"], marker_color=COULEUR["FR"], marker_line=dict(color=SURFACE, width=2),
                                 hovertemplate="%{x} : %{y:,.0f} boîtes<extra></extra>"))
    mise_en_page(fig_conso, 280, yaxis_title="boîtes remboursées", separators=", ")
    if c.empty:
        fig_conso.add_annotation(text="Pas de consommation en ville connue", x=0.5, y=0.5, xref="paper", yref="paper",
                                 showarrow=False, font_color=MUET)

    v = SUB[SUB["atc4"] == s["atc4"]].sort_values("nb_signalements", ascending=False)
    df = pd.DataFrame({"Substance": v["nom"], "ATC": v["atc5"], "Signalements ANSM": v["nb_signalements"],
                       **{f"Épisodes {p}": v[f"ep_{p}"] for p in PAYS},
                       "Boîtes 2024": v["boites_2024"].map(lambda x: "" if pd.isna(x) else fmt(x))})
    voisines = dash_table.DataTable(
        data=df.to_dict("records"), columns=[{"name": k, "id": k} for k in df.columns], sort_action="native",
        page_size=10, style_table={"overflowX": "auto"},
        style_cell={"padding": "4px 8px", "textAlign": "left", "border": f"1px solid {GRILLE}"},
        style_header={"fontWeight": 650, "backgroundColor": "#f3f2ee"},
        style_data_conditional=[{"if": {"filter_query": f'{{ATC}} = "{atc5}"'}, "fontWeight": 700, "backgroundColor": "#eef4fc"}])
    return badges, gantt, fig_sig, fig_conso, voisines


# ── Vue 7 : panel et qualité ────────────────────────────────────────────────────────────────
def bandes_jeux(fig: go.Figure) -> None:
    p = PANEL.sort_values("mois")
    blocs = (p["jeu"] != p["jeu"].shift()).cumsum()
    for _, g in p.groupby(blocs):
        jeu = g["jeu"].iat[0]
        fig.add_vrect(x0=g["mois"].min() - pd.Timedelta(days=15), x1=g["mois"].max() + pd.Timedelta(days=15),
                      fillcolor=JEUX[jeu][1], line_width=0, layer="below",
                      annotation_text=JEUX[jeu][0] if jeu != "ecart" else "", annotation_position="top left",
                      annotation_font=dict(size=11, color=ENCRE2))


@app.callback(Output("panel-cibles", "figure"), Input("onglets", "value"))
def fig_panel(_) -> go.Figure:
    p = PANEL.sort_values("mois")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p["mois"], y=p["cible"] * 100, name="Signalement dans les 3 mois (toutes les substances)",
                             line=dict(color=COULEUR["FR"], width=2), customdata=p["lignes"],
                             hovertemplate="%{x|%m/%Y} : %{y:.1f} % de %{customdata} substances<extra></extra>"))
    fig.add_trace(go.Scatter(x=p["mois"], y=p["cible_nouvelle_12m"] * 100,
                             name="Nouvelle pénurie (substances calmes depuis 12 mois)", line=dict(color=COULEUR["IT"], width=2),
                             customdata=np.stack([p["calmes_12m"], p["nouvelles_12m"]], axis=1),
                             hovertemplate="%{x|%m/%Y} : %{y:.1f} % (%{customdata[1]} sur %{customdata[0]} substances calmes)<extra></extra>"))
    bandes_jeux(fig)
    return mise_en_page(fig, 380, yaxis=dict(title="part de positifs (%)", rangemode="tozero"), hovermode="x unified")


@app.callback(Output("panel-signaux", "figure"), Input("signaux-echelle", "value"))
def fig_signaux(echelle: str) -> go.Figure:
    noms = {"deja_signale": "Déjà signalée à l'ANSM (cumul publié)", "fr_affichee_3m": "Affichée en pénurie sur le site ANSM",
            "fr_autres": "Autre substance de sa classe ATC4 affichée", "it_affichee_3m": "En pénurie en Italie",
            "no_nouvelle_3m": "Nouvelle pénurie en Norvège", "se_nouvelle_3m": "Nouvelle pénurie en Suède",
            "cz_nouvelle_3m": "Nouvelle pénurie en Tchéquie", "sk_nouvelle_3m": "Nouvelle pénurie en Slovaquie",
            "ue_critique": "Sur la liste critique de l'UE"}
    s = SIGNAUX.set_index("mois")[list(noms)].T
    z = s.values * 100
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
    fig.update_layout(plot_bgcolor="#e9e8e3")
    return mise_en_page(fig, 340, plot_bgcolor="#d9d8d2")


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
