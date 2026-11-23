"""Figure : période couverte par chaque source, photos d'archive, et fenêtre du panel (entraînement / validation / test).

Lit dashboard/donnees/ (tables résumées versionnées) ; écrit livrables/avancement/figures/couverture.pdf.
Lancer depuis la racine : .venv/bin/python dataset/analyse/figure_couverture.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RACINE = Path(__file__).resolve().parents[2]
META = json.loads((RACINE / "dashboard/donnees/meta.json").read_text())
PHOTOS = pd.read_parquet(RACINE / "dashboard/donnees/photos.parquet")
fin_defaut = pd.Timestamp(META["dernier_mois"] + "-01") + pd.offsets.MonthEnd(0)

LIGNES = [  # (étiquette, pays pour les photos, début, fin, couleur)
    ("France, page ANSM (116 photos)", "FR", META["couverture"]["FR"], None, "#2a78d6"),
    ("France, signalements ANSM (cible)", None, "2019-01", "2025-12", "#0d366b"),
    ("Italie, AIFA (54 photos)", "IT", META["couverture"]["IT"], None, "#eb6834"),
    ("Allemagne, BfArM (13 photos)", "DE", META["couverture"]["DE"], None, "#1baf7a"),
    ("Norvège, DMP (6 archives)", "NO", META["couverture"]["NO"], META["fin_couverture"]["NO"], "#eda100"),
    ("Suède, Läkemedelsverket", None, META["couverture"]["SE"], None, "#e87ba4"),
    ("Tchéquie, SÚKL", None, META["couverture"]["CZ"], None, "#008300"),
    ("Slovaquie, ŠÚKL", None, META["couverture"]["SK"], None, "#6250d6"),
]
JEUX = [("entraînement", "2020-04", "2023-09", "#2a78d6"), ("val.", "2024-01", "2024-09", "#eb6834"),
        ("test", "2025-01", "2025-09", "#1baf7a")]

fig, ax = plt.subplots(figsize=(7.2, 2.7))
for nom, d, f, c in JEUX:
    ax.axvspan(pd.Timestamp(d + "-01"), pd.Timestamp(f + "-01") + pd.offsets.MonthEnd(0), color=c, alpha=0.10, lw=0)
    ax.text(pd.Timestamp(d + "-01") + (pd.Timestamp(f + "-01") - pd.Timestamp(d + "-01")) / 2, -0.95, nom,
            ha="center", va="bottom", fontsize=6.5, color=c)
for i, (etiq, pays, d, f, c) in enumerate(LIGNES):
    debut = pd.Timestamp(d + "-01")
    fin = pd.Timestamp(f + "-01") + pd.offsets.MonthEnd(0) if f else fin_defaut
    ax.plot([debut, fin], [i, i], color=c, lw=7, alpha=0.35, solid_capstyle="butt")
    if pays:
        dates = PHOTOS.loc[PHOTOS["pays"] == pays, "date_photo"].drop_duplicates()
        ax.plot(dates, [i] * len(dates), "|", color=c, ms=8, mew=1.1)
ax.axvline(pd.Timestamp("2021-04-05"), color="#898781", lw=0.8, ls=":")
ax.text(pd.Timestamp("2021-03-01"), -0.6, "nouveau site ANSM", fontsize=6.5, color="#898781", va="center", ha="right")
ax.set_yticks(range(len(LIGNES)), [l[0] for l in LIGNES], fontsize=7)
ax.invert_yaxis()
ax.set_ylim(len(LIGNES) - 0.4, -1.05)
ax.set_xlim(pd.Timestamp("2015-01-01"), pd.Timestamp("2026-12-31"))
ax.tick_params(axis="x", labelsize=7)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
plt.tight_layout(pad=0.3)
sortie = RACINE / "livrables/avancement/figures/couverture.pdf"
fig.savefig(sortie)
print(sortie)
