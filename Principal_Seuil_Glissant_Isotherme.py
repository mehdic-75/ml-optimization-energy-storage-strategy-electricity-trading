# -*- coding: utf-8 -*-

"""
Pilotage principal CAES
- Pricing ML + GARCH (Pricing_Elec_v4)
- Décision par seuil glissant
- Simulation CAES
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Pricing_Elec_v4 import (
    charger_prix_electricite,
    construire_features,
    entrainer_modele,
    ajuster_garch,
    prevoir_prix
)

from CAES_Model_Isotherme import ParametresCAESIsotherme, simuler_caes_isotherme
from Modele_Decision_Seuil_Glissant import (
    ParametresDecisionBis,
    decisions_par_seuil_glissant
)


# Paramètres
DATE_DEBUT = "2022-01-01"
DATE_FIN   = "2026-01-01"

HORIZON_PREVISION_JOURS = 30
HORIZON_PREVISION_HEURES = HORIZON_PREVISION_JOURS * 24

PATH_PRIX = r"C:\Users\mehdi\Documents\Mehdi\ENSAM\PJE_BIS_ENSAM\France.csv"

prix_hist = charger_prix_electricite(PATH_PRIX, pays="France")

prix_hist = prix_hist.loc[DATE_DEBUT:DATE_FIN]
prix_hist = prix_hist.asfreq("H")

# Sécurité : suppression trous éventuels
prix_hist = prix_hist.interpolate()

print(f"Historique chargé : {len(prix_hist)} heures")

# Entrainement pricing

X, y = construire_features(prix_hist)

model_prix = entrainer_modele(X, y)

garch_res = ajuster_garch(prix_hist)



# Prévision prix

prix_prevu = prevoir_prix(
    model=model_prix,
    garch_res=garch_res,
    prix=prix_hist,
    horizon_h=HORIZON_PREVISION_HEURES
)



# Modèle décision

param_decision = ParametresDecisionBis(
    fenetre_heures=720,               # 30 jours glissants
    capacite_stockage_mwh=200.0,
    puissance_compresseur_mw=70.0,
    puissance_turbine_mw=120.0,
    energie_initiale_mwh=50.0,
)

df_decision = decisions_par_seuil_glissant(
    prix_prevu,
    param_decision
)

consigne_caes = (
    df_decision["charge_caes_mw"]
    - df_decision["decharge_caes_mw"]
)



# Simulation CAES

param_caes = ParametresCAESIsotherme()

df_caes = simuler_caes_isotherme(
    index=prix_prevu.index,
    consigne_MW=consigne_caes,
    p=param_caes
)



# Visualisation

plt.figure(figsize=(14, 5))
plt.plot(prix_prevu, label="Prix prévisionnel")
plt.title("Prévision du prix de l'électricité (ML + GARCH)")
plt.ylabel("€/MWh")
plt.grid()
plt.legend()
plt.show()


plt.figure(figsize=(14, 5))
plt.plot(df_caes.index, df_caes["energie_stockee_mwh"], label="Énergie stockée")
plt.title("État de charge du CAES")
plt.ylabel("MWh")
plt.grid()
plt.legend()
plt.show()


plt.figure(figsize=(14, 5))
plt.plot(prix_prevu, label="Prix")
plt.plot(consigne_caes, label="Consigne CAES (MW)")
plt.title("Stratégie de pilotage CAES")
plt.grid()
plt.legend()
plt.show()
