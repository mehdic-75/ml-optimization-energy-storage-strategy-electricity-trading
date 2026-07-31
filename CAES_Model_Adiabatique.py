# -*- coding: utf-8 -*-

from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class ParametresCAESAdiabatique:
    volume_caverne_m3: float = 5e5
    pression_min_bar: float = 1.0
    pression_max_bar: float = 70.0
    temperature_ambiante_K: float = 293.15
    gamma: float = 1.4
    R: float = 287.0
    fenetre_heures: int = 720
    puissance_compresseur_mw: float = 70.0
    puissance_turbine_mw: float = 120.0
    capacite_stockage_mwh: float = 200.0
    rendement_charge: float = 0.85
    rendement_decharge: float = 0.88
    pertes_horaires: float = 1e-4
    energie_initiale_mwh: float = 50.0


def simuler_caes_adiabatique(index, consigne_MW, p: ParametresCAESAdiabatique):

    dt_h = (index[1] - index[0]).total_seconds() / 3600  

    n = len(index)

    P = np.zeros(n)  
    T = np.zeros(n)  
    m = np.zeros(n)  

    energie = np.zeros(n)  

    P[0] = p.pression_min_bar * 1e5
    T[0] = p.temperature_ambiante_K
    m[0] = P[0] * p.volume_caverne_m3 / (p.R * T[0])
    energie[0] = p.energie_initiale_mwh

    W_comp = np.zeros(n)
    W_turb = np.zeros(n)
    W_reseau = np.zeros(n)

    for t in range(1, n):

   
        P[t], T[t], m[t], energie[t] = P[t-1], T[t-1], m[t-1], energie[t-1]
        P[t] *= (1 - p.pertes_horaires * dt_h)

        consigne = consigne_MW.iloc[t]

        #Compression
        if consigne > 0:

            P_elec = min(consigne, p.puissance_compresseur_mw)
            P_elec_eff = P_elec * p.rendement_charge

            energie[t] += P_elec_eff * dt_h
            energie[t] = min(energie[t], p.capacite_stockage_mwh)

            W_comp[t] = P_elec
            W_reseau[t] = +P_elec

        # Détente
        elif consigne < 0:

            P_elec = min(-consigne, p.puissance_turbine_mw)

            P_elec_eff = P_elec / p.rendement_decharge

            energie[t] -= P_elec_eff * dt_h
            energie[t] = max(energie[t], 0)

            W_turb[t] = P_elec
            W_reseau[t] = -P_elec

        # Mise à jour pression fictive 
        taux = energie[t] / p.capacite_stockage_mwh
        P[t] = (p.pression_min_bar + taux * (p.pression_max_bar - p.pression_min_bar)) * 1e5

    return pd.DataFrame(
        {
            "pression_bar": P / 1e5,
            "temperature_C": T - 273.15,
            "masse_air_kg": m,
            "energie_stockee_mwh": energie,  
            "P_comp_MW": W_comp,
            "P_turb_MW": W_turb,
            "P_reseau_MW": W_reseau,
        },
        index=index,
    )
