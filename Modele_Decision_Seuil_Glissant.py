# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Modele de décision sur la base d'un seuil égal à la moyenne mobile des prix, ce qui provoque beaucoup de variations dans les consignes
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


@dataclass
class ParametresDecisionBis:
    fenetre_heures: int = 720
    puissance_compresseur_mw: float = 70.0
    puissance_turbine_mw: float = 120.0
    capacite_stockage_mwh: float = 200.0
    rendement_charge: float = 0.80
    rendement_decharge: float = 0.85
    pertes_horaires_fraction: float = 0.0005
    energie_initiale_mwh: float = 50.0


def decisions_par_seuil_glissant(prix: pd.Series,
                                 param: ParametresDecisionBis) -> pd.DataFrame:
    index = prix.index
    dt_heures = (index[1] - index[0]).total_seconds() / 3600.0

    fenetre = int(param.fenetre_heures)
    seuil = prix.rolling(window=fenetre, min_periods=1).mean()

    energie = np.zeros(len(index))
    energie[0] = np.clip(param.energie_initiale_mwh, 0, param.capacite_stockage_mwh)
    charge = np.zeros(len(index))
    decharge = np.zeros(len(index))

    for t in range(1, len(index)):
        e = energie[t-1] * (1.0 - param.pertes_horaires_fraction * dt_heures)

        p = prix.iloc[t]
        p_seuil = seuil.iloc[t]

        if p < p_seuil and e < param.capacite_stockage_mwh:

            p_c = param.puissance_compresseur_mw

            e_max_add = (param.capacite_stockage_mwh - e)
            p_c = min(p_c, e_max_add / (dt_heures * param.rendement_charge))
            delta_e = p_c * dt_heures * param.rendement_charge
            e = e + delta_e
            charge[t] = p_c

        elif p > p_seuil and e > 0:

            p_d = param.puissance_turbine_mw

            e_max_prelevee = e
            p_d = min(p_d, (e_max_prelevee * param.rendement_decharge) / dt_heures)
            e_prelevee = (p_d * dt_heures) / max(param.rendement_decharge, 1e-9)
            e = e - e_prelevee
            decharge[t] = p_d


        energie[t] = np.clip(e, 0, param.capacite_stockage_mwh)

    df = pd.DataFrame(
        {
            "prix_eur_mwh": prix.values,
            "seuil_glissant": seuil.values,
            "charge_caes_mw": charge,
            "decharge_caes_mw": decharge,
            "energie_stockee_mwh": energie,
        },
        index=index,
    )
    return df
