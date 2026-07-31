# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Modele de décision sur la base d'un seuil fixe, ce qui provoque peu de variations dans les consignes
"""
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


@dataclass
class ParametresDecision:
    fenetre_heures: int = 720
    puissance_compresseur_mw: float = 70.0
    puissance_turbine_mw: float = 120.0
    capacite_stockage_mwh: float = 200.0
    rendement_charge: float = 0.85
    rendement_decharge: float = 0.88
    pertes_horaires_fraction: float = 5e-5
    energie_initiale_mwh: float = 50.0


def construire_features_decision(df: pd.DataFrame, nb_lags: int = 24) -> Tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(index=df.index)
    X["prix"] = df["prix_eur_mwh"]
    X["energie_stockee"] = df["energie_stockee_mwh"]
    X["demande"] = df["demande_mw"]

    X["heure"] = df.index.hour
    X["jour_semaine"] = df.index.dayofweek
    X["weekend"] = (df.index.dayofweek >= 5).astype(int)
    X["sin_heure"] = np.sin(2 * np.pi * X["heure"] / 24.0)
    X["cos_heure"] = np.cos(2 * np.pi * X["heure"] / 24.0)

    for k in range(1, nb_lags + 1):
        X[f"lag_prix_{k}"] = df["prix_eur_mwh"].shift(k)

    y = df["action"].astype(int)
    donnees = pd.concat([X, y.rename("cible")], axis=1).dropna()
    return donnees.drop(columns=["cible"]), donnees["cible"]


def calculer_actions_oracle_cvxpy(prix: pd.Series, demande_mw: pd.Series, param: ParametresDecision, energie_initiale_mwh: float) -> pd.DataFrame:

    try:
        import cvxpy as cp
    except ImportError as e:
        raise ImportError("Installe cvxpy pour utiliser l'oracle : pip install cvxpy") from e

    index = prix.index
    dt_heures = (index[1] - index[0]).total_seconds() / 3600.0
    n = len(index)

    achat = cp.Variable(n, nonneg=True)
    charge = cp.Variable(n, nonneg=True)
    decharge = cp.Variable(n, nonneg=True)
    energie = cp.Variable(n)

    contraintes = []

    contraintes += [charge <= param.puissance_compresseur_mw]
    contraintes += [decharge <= param.puissance_turbine_mw]

    contraintes += [energie >= 0]
    contraintes += [energie <= param.capacite_stockage_mwh]

    pertes = (1.0 - param.pertes_horaires_fraction * dt_heures)
    contraintes += [energie[0] == energie_initiale_mwh]
    for t in range(1, n):
        contraintes += [
            energie[t] == energie[t - 1] * pertes
            + charge[t] * dt_heures * param.rendement_charge
            - (decharge[t] * dt_heures) / max(param.rendement_decharge, 1e-9)
        ]

    contraintes += [achat + decharge == demande_mw.values + charge]

    # Minimise cout d'achat
    cout = cp.sum(cp.multiply(prix.values, achat) * dt_heures)
    #On ajoute les lignes suivantes si l'on souhaite pousser un modèle valorisant la charge
    # et on prend achat - decharge dans la fonction de cout au lieu d'achat tout seul
    #c_cycle = 1.0
    #cout += c_cycle * cp.sum(charge * dt_heures + decharge * dt_heures)
    #E_cible = 50.0
    #penalite_finale = 0.05
    #cout += penalite_finale * cp.square(energie[-1] - E_cible)
    #E_max_oper = 0.9 * param.capacite_stockage_mwh
    #contraintes += [energie <= E_max_oper]
    
    probleme = cp.Problem(cp.Minimize(cout), contraintes)
    probleme.solve(solver=cp.ECOS)



    df = pd.DataFrame(
        {
            "prix_eur_mwh": prix.values,
            "demande_mw": demande_mw.values,
            "achat_reseau_mw": achat.value,
            "charge_caes_mw": charge.value,
            "decharge_caes_mw": decharge.value,
            "energie_stockee_mwh": energie.value,
        }, index=index,)


    seuil = 1e-3
    action = np.zeros(n, dtype=int)
    action[df["charge_caes_mw"].values > seuil] = 1
    action[df["decharge_caes_mw"].values > seuil] = 2
    df["action"] = action
    return df


def entrainer_modele_decision(df_oracle: pd.DataFrame) -> RandomForestClassifier:
    X, y = construire_features_decision(df_oracle, nb_lags=24)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    modele = RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced_subsample", min_samples_leaf=5)
    
    modele.fit(X_train, y_train)
    pred = modele.predict(X_test)

    print(classification_report(y_test, pred, digits=3))
    return modele


def appliquer_modele_en_ligne(modele: RandomForestClassifier, df_etat: pd.DataFrame) -> pd.Series:

    X, _ = construire_features_decision(df_etat.assign(action=0), nb_lags=24)
    pred = modele.predict(X)
    return pd.Series(pred, index=X.index, name="action_predite")


if __name__ == "__main__":
    index = pd.date_range("2024-01-01", periods=24*60, freq="h", tz="Europe/Paris")  
    rng = np.random.default_rng(0)
    
    # Exemple prix
    prix = pd.Series(70 + 20*np.sin(2*np.pi*index.hour/24) + rng.normal(0, 5, len(index)), index=index)

    # Exemple demande
    demande = pd.Series(30 + 10*((index.hour>=18) & (index.hour<=23)).astype(int), index=index)

    param = ParametresDecision()
    df_oracle = calculer_actions_oracle_cvxpy(prix, demande, param, energie_initiale_mwh=50.0)

    modele = entrainer_modele_decision(df_oracle)

    actions_pred = appliquer_modele_en_ligne(modele, df_oracle[["prix_eur_mwh","energie_stockee_mwh","demande_mw","action"]].copy())
    print(actions_pred.head())
