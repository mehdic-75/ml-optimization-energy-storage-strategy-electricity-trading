# -*- coding: utf-8 -*-
"""
Pricing_Elec_v4.py
Modèle de pricing de l'électricité :
- apprentissage ML (RandomForestRegressor)
- volatilité réaliste via GARCH (arch)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from arch import arch_model


def charger_prix_electricite(path_csv: str, pays: str = "France") -> pd.Series:
    df = pd.read_csv(path_csv)

    df = df[df["Country"] == pays]

    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
    df = df.set_index("Datetime (UTC)")

    prix = df["Price (EUR/MWhe)"].sort_index()

    return prix

def construire_features(prix: pd.Series, nb_lags: int = 30):

    df = pd.DataFrame({"y": prix})

    # Lags autorégressifs
    for lag in range(1, nb_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)

    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month

    # Encodage cyclique pour ne pas avoir de cassure à minuit
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df = df.dropna()

    X = df.drop(columns="y")
    y = df["y"]

    return X, y

def entrainer_modele(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"MAE validation : {mae:.2f} €/MWh")

    return model

def ajuster_garch(prix: pd.Series):

    rendements = prix.diff().dropna()

    model_garch = arch_model(
        rendements,
        vol="Garch",
        p=1,
        q=1,
        mean="Zero",
        dist="normal"
    )

    res = model_garch.fit(disp="off")

    return res

def prevoir_prix(
    model,
    garch_res,
    prix: pd.Series,
    horizon_h: int = 8760,
    nb_lags: int = 30
):

    predictions = []

    prix_moyen = prix.mean()
    alpha_reversion = 0.05  # rappel vers la moyenne

    # Dernier état connu
    historique = prix.iloc[-nb_lags:].tolist()
    current_time = prix.index[-1]

    # Volatilité GARCH
    sigma = np.sqrt(garch_res.forecast(horizon=horizon_h).variance.values[-1])

    for h in range(horizon_h):

        current_time += pd.Timedelta(hours=1)

        # Construction features
        features = {}

        for lag in range(1, nb_lags + 1):
            features[f"lag_{lag}"] = historique[-lag]

        hour = current_time.hour
        dow = current_time.dayofweek
        month = current_time.month

        features["hour"] = hour
        features["dayofweek"] = dow
        features["month"] = month

        features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        features["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        features["dow_cos"] = np.cos(2 * np.pi * dow / 7)

        features["month_sin"] = np.sin(2 * np.pi * month / 12)
        features["month_cos"] = np.cos(2 * np.pi * month / 12)

        X_pred = pd.DataFrame([features])

        # ML + bruit
        pred_ml = model.predict(X_pred)[0]
        bruit = np.random.normal(0, sigma[h])

        pred = pred_ml + bruit

        # Mean reversion
        pred += alpha_reversion * (prix_moyen - pred)

        # Bornes économiques réalistes
        pred = np.clip(pred, 0, 500)

        predictions.append(pred)

        historique.append(pred)
        historique.pop(0)

    index_pred = pd.date_range(
        start=prix.index[-1] + pd.Timedelta(hours=1),
        periods=horizon_h,
        freq="h"
    )

    return pd.Series(predictions, index=index_pred)

def plot_prevision(prix, prevision):

    plt.figure(figsize=(14, 6))

    plt.plot(prix[-500:], label="Historique", linewidth=2)
    plt.plot(prevision[:500], label="Prévision", linestyle="--")

    plt.ylabel("Prix €/MWh")
    plt.xlabel("Temps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    path_csv = r"/Users/mehdi/Documents/Mehdi/ENSAM/PJE_BIS_ENSAM/France.csv"

    prix = charger_prix_electricite(path_csv)

    X, y = construire_features(prix)

    model = entrainer_modele(X, y)

    garch_res = ajuster_garch(prix)

    prevision = prevoir_prix(
        model=model,
        garch_res=garch_res,
        prix=prix,
        horizon_h=8760
    )

    plot_prevision(prix, prevision)
