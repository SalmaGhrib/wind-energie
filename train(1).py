"""Regression : prediction des kWh produits en une heure par une eolienne fictive.

python train.py --train-data donnees_A/train.csv --val-data donnees_A/val.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURES = ["vitesse_vent_ms", "temperature_c", "densite_air_kg_m3", "turbulence_pct"]
TARGET = "energie_kwh"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", "--train", dest="train_data", default="data/train.csv")
    parser.add_argument("--val-data", "--validation", dest="val_data", default="data/val.csv")
    parser.add_argument("--output-dir", default="output")
    args, _ = parser.parse_known_args()

    train = pd.read_csv(args.train_data)
    val = pd.read_csv(args.val_data)
    for name, frame in (("train", train), ("validation", val)):
        missing = set(FEATURES + [TARGET]) - set(frame.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes dans {name}: {sorted(missing)}")

    model = RandomForestRegressor(n_estimators=180, min_samples_leaf=2,
                                  max_features=1.0, random_state=42, n_jobs=-1)
    model.fit(train[FEATURES], train[TARGET])
    predictions = model.predict(val[FEATURES])
    metrics = {
        "mae_kwh": round(mean_absolute_error(val[TARGET], predictions), 2),
        "rmse_kwh": round(mean_squared_error(val[TARGET], predictions) ** 0.5, 2),
        "r2": round(r2_score(val[TARGET], predictions), 4),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "model.pkl")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
