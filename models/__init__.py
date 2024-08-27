import json
from pathlib import Path

import pandas as pd

from models.price_estimation import PriceEstimator


def deploy_price_estimator_model(data_path: str | Path):
    print("===============================================")
    print("deploying price_estimator")
    with open(data_path, "r") as file:
        json_data = json.load(file)

    # Extracting the list of records
    records = json_data["data"]

    # Creating a DataFrame
    df = pd.DataFrame(records)

    price_estimator = PriceEstimator(df)
    price_estimator.preprocess()
    price_estimator.train_model()

    # Optional: Tune the model
    price_estimator.tune_model()

    print(price_estimator.X_test)

    # Evaluate the model
    y_pred = price_estimator.estimate(price_estimator.X_test)
    print("Estimation: ", y_pred)
    mse, r2 = price_estimator.get_performance(y_pred)
    print(f"MSE: {mse}, R²: {r2}")

    # Save the trained model
    price_estimator.deploy("price_estimation_model.pkl")
    print(("price_estimator deployed at => 'price_estimation_model.pkl'"))
    print("===============================================")

    return price_estimator
