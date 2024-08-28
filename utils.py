import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


def read_json_data(data_path: str | Path = "./data/properties.json"):
    if data_path is None:
        data_path = "./data/properties.json"

    with open(data_path, "r") as file:
        json_data = json.load(file)

    # Extracting the list of records
    records = json_data["data"]

    # Creating a DataFrame
    df = pd.DataFrame(records)
    # print(df.head())
    return df


def clean_data(df: pd.DataFrame, logging: logging):
    # Convert size and price to numeric values

    df_columns = df.columns

    # Handle missing fields and clean data
    try:
        if "size" in df_columns and isinstance(df["size"], str):
            df["size"] = (
                df["size"].str.replace(",", "").str.extract(r"(\d+)").astype(float)
            )

        if "price" in df_columns and isinstance(df["price"], str):
            df["price"] = df["price"].str.replace("[\$,]", "", regex=True).astype(float)
            df["price"] = np.log1p(df["price"])

        # Extract latitude and longitude
        if "gpsPosition" in df_columns:
            df["latitude"] = df["gpsPosition"].apply(lambda x: x["lat"])
            df["longitude"] = df["gpsPosition"].apply(lambda x: x["long"])

        if "bathrooms" in df_columns:
            df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors="coerce")

        if "bedrooms" in df_columns:
            df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce")

        if "year_built" in df_columns:
            df["year_built"] = pd.to_numeric(df["year_built"], errors="coerce")

        if "rating" in df_columns:
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

        if "rules" in df_columns:
            df["rules"] = [df["rules"]] if isinstance(df["rules"], str) else df["rules"]

        return df

    except Exception as e:
        logging.error(f"Error cleaning input data: {str(e)}")
        return None
