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


def clean_data(df: pd.DataFrame, logging: logging) -> pd.DataFrame:
    """
    Cleans the DataFrame by handling missing values and formatting columns.

    :param df: Raw pandas DataFrame.
    :param logger: Logger object for logging.
    :return: Cleaned pandas DataFrame.
    """
    try:
        # Convert size to numeric by extracting digits and converting to float
        if "size" in df.columns:
            df["size"] = df["size"].apply(
                lambda x: (
                    float(x.replace(",", "").split()[0]) if isinstance(x, str) else x
                )
            )

        # Convert price to numeric by removing symbols and converting to float
        if "price" in df.columns:
            df["price"] = df["price"].str.replace("[$,]", "", regex=True).astype(float)
            df["price"] = np.log1p(df["price"])

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], unit="ms")
            df["year_listed"] = df["date"].dt.year

        # Extract latitude and longitude from gpsPosition
        if "gpsPosition" in df.columns:
            df["latitude"] = df["gpsPosition"].apply(
                lambda x: x["lat"] if isinstance(x, dict) else np.nan
            )
            df["longitude"] = df["gpsPosition"].apply(
                lambda x: x["long"] if isinstance(x, dict) else np.nan
            )

        # Convert other numerical fields to numeric
        for col in ["bathrooms", "bedrooms", "year_built", "rating"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Convert rules to list if it's a string
        if "rules" in df.columns and isinstance(df["rules"], str):
            df["rules"] = [df["rules"]]

        return df

    except Exception as e:
        logging.error(f"Error cleaning input data: {str(e)}")
        return None


# def clean_data(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
#     """
#     Cleans the DataFrame by handling missing values and formatting columns.

#     :param df: Raw pandas DataFrame.
#     :param logger: Logger object for logging.
#     :return: Cleaned pandas DataFrame.
#     """
#     try:
#         # Drop duplicates
#         initial_shape = df.shape
#         df = df.drop_duplicates()
#         logger.info(f"Dropped duplicates: {initial_shape} -> {df.shape}")

#         # Handle missing values
#         df = df.dropna(
#             subset=[
#                 "price",
#                 "size",
#                 "year_built",
#                 "bedrooms",
#                 "bathrooms",
#                 "rating",
#                 "gpsPosition",
#                 "rentType",
#             ]
#         )
#         logger.info(
#             f"Dropped rows with missing essential values. New shape: {df.shape}"
#         )

#         # Additional cleaning steps can be added here

#         return df
#     except Exception as e:
#         logger.error(f"Error cleaning data: {str(e)}")
#         raise
