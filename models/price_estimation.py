import logging
from pathlib import Path
from typing import Any
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(filename=__name__, level=logging.INFO)


def read_data(source: str | Path):
    if type(source) is str:
        match (source.split(".")[-1]):
            case "csv":
                return pd.read_csv(source)
            case "json":
                return pd.read_json(source)
            case "xlc":
                return pd.read_excel(source)


class PriceEstimator:

    def __init__(self, data: str | pd.DataFrame, features: list[str] = None) -> None:
        self._df = data if isinstance(data, pd.DataFrame) else read_data(data)

        if self._df is None:
            return
        # Convert size and price to numeric values
        self._df["size"] = (
            self._df["size"].str.replace(",", "").str.extract(r"(\d+)").astype(float)
        )
        self._df["price"] = (
            self._df["price"].str.replace("[\$,]", "", regex=True).astype(float)
        )
        self._df["price"] = np.log1p(self._df["price"])

        # Define the features for price prediction
        self.features = (
            [
                "rating",
                "bathrooms",
                "year_built",
                "bedrooms",
                "size",
                "latitude",
                "longitude",
                "rentType",
            ]
            if features is None
            else features
        )

    def preprocess(
        self, df: pd.DataFrame = None, transformers: list[tuple[str, Any, list]] = None
    ):
        _df = self._df if df is None else df

        # Extract latitude and longitude
        _df["latitude"] = _df["gpsPosition"].apply(lambda x: x["lat"])
        _df["longitude"] = _df["gpsPosition"].apply(lambda x: x["long"])

        _transformers = (
            [
                (
                    "num",
                    StandardScaler(),
                    [
                        "rating",
                        "bathrooms",
                        "year_built",
                        "bedrooms",
                        "size",
                        "latitude",
                        "longitude",
                    ],
                ),
                ("cat", OneHotEncoder(), ["rentType"]),
            ]
            if transformers is None
            else transformers
        )

        # Define the preprocessor
        self.preprocessor = ColumnTransformer(_transformers)

        return self.preprocessor

    def train_model(self, df: pd.DataFrame = None, test_size=0.2, random_state=42):
        _df = self._df if df is None else df

        # Split the data into training and testing sets
        X, y = _df[self.features], _df["price"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Create a pipeline with the preprocessor and a RandomForestRegressor
        self.price_pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("regressor", RandomForestRegressor(random_state=42)),
            ]
        )

        # Train the model
        self.price_pipeline.fit(self.X_train, self.y_train)

    def estimate(self, data):

        # Predict on the provided data
        y_pred = self.price_pipeline.predict(data)
        return y_pred

    def get_performance(self, pred, y_test=None):
        y_test = y_test if y_test is not None else self.y_test
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        return mse, r2

    def feature_importances(self):
        importances = self.price_pipeline.named_steps["regressor"].feature_importances_
        feature_names = self.preprocessor.transformers_[0][2] + list(
            self.preprocessor.named_transformers_["cat"].get_feature_names_out()
        )
        feature_importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importances}
        )
        return feature_importance_df.sort_values(by="Importance", ascending=False)

    def tune_model(self):
        param_grid = {
            "regressor__n_estimators": [100, 200, 300],
            "regressor__max_depth": [None, 10, 20, 30],
            "regressor__min_samples_split": [2, 5, 10],
        }
        grid_search = GridSearchCV(
            self.price_pipeline, param_grid, cv=5, n_jobs=-1, verbose=1
        )
        grid_search.fit(self.X_train, self.y_train)
        self.price_pipeline = grid_search.best_estimator_
        print(f"Best Parameters: {grid_search.best_params_}")

    def cross_validate(self, cv=5):
        scores = cross_val_score(
            self.price_pipeline,
            self.X_train,
            self.y_train,
            cv=cv,
            scoring="neg_mean_squared_error",
        )
        return np.sqrt(-scores).mean()

    def deploy(self, model_path="price_estimation_model.pkl"):
        joblib.dump(self.price_pipeline, model_path)
        print(f"Model saved to {model_path}")

    @classmethod
    def load_model(cls, model_path="price_estimation_model.pkl"):
        try:
            logging.info(f"Loading model from {model_path}")
            loaded_pipeline = joblib.load(model_path)
            instance = cls(
                data=None
            )  # No need to pass data if loading a pre-trained model
            instance.price_pipeline = loaded_pipeline
            return instance
        except FileNotFoundError:
            logging.error(f"Model file '{model_path}' not found.")
            return None
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return None
