import logging
from typing import Any, Union
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

from utils import clean_data, read_json_data

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class PriceEstimator:
    def __init__(
        self,
        data: Union[str, list[dict], pd.DataFrame] = None,
        features: list[str] = None,
        regressor=None,
    ) -> None:
        if data is None:
            logger.warning("No data provided, model will not be initialized.")
            return

        self._df = data if isinstance(data, pd.DataFrame) else read_json_data(data)
        self._df = clean_data(self._df, logger)

        # Feature engineering
        self._df["price"] = (
            self._df["price"].replace({"$": "", ",": ""}, regex=True).astype(float)
        )
        self._df["size"] = (
            self._df["size"].replace({" sqft": "", ",": ""}, regex=True).astype(float)
        )
        self._df["date"] = pd.to_datetime(self._df["date"], unit="ms")
        self._df["year_listed"] = self._df["date"].dt.year

        self._df["latitude"] = self._df["gpsPosition"].apply(lambda x: x["lat"])
        self._df["longitude"] = self._df["gpsPosition"].apply(lambda x: x["long"])

        self.features = features or [
            "rating",
            "bathrooms",
            "year_built",
            "bedrooms",
            "size",
            "latitude",
            "longitude",
            "rentType",
            "year_listed",
        ]
        self.regressor = regressor or RandomForestRegressor(random_state=3)
        logger.info(f"Features set to: {self.features}")

    def preprocess(
        self, transformers: list[tuple[str, Any, list]] = None
    ) -> ColumnTransformer:
        _transformers = transformers or [
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
                    "year_listed",
                ],
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["rentType"]),
        ]
        self.preprocessor = ColumnTransformer(_transformers)
        logger.info("Preprocessing pipeline created.")
        return self.preprocessor

    def validate_future_year(self, future_year: int) -> bool:
        """Validate if future_year is reasonable (e.g., within 50 years from now)."""
        current_year = pd.Timestamp.now().year
        if future_year < current_year or future_year > current_year + 50:
            logger.warning(f"Future year {future_year} is out of valid range.")
            return False
        return True

    def train_model(
        self, test_size: float = 0.25, random_state: int = 2, future_year: int = None
    ) -> None:
        try:
            X, y = self._df[self.features], self._df["price"]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            logger.info(
                f"Data split into train and test sets with test size {test_size}"
            )

            self.preprocess()

            # Create two pipelines: one for current price and one for future price
            self.current_pipeline = Pipeline(
                steps=[
                    ("preprocessor", self.preprocessor),
                    ("regressor", self.regressor),
                ]
            )

            self.future_pipeline = Pipeline(
                steps=[
                    ("preprocessor", self.preprocessor),
                    ("regressor", self.regressor),
                ]
            )

            logger.info("Training current price prediction pipeline...")
            self.current_pipeline.fit(self.X_train, self.y_train)
            logger.info("Current price model training completed.")

            if future_year is not None and self.validate_future_year(future_year):
                self.X_train["year_listed"] = future_year
                self.X_test["year_listed"] = future_year

                logger.info(
                    f"Training future price prediction pipeline for {future_year}..."
                )
                self.future_pipeline.fit(self.X_train, self.y_train)
                logger.info(
                    f"Future price model training completed for year {future_year}."
                )
            else:
                logger.warning(
                    "Future price model training skipped due to invalid future year."
                )

        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise

    def estimate(
        self, data: pd.DataFrame, future: bool = False, future_year: int = None
    ) -> np.ndarray:
        try:
            if future:
                if future_year is None or not self.validate_future_year(future_year):
                    raise ValueError("Invalid future year provided.")
                data["year_listed"] = future_year
                pred = self.future_pipeline.predict(data)
                logger.info(
                    f"Future price estimation for year {future_year} completed."
                )
            else:
                pred = self.current_pipeline.predict(data)
                logger.info("Current price estimation completed.")
            return pred
        except Exception as e:
            logger.error(f"Error during price estimation: {str(e)}")
            raise

    def get_performance(
        self, pred: np.ndarray, y_test: pd.Series = None, future: bool = False
    ) -> tuple:
        """Returns the performance of the model."""
        y_test = y_test or self.y_test
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        logger.info(
            f"Model performance {'(Future)' if future else ''} - MSE: {mse}, R2: {r2}"
        )
        return mse, r2

    def feature_importance(self, future: bool = False) -> pd.DataFrame:
        try:
            pipeline = self.future_pipeline if future else self.current_pipeline
            importances = pipeline.named_steps["regressor"].feature_importances_
            numeric_features = self.preprocessor.transformers_[0][2]
            categorical_features = list(
                self.preprocessor.named_transformers_["cat"].get_feature_names_out()
            )
            feature_names = numeric_features + categorical_features

            feature_importance_df = pd.DataFrame(
                {"Feature": feature_names, "Importance": importances}
            )
            logger.info("Feature importance retrieved.")
            return feature_importance_df.sort_values(by="Importance", ascending=False)
        except Exception as e:
            logger.error(f"Error retrieving feature importances: {str(e)}")
            raise

    def deploy(self, model_dir: str = "model_deployments") -> None:
        """Saves both the current and future trained models to disk."""
        try:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            current_model_path = os.path.join(model_dir, "current_price_model.pkl")
            future_model_path = os.path.join(model_dir, "future_price_model.pkl")

            # Save the current price model
            current_metadata = {
                "features": self.features,
                "model_type": type(self.regressor).__name__,
                "model_version": "1.0",
            }
            joblib.dump(
                {"pipeline": self.current_pipeline, "metadata": current_metadata},
                current_model_path,
            )
            logger.info(f"Current price model saved to {current_model_path}.")

            # Save the future price model if available
            if hasattr(self, "future_pipeline"):
                future_metadata = {
                    "features": self.features,
                    "model_type": type(self.regressor).__name__,
                    "model_version": "1.0",
                }
                joblib.dump(
                    {"pipeline": self.future_pipeline, "metadata": future_metadata},
                    future_model_path,
                )
                logger.info(f"Future price model saved to {future_model_path}.")
            else:
                logger.warning("Future price model was not trained and won't be saved.")

        except Exception as e:
            logger.error(f"Error during model deployment: {str(e)}")
            raise

    @classmethod
    def load_model(cls, model_dir: str = "model_deployments") -> "PriceEstimator":
        """Loads both current and future models from disk if available."""
        try:
            current_model_path = os.path.join(model_dir, "current_price_model.pkl")
            future_model_path = os.path.join(model_dir, "future_price_model.pkl")

            logger.info(f"Loading current price model from {current_model_path}")
            current_data = joblib.load(current_model_path)
            instance = cls()  # No need to pass data if loading a pre-trained model
            instance.current_pipeline = current_data["pipeline"]
            instance.features = current_data["metadata"]["features"]

            if os.path.exists(future_model_path):
                logger.info(f"Loading future price model from {future_model_path}")
                future_data = joblib.load(future_model_path)
                instance.future_pipeline = future_data["pipeline"]

            return instance
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
