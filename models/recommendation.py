import logging
from typing import Union
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from utils import clean_data, read_json_data


logger = logging.getLogger(__name__)


class RecommendationModel:

    def __init__(self, data: Union[str, pd.DataFrame] = None) -> None:
        self._df = data if isinstance(data, pd.DataFrame) else read_json_data(data)
        self.original_df = self._df.copy()
        self._df = clean_data(self._df, logger)

        # Ensure text fields are non-null
        self.text_columns = [
            "features",
            "amenities",
            "description",
            "comments",
            "neighborhood",
            "rules",
        ]
        for col in self.text_columns:
            self._df[col] = self._df[col].apply(
                lambda x: (
                    " ".join(x)
                    if isinstance(x, list)
                    else (x if isinstance(x, str) else "")
                )
            )

    def preprocess(self):
        # Numerical features preprocessing
        numeric_features = [
            "price",
            "size",
            "bathrooms",
            "bedrooms",
            "year_built",
            "rating",
        ]
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        # One-hot encoding for categorical data
        categorical_features = ["rentType", "location"]
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # TF-IDF Vectorizer for text features
        text_transformers = [
            ("features_tfidf", TfidfVectorizer(), "features"),
            ("amenities_tfidf", TfidfVectorizer(), "amenities"),
            ("description_tfidf", TfidfVectorizer(), "description"),
            ("comments_tfidf", TfidfVectorizer(), "comments"),
            ("neighborhood_tfidf", TfidfVectorizer(), "neighborhood"),
            ("rules_tfidf", TfidfVectorizer(), "rules"),
        ]

        # Combine all feature transformations
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
                *text_transformers,  # Unpack the list of text transformers
            ]
        )

    def train_model(self):
        # Fit and transform the data
        X = self.preprocessor.fit_transform(self._df)

        # Fit the NearestNeighbors model
        self.model = NearestNeighbors(n_neighbors=5, metric="cosine")
        self.model.fit(X)

    def get_recommendation(
        self,
        property_df: pd.DataFrame,
        n_recommendation: int = 3,
    ):
        # Ensure text fields are non-null in input
        for col in self.text_columns:
            if col in property_df.columns:
                property_df[col] = property_df[col].apply(
                    lambda x: (
                        " ".join(x)
                        if isinstance(x, list)
                        else (x if isinstance(x, str) else "")
                    )
                )

        # Transform the input data
        input_X = self.preprocessor.transform(property_df)

        # Find the nearest neighbors
        distances, indices = self.model.kneighbors(input_X, n_recommendation + 1)

        # Return the recommended properties without the property itself
        recommendations = self.original_df.iloc[indices[0][1:]]
        return recommendations

    def deploy(self, model_path="property_recommendation_model.pkl"):
        # Save the model and the preprocessor together
        joblib.dump(
            {"model": self.model, "preprocessor": self.preprocessor}, model_path
        )
        print(f"Model and preprocessor saved to {model_path}")

    @classmethod
    def load_model(cls, model_path="property_recommendation_model.pkl"):
        try:
            logger.info(f"Loading model from {model_path}")
            data = joblib.load(model_path)
            instance = cls(
                data=None
            )  # No need to pass data if loading a pre-trained model
            instance.model = data["model"]
            instance.preprocessor = data["preprocessor"]
            return instance
        except FileNotFoundError:
            logger.error(f"Model file '{model_path}' not found.")
            return None
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None


if __name__ == "__main__":
    rec = RecommendationModel()
    rec.preprocess()
    rec.train_model()

    original_df = rec.original_df.copy().iloc[8].to_dict()
    # Get recommendations
    rec = rec.get_recommendation(original_df, 6)
    print(rec.to_dict())
