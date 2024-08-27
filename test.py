from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

from models.price_estimation import PriceEstimator


def create_dummy_model():
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("regressor", RandomForestRegressor())]
    )
    joblib.dump(pipeline, "dummy_model.pkl")


def test_load_model():
    create_dummy_model()
    model = PriceEstimator.load_model("dummy_model.pkl")
    if model:
        print("Model loaded successfully.")
    else:
        print("Model loading failed.")


test_load_model()
