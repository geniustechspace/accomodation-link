from models.price_estimation import PriceEstimator
from models.recommendation import RecommendationModel
from utils import read_json_data

__df = read_json_data()


def get_df():
    return __df.copy()


def deploy_price_estimator_model():
    print("===============================================")
    print("deploying price_estimator")

    df = get_df()

    price_estimator = PriceEstimator(df)
    price_estimator.preprocess()
    price_estimator.train_model()

    # Optional: Tune the model
    price_estimator.tune_model()

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


def deploy_property_recommendation_model():
    print("===============================================")
    print("deploying property_recommendation_model")

    df = get_df()

    pr_model = RecommendationModel(df)
    pr_model.preprocess()
    pr_model.train_model()

    # Save the trained model
    pr_model.deploy("property_recommendation_model.pkl")
    print(
        (
            "property_recommendation_model deployed at => 'property_recommendation_model.pkl'"
        )
    )
    print("===============================================")

    return pr_model
