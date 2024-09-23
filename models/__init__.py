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

    # # Optional: Tune the model
    # price_estimator.tune_model()
    # cross_validate = price_estimator.cross_validate()
    # print("cross_validate: ", cross_validate)
    feature_importance = price_estimator.feature_importance()
    print("feature_importance:", feature_importance)
    future_feature_importance = price_estimator.feature_importance(future=True)
    print("future_feature_importance:", future_feature_importance)

    # Evaluate the model
    y_pred = price_estimator.estimate(price_estimator.X_test)
    print("Estimation: ", y_pred)
    mse, r2 = price_estimator.get_performance(y_pred)
    print(f"MSE: {mse}, R²: {r2}")
    
    # Evaluate the model
    y_pred = price_estimator.estimate(price_estimator.X_test, future=True, future_year=2030)
    print("Future Estimation: ", y_pred)
    mse, r2 = price_estimator.get_performance(y_pred, future=True)
    print(f"MSE: {mse}, R²: {r2}")

    # Save the trained model
    price_estimator.deploy()
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
