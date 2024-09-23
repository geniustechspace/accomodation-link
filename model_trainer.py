from models import deploy_price_estimator_model, deploy_property_recommendation_model


if __name__ == "__main__":
    # Deploy models before starting the server
    deploy_price_estimator_model()
    deploy_property_recommendation_model()
