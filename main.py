import logging
from fastapi import FastAPI
import uvicorn

from models import deploy_price_estimator_model, deploy_property_recommendation_model
import routes

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to console
        # logging.FileHandler("app.log"),  # Log to a file
    ],
)

logger = logging.getLogger(__name__)


def start_app():
    app = FastAPI(
        title="ACCOMMODATION LINK",
        description="PROPERTY RECOMMENDATION AND PRICE ESTIMATION BACKEND SERVICE - WORKS WITH BOTH IMAGES AND VIDEOS",
        version="0.1.0",
    )

    app.include_router(routes.router)

    return app


app = start_app()

if __name__ == "__main__":
    logger.info("Starting the application...")

    # Deploy models before starting the server
    deploy_price_estimator_model()
    deploy_property_recommendation_model()

    # Start Uvicorn server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
        use_colors=True,
    )
