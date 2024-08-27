

from fastapi import FastAPI
import uvicorn

from models import deploy_price_estimator_model
import routes


def start_app():
    app = FastAPI(
        title="ACCOMODATION LINK",
        description="PROPERTY RECOMMENDATION AND PRICE ESTIMATION BACKEND SERVICE - WORKS WITH BOTH IMAGES AND VIDEOS",
        version="0.1.0",
    )

    app.include_router(routes.router)

    return app


app = start_app()


if __name__ == "__main__":
    deploy_price_estimator_model("./data/properties.json")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        # workers=4,
        reload=True,
        # log_config=LOGGING_CONFIG,
        # log_level="info",
        use_colors=True,
    )
