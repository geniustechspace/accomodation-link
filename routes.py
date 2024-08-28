import logging
from typing import List, Optional, Union
from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd
from pydantic import BaseModel
from models import PriceEstimator, RecommendationModel
from schemas import PropertyData
from utils import clean_data


logging.basicConfig(filename=__name__, level=logging.INFO)

router = APIRouter()


class gpsPositionModel(BaseModel):
    lat: float
    long: float


class PriceEstimationRequest(BaseModel):
    rating: float
    bathrooms: int
    year_built: int
    bedrooms: int
    gpsPosition: gpsPositionModel
    size: float
    rentType: str

    class Config:
        json_schema_extra = {
            "example": {
                "rating": 3.8,
                "bathrooms": 1,
                "year_built": 2020,
                "bedrooms": 1,
                "gpsPosition": {"lat": 0, "long": 0},
                "size": 4000,
                "rentType": "hostel rent",
            }
        }


class PriceEstimationResponse(BaseModel):
    estimated_price: float


@router.post("/estimate-price", response_model=PriceEstimationResponse)
async def estimate_price(request: Union[PriceEstimationRequest, PropertyData]):
    try:
        loaded_model = PriceEstimator.load_model("price_estimation_model.pkl")
    except FileNotFoundError:
        logging.error("Model file not found.")
        raise HTTPException(status_code=500, detail="Model file not found.")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    try:
        new_data = pd.DataFrame([request.model_dump()])
        # Clean data
        new_data = clean_data(new_data, logging)
        logging.info(f"New data for prediction: {new_data}")
        estimated_price = loaded_model.estimate(new_data)[0]
        estimated_price = round(np.expm1(estimated_price), 2)
        return PriceEstimationResponse(estimated_price=estimated_price)
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class RecommendationRequest(BaseModel):
    property_data: PropertyData
    number_of_recommendations: int = 3


class RecommendationResponse(BaseModel):
    recommendations: Optional[List[PropertyData]]


@router.post("/recommendations", response_model=RecommendationResponse)
async def recommendations(request: RecommendationRequest):
    try:
        # Load the trained model
        loaded_model = RecommendationModel.load_model(
            "property_recommendation_model.pkl"
        )
        if not loaded_model:
            raise HTTPException(status_code=500, detail="Model could not be loaded.")
    except FileNotFoundError:
        logging.error("Model file not found.")
        raise HTTPException(status_code=500, detail="Model file not found.")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    try:
        data = request.property_data.model_dump()
        logging.info(f"New data for recommendations: {data}")
        property_df = pd.DataFrame([data])
        property_df = clean_data(property_df, logging)
        recommendations = loaded_model.get_recommendation(
            property_df, request.number_of_recommendations
        )

        if recommendations is not None:
            recommendations = recommendations.replace(
                [np.nan, np.inf, -np.inf], value=None
            )
            response_data = recommendations.to_dict(orient="records")
            return RecommendationResponse(recommendations=response_data)

        return RecommendationResponse(recommendations=[])

    except Exception as e:
        logging.error(f"Error during recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
