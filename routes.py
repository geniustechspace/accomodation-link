import logging
from typing import List
from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd
from pydantic import BaseModel
from models import PriceEstimator


logging.basicConfig(filename=__name__, level=logging.INFO)

router = APIRouter()


class PriceEstimationRequest(BaseModel):
    rating: float
    bathrooms: int
    year_built: int
    bedrooms: int
    size: float
    latitude: float
    longitude: float
    rentType: str


class PriceEstimationResponse(BaseModel):
    predicted_price: List[float]


@router.post("/estimate-price", response_model=PriceEstimationResponse)
async def estimate_price(data: PriceEstimationRequest):
    try:
        loaded_model = PriceEstimator.load_model("price_estimation_model.pkl")
    except FileNotFoundError:
        logging.error("Model file not found.")
        raise HTTPException(status_code=500, detail="Model file not found.")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    try:
        new_data = pd.DataFrame([data.model_dump()])
        logging.info(f"New data for prediction: {new_data}")
        predicted_price = loaded_model.estimate(new_data)
        predicted_price_list = np.expm1(predicted_price).tolist()
        return PriceEstimationResponse(predicted_price=predicted_price_list)
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommendations")
async def recommendations():
    pass
