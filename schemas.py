from pydantic import BaseModel, Field
from typing import List, Optional, Union


class GPSPosition(BaseModel):
    long: float = Field(..., example=2.352222)
    lat: float = Field(..., example=48.856614)


class PropertyData(BaseModel):
    price: str = Field(..., example="$800,000")
    image_prompt: Optional[str] = Field(
        None, example="Create an image of a modern 5-bedroom house with a pool."
    )
    date: Optional[int] = Field(None, example=1722879007576)
    rules: Optional[Union[str, List[str]]] = Field(
        None, example="No loud noises after 11pm."
    )
    rating: Optional[float] = Field(None, example=4.7)
    features: List[str] = Field(
        ...,
        example=[
            "Pool",
            "Fully equipped kitchen",
            "Beautiful living room",
            "Close to restaurants and shops",
        ],
    )
    amenities: List[str] = Field(
        ...,
        example=[
            "Air conditioning",
            "Central heating",
            "Washer and dryer",
            "Parking",
            "24-hour security",
        ],
    )
    bathrooms: Optional[int] = Field(None, example=4)
    year_built: Optional[int] = Field(None, example=2017)
    name: str = Field(..., example="Modern 5-Bedroom House with a Pool")
    gpsPosition: GPSPosition = Field(..., example={"long": 2.352222, "lat": 48.856614})
    size: str = Field(..., example="4,000 sqft")
    rentType: Optional[str] = Field(None, example="hostel rent")
    location: str = Field(..., example="2424 Pine St, Accra, Ghana")
    image: Optional[List[str]] = Field(
        None,
        example=[
            "https://firebasestorage.googleapis.com/v0/b/accommodationlink-17176.appspot.com/o/house_images%2F11a-min.png?alt=media&token=cf7669a1-fbe4-47ec-9fb3-d6ce59e19472",
            "https://firebasestorage.googleapis.com/v0/b/accommodationlink-17176.appspot.com/o/house_images%2F11b-min.png?alt=media&token=978f28dc-d394-4dc8-8461-8c9edfa45c98",
        ],
    )
    bedrooms: Optional[int] = Field(None, example=5)
    id: Optional[str] = Field(None, example="6nHxqAMiyvcXHuXQUU2I")
    rentStatus: Optional[str] = Field(None, example="available")
    status: Optional[str] = Field(None, example="accepted")
    comments: Optional[str] = Field(
        None, example="The apartment is spacious and well-lit."
    )
    description: Optional[str] = Field(
        None,
        example="This modern 5-bedroom house is perfect for those who love to swim. The house features a pool, a fully equipped kitchen, and a beautiful living room. The house is also close to several restaurants and shops.",
    )
    neighborhood: Optional[List[str]] = Field(
        None,
        example=[
            "Close to schools",
            "Close to parks",
            "Close to hospitals",
            "Close to shopping centers",
        ],
    )

    class Config:
        json_schema_extra = {
            "example": {
                "price": "$800,000",
                "image_prompt": "Create an image of a modern 5-bedroom house with a pool.",
                "date": 1722879007576,
                "rules": "No loud noises after 11pm.",
                "rating": 4.7,
                "features": [
                    "Pool",
                    "Fully equipped kitchen",
                    "Beautiful living room",
                    "Close to restaurants and shops",
                ],
                "amenities": [
                    "Air conditioning",
                    "Central heating",
                    "Washer and dryer",
                    "Parking",
                    "24-hour security",
                ],
                "bathrooms": 4,
                "year_built": 2017,
                "name": "Modern 5-Bedroom House with a Pool",
                "gpsPosition": {"long": 2.352222, "lat": 48.856614},
                "size": "4,000 sqft",
                "rentType": "hostel rent",
                "location": "2424 Pine St, Accra, Ghana",
                "image": [
                    "https://firebasestorage.googleapis.com/v0/b/accommodationlink-17176.appspot.com/o/house_images%2F11a-min.png?alt=media&token=cf7669a1-fbe4-47ec-9fb3-d6ce59e19472",
                    "https://firebasestorage.googleapis.com/v0/b/accommodationlink-17176.appspot.com/o/house_images%2F11b-min.png?alt=media&token=978f28dc-d394-4dc8-8461-8c9edfa45c98",
                ],
                "bedrooms": 5,
                "id": "6nHxqAMiyvcXHuXQUU2I",
                "rentStatus": "available",
                "status": "accepted",
                "comments": "The apartment is spacious and well-lit.",
                "description": "This modern 5-bedroom house is perfect for those who love to swim. The house features a pool, a fully equipped kitchen, and a beautiful living room. The house is also close to several restaurants and shops.",
                "neighborhood": [
                    "Close to schools",
                    "Close to parks",
                    "Close to hospitals",
                    "Close to shopping centers",
                ],
            }
        }
