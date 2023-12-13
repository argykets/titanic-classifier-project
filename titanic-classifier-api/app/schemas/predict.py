from typing import Any, List, Optional

from pydantic import BaseModel
from classification_model.processing.validation import TitanicDataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]


class MultipleDataInputs(BaseModel):
    inputs: List[TitanicDataInputSchema]

    model_config = {
        "json_schema_extra": {
            "openapi_examples": [
                {
                    "pclass": 1,
                    "survived": 1,
                    "name": "Allen, Miss. Elisabeth Walton",
                    "sex": "female",
                    "age": 29.0,
                    "sibsp": 0,
                    "parch": 0,
                    "ticket": "113781",
                    "fare": 211.3375,
                    "cabin": "B5",
                    "embarked": "S",
                    "boat": "?",
                    "body": "?",
                    "home_dest": "St Louis, MO",
                }
            ]
        }
    }