from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from classification_model.config.core import config
from classification_model.processing.feature_transformation import get_first_cabin, get_title


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.copy()
    new_vars_with_na = [
        var
        for var in config.model_config.features
        if var
        not in config.model_config.categorical_vars_with_na_frequent
        + config.model_config.categorical_vars_with_na_missing
        + config.model_config.numerical_vars_with_na
        and validated_data[var].isnull().sum() > 0
    ]
    validated_data.dropna(subset=new_vars_with_na, inplace=True)

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    # convert syntax error field names (beginning with numbers)
    dataframe = input_data.replace('?', np.nan)
    dataframe = dataframe.replace(np.inf, np.nan)
    dataframe.fillna(0, inplace=True)
    dataframe['cabin'] = dataframe['cabin'].apply(get_first_cabin)
    dataframe['title'] = dataframe['name'].apply(get_title)
    dataframe['fare'] = dataframe['fare'].astype('float')
    dataframe['age'] = dataframe['age'].astype('float')
    validated_data = dataframe.drop(columns=['name','ticket', 'boat', 'body','home_dest'], axis=1, inplace=False)
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs=validated_data.to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors

class TitanicDataInputSchema(BaseModel):
    pclass: Optional[int] = None
    survived: int
    name: Optional[str] = None
    sex: Optional[str] = None
    age: Optional[float] = None
    sibsp: Optional[int] = None
    parch: Optional[int] = None
    ticket: Optional[str] = None
    fare: Optional[float] = None
    cabin: Optional[str] = None
    boat: Optional[str] = None
    body: Optional[str] = None
    home_dest: Optional[str] = None
    embarked: Optional[str] = None
    home_dest: Optional[str] = None

class MultipleDataInputs(BaseModel):
    inputs: List[TitanicDataInputSchema]