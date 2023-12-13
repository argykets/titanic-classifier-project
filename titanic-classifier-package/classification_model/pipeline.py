from feature_engine.encoding import OneHotEncoder, RareLabelEncoder
from feature_engine.imputation import AddMissingIndicator, CategoricalImputer, MeanMedianImputer
from sklearn.pipeline import Pipeline
from processing.features import ExtractLetterTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from classification_model.config.core import config


# set up the pipeline
titanic_pipe = Pipeline([

    # ===== IMPUTATION =====
    # impute categorical variables with string missing
    ('categorical_imputation', CategoricalImputer(
        imputation_method='missing', variables=config.app_config.categorical_variables)),

    # add missing indicator to numerical variables
    ('missing_indicator', AddMissingIndicator(variables=config.app_config.numerical_variables)),

    # impute numerical variables with the median
    ('median_imputation', MeanMedianImputer(
        imputation_method='median', variables=config.app_config.numerical_variables)),


    # Extract letter from cabin
    ('extract_letter', ExtractLetterTransformer(variables=config.app_config.cabin)),


    # == CATEGORICAL ENCODING ======
    # remove categories present in less than 5% of the observations (0.05)
    # group them in one category called 'Rare'
    ('rare_label_encoder', RareLabelEncoder(
        tol=config.app_config.tol,
        n_categories=config.app_config.n_categories,
        variables=config.app_config.categorical_variables)),


    # encode categorical variables using one hot encoding into k-1 variables
    ('categorical_encoder', OneHotEncoder(
        drop_last=True, variables=config.app_config.categorical_variables)),

    # scale
    ('scaler', StandardScaler()),

    ('Logit', LogisticRegression(C=config.app_config.C, random_state=config.app_config.random_state)),
])