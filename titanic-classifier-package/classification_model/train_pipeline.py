import numpy as np
from config.core import config
from pipeline import titanic_pipe
from processing.data_manager import load_dataset, save_pipeline
from sklearn.model_selection import train_test_split


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.app_config.dataset)
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.app_config.features],  # predictors
        data[config.app_config.target],
        test_size=config.app_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.app_config.random_state,
    )

    # fit model
    titanic_pipe.fit(X_train, y_train)

    # persist trained model
    save_pipeline(pipeline_to_persist=titanic_pipe)


if __name__ == "__main__":
    run_training()