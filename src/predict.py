import os
import re
import numpy as np
import pandas as pd
import mlflow
from utils import load_pickle, load_data
from config import DATA_DIR, PROCESSED_TEST_PATH

from prefect import task, flow

# Defining a class to create an ensemble of models
class EnsembleModel:
    def __init__(self, models: str) -> None:
        self.models = models

    # Function to predict class probabilities using the ensemble of models
    def predict_proba(self, X: pd.DataFrame) -> float:
        proba = np.zeros((len(X), self.models[0].n_classes_))
        for model in self.models:
            proba += model.predict_proba(X, num_iteration=model.best_iteration_)
        return proba / len(self.models)

 
# Creating a Prefect flow for predicting on test data
@flow(name="Predict", retries=1, retry_delay_seconds=30)
def main() -> None:
    # Set the ID of the MLflow experiment to use
    EXPERIMENT_ID = '1'
    # Retrieve the top 10 runs based on accuracy and start time
    runs = mlflow.search_runs(EXPERIMENT_ID,
                              order_by=['metrics.accuracy DESC', 'attribute.start_time DESC'],
                              max_results=10)
    print(runs)
    # Extract the path of the trained models from the top run
    models_path = re.sub('^file://', '', runs.loc[0, 'params.model_path'])
    print(models_path)
    # Load the trained models from the path
    models = load_pickle(models_path)
    # Create an ensemble model from the loaded models
    model = EnsembleModel(models)

    # Load the test data from the specified path
    X_test = load_data(PROCESSED_TEST_PATH)
    # Predict the class probabilities using the ensemble model
    proba = model.predict_proba(X_test)[:, 1]
    # Save the predicted probabilities as a CSV file
    fp = os.path.join(DATA_DIR, 'prediction.csv')
    pd.DataFrame(proba, columns=['proba']).to_csv(fp, index=False)


if __name__ == '__main__':
    main()
