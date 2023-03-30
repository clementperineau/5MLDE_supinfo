import os
import re
import numpy as np
import pandas as pd
import mlflow
from utils import load_pickle, load_data
from config import DATA_DIR, PROCESSED_TEST_PATH

from prefect import task, flow

class EnsembleModel:
    def __init__(self, models: str) -> None:
        self.models = models

    def predict_proba(self, X: pd.DataFrame) -> float:
        proba = np.zeros((len(X), self.models[0].n_classes_))
        for model in self.models:
            proba += model.predict_proba(X, num_iteration=model.best_iteration_)
        return proba / len(self.models)

 

@flow(name="Predict", retries=1, retry_delay_seconds=30)
def main() -> None:
    EXPERIMENT_ID = '1'
    runs = mlflow.search_runs(EXPERIMENT_ID,
                              order_by=['metrics.accuracy DESC', 'attribute.start_time DESC'],
                              max_results=10)
    print(runs)

    models_path = re.sub('^file://', '', runs.loc[0, 'params.model_path'])
    print(models_path)
    models = load_pickle(models_path)
    model = EnsembleModel(models)

    X_test = load_data(PROCESSED_TEST_PATH)
    proba = model.predict_proba(X_test)[:, 1]
    fp = os.path.join(DATA_DIR, 'prediction.csv')
    pd.DataFrame(proba, columns=['proba']).to_csv(fp, index=False)


if __name__ == '__main__':
    main()
