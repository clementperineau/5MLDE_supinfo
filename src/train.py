

import os
import warnings
import pickle
from collections import defaultdict
from typing import Callable, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             confusion_matrix,
                             roc_curve,
                             roc_auc_score,
                             precision_recall_curve,
                             average_precision_score)
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgbm
import mlflow

import plot_funcs as pf
from utils import print_devider
from config import PROCESSED_TRAIN_PATH

from prefect import task, flow

# Serializing the provided object and saving it to the specified file path
@task(name='devide_by_sum', tags=['train'])
def devide_by_sum(x: float) -> float:
    return x / x.sum()

# Task to calculate evaluation scores for a classification model
@task(name='get_scores', tags=['train'])
def get_scores(y_true: list, y_pred: list) -> dict:
    # Calculate and return the evaluation scores
    return {
      'accuracy': accuracy_score(y_true, y_pred),
      'precision': precision_score(y_true, y_pred),
      'recall': recall_score(y_true, y_pred),
      'f1': f1_score(y_true, y_pred),
    }

# A function that logs a plot and deletes the local file.
@flow(name='log_plot')
def log_plot(args: Any, plot_func: Callable, fp: str) -> None:
    if not isinstance(args, (tuple)):
        args = (args,)

    plot_func(*args, fp)
    mlflow.log_artifact(fp)
    os.remove(fp)
    print(f'Logged {fp}')

@flow(name='train_model')
def train_model(X, y, params: dict, exp_path: str) -> tuple[str, str]:
    """
    A function that trains a machine learning model and logs various artifacts using MLflow.

    Parameters:
    X (pandas.DataFrame): The input features.
    y (pandas.Series): The target variable.
    params (dict): The hyperparameters to use for training the model.
    exp_path (str): The path of the experiment to log artifacts to.

    Returns:
    tuple[str, str]: A tuple containing the experiment ID and run ID of the MLflow run.
    """
    fold_params = params['fold']
    model_params = params['model']
    fit_params = params['fit']

    # set mlflow experiment
    try:
        mlflow.create_experiment(exp_path)
    except (mlflow.exceptions.RestException, mlflow.exceptions.MlflowException):
        print('The specified experiment ({}) already exists.'.format(exp_path))

    mlflow.set_experiment(exp_path)

    skf = StratifiedKFold(**fold_params)
    models = []
    metrics = []

    y_proba = np.zeros(len(X))
    y_pred = np.zeros(len(X))

    feature_importances_split = np.zeros(X.shape[1])
    feature_importances_gain = np.zeros(X.shape[1])

    scores = defaultdict(int)

    with mlflow.start_run() as run:
        corr = pd.concat((X, y), axis=1).corr()
        log_plot(corr, pf.corr_matrix, 'correlation_matrix.png')
        log_plot(y.value_counts(), pf.label_share, 'label_share.png')

        for fold_no, (idx_train, idx_valid) in enumerate(skf.split(X, y)):
            print_devider(f'Fold: {fold_no}')

            X_train, X_valid = X.iloc[idx_train, :], X.iloc[idx_valid, :]
            y_train, y_valid = y.iloc[idx_train], y.iloc[idx_valid]

            # train model
            model = lgbm.LGBMClassifier(**model_params)
            model.fit(X_train, y_train, **fit_params, eval_set=[(X_valid, y_valid)], eval_names=['valid'])
            metrics.append({
              'name': model.metric,
              'values': model.evals_result_['valid'][model.metric],
              'best_iteration': model.best_iteration_
            })
            models.append(model)

            # feature importance
            feature_importances_split += devide_by_sum(model.booster_.feature_importance(importance_type='split')) / skf.n_splits
            feature_importances_gain += devide_by_sum(model.booster_.feature_importance(importance_type='gain')) / skf.n_splits

            # predict
            y_valid_proba = model.predict_proba(X_valid, num_iteration=model.best_iteration_)[:, 1]
            y_valid_pred = model.predict(X_valid, num_iteration=model.best_iteration_)
            y_proba[idx_valid] = y_valid_proba
            y_pred[idx_valid] = y_valid_pred

            # evaluate
            scores_valid = get_scores(y_valid, y_valid_pred)

            mlflow.log_metrics({
              **scores_valid,
              'best_iteration': model.best_iteration_,
            }, step=fold_no)

            print('\nScores')
            print(scores_valid)

            # record scores
            for k, v in scores_valid.items():
                scores[k] += v / skf.n_splits

        # log training parameters
        mlflow.log_params({
          **fold_params,
          **model_params,
          **fit_params,
          'cv': skf.__class__.__name__,
          'model': model.__class__.__name__
        })

        print_devider('Saving plots')

        # scores
        log_plot(scores, pf.scores, 'scores.png')

        # feature importance
        features = np.array(model.booster_.feature_name())
        log_plot((features, feature_importances_split, 'Feature Importance: split'),
                 pf.feature_importance, 'feature_importance_split.png')
        log_plot((features, feature_importances_gain, 'Feature Importance: gain'),
                 pf.feature_importance, 'feature_importance_gain.png')

        # metric history
        log_plot(metrics, pf.metric, 'metric_history.png')

        # confusion matrix
        cm = confusion_matrix(y, y_pred)
        log_plot(cm, pf.confusion_matrix, 'confusion_matrix.png')

        # roc curve
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = roc_auc_score(y, y_pred)
        log_plot((fpr, tpr, roc_auc), pf.roc_curve, 'roc_curve.png')

        # precision-recall curve
        pre, rec, _ = precision_recall_curve(y, y_proba)
        pr_auc = average_precision_score(y, y_pred)
        log_plot((pre, rec, pr_auc), pf.pr_curve, 'pr_curve.png')

        # pickle trained models
        models_path = 'models.pkl'
        with open(models_path, 'wb') as f:
            pickle.dump(models, f)
            mlflow.log_artifact(models_path)
            mlflow.log_param('model_path', os.path.join(run.info.artifact_uri, models_path))
            os.remove(models_path)

    return run.info.experiment_id, run.info.run_id

@flow(name="Training", retries=1, retry_delay_seconds=30)
def main() -> None:
    warnings.filterwarnings('ignore')

    print(os.listdir('data'))
    train = pd.read_pickle(PROCESSED_TRAIN_PATH)

    X = train.drop('Survived', axis=1)
    y = train['Survived']

    SEED = 0

    params = {
      'model': {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 100000,
        'learning_rate': 0.05,
        'random_state': SEED,
        'n_jobs': -1
      },

      'fit': {
        'early_stopping_rounds': 100,
        'verbose': 10
      },

      'fold': {
        'n_splits': 5,
        'shuffle': True,
        'random_state': SEED
      }
    }

    experiment_id, run_uuid = train_model(X, y, params, 'titanic')


if __name__ == '__main__':
    main()
