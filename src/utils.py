import pickle
import pandas as pd
from typing import Any

from prefect import task

@task(name='print_devider', tags=['utils'])
def print_devider(title: str) -> None:
    print('\n{} {} {}\n'.format('-' * 25, title, '-' * 25))

@task(name='load_pickle', tags=['utils'], retries=2, retry_delay_seconds=60)
def load_pickle(fp):
    with open(fp, 'rb') as f:
        return pickle.load(f)

@task(name='load_data', tags=['utils'], retries=2, retry_delay_seconds=60)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_pickle(path)   

@task(name='load_csv', tags=['utils'], retries=2, retry_delay_seconds=60)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@task(name='load_csv', tags=['utils'], retries=2, retry_delay_seconds=60)
def save_pickle(path: str, obj: dict):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)