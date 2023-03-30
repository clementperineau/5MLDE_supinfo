import pickle
import pandas as pd
from typing import Any

from prefect import task

# Defining a Prefect task to print a divider
@task(name='print_devider', tags=['utils'])
def print_devider(title: str) -> None:
    # Printing a divider with the provided title
    print('\n{} {} {}\n'.format('-' * 25, title, '-' * 25))

# Defining a Prefect task to load a pickled object from a file
@task(name='load_pickle', tags=['utils'], retries=2, retry_delay_seconds=60)
def load_pickle(fp):
    # Opening the pickled file for reading and returning the deserialized object
    with open(fp, 'rb') as f:
        return pickle.load(f)

# Defining a Prefect task to load a pickled pandas DataFrame from a file
@task(name='load_data', tags=['utils'], retries=2, retry_delay_seconds=60)
def load_data(path: str) -> pd.DataFrame:
    # Loading the pickled DataFrame from the provided file path and returning it
    return pd.read_pickle(path)   

# Defining a Prefect task to load a CSV file into a pandas DataFrame
@task(name='load_csv', tags=['utils'], retries=2, retry_delay_seconds=60)
def load_csv(path: str) -> pd.DataFrame:
    # Loading the CSV file into a pandas DataFrame and returning it
    return pd.read_csv(path)

# Defining a Prefect task to save an object to a pickled file
@task(name='load_csv', tags=['utils'], retries=2, retry_delay_seconds=60)
def save_pickle(path: str, obj: dict):
    # Serializing the provided object and saving it to the specified file path
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
