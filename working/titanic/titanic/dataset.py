import numpy as np
import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import Tuple

from .utils import try_except


@try_except(True)
def read_csv_file(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

@try_except(True)
def process_data(df: pd.DataFrame, imputer=None, scaler=None) -> Tuple[np.array, SimpleImputer, StandardScaler]:
    data = df[["Pclass", "Age", "SibSp", "Parch", "Fare"]]
    
    print(imputer, scaler)
    if imputer is None:
        imputer = SimpleImputer(strategy='median')
        data_imputed = imputer.fit_transform(data)
    else:
        data_imputed = imputer.transform(data)
    
    if scaler is None:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_imputed)
    else:
        data_scaled = scaler.transform(data_imputed)

    #print("data:", data)
    #print("data_imputed:", data_imputed)
    #print("data_scaled:", data_scaled)

    return np.array(data_scaled), imputer, scaler

@try_except(True)
def get_train_data(df: pd.DataFrame, label_name: str, imputer=None, scaler=None) -> Tuple[np.array, np.array, SimpleImputer, StandardScaler]:
    X, imputer, scaler = process_data(df.drop(label_name, axis=1), imputer, scaler)
    y = df[label_name].values
    return X, y, imputer, scaler

@try_except(True)
def get_test_data(df: pd.DataFrame, imputer=None, scaler=None) -> Tuple[np.array, np.array, SimpleImputer, StandardScaler]:
    ID = df["PassengerId"].values
    return ID, *process_data(df, imputer, scaler)

def get_input_data(file_name: str, is_train: bool, imputer=None, scaler=None) -> Tuple[np.array, np.array, SimpleImputer, StandardScaler]:
    input_dir = os.path.join("..", "..", "input", "titanic")
    df = read_csv_file(os.path.join(input_dir, file_name))
    return get_train_data(df, "Survived", imputer, scaler) if is_train else (
           get_test_data(df, imputer, scaler) )