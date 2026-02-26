from cargar_datos import cargar_datos
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline as Pipe

from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.outliers import Winsorizer
from feature_engine.selection import DropFeatures

df = cargar_datos(needed_sheet='30Hz_P1')

def fix_dataframe(df:pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and reformats the raw turbine experimental data for feature engineering.

    This function performs a sequential cleanup process:
    1. Removes completely empty rows.
    2. Sets the first row as column headers and resets the index.
    3. Drops redundant or constant columns ('RPM', 'Power').
    4. Normalizes physical measurements (RPM, Power, Angle, Torque) to absolute values 
       to ensure physical consistency.
    5. Casts categorical/object columns to appropriate numeric (float/int) and 
       datetime dtypes for model compatibility.

    Args:
        df (pd.DataFrame): The raw input DataFrame containing sensor readings and 
            turbine operational data.

    Returns:
        pd.DataFrame: A cleaned DataFrame with corrected headers, absolute physical 
            values, and optimized data types.
    """

    df.dropna(how='all', inplace=True) # Eliminamos filas nulas

    # Renombramos columnas
    df.columns = df.iloc[0] 
    df = df[1:].reset_index(drop=True)

    # Eliminamos columnas innecesarias
    df.drop(columns=['RPM', 'Power'], inplace=True)

    # Valores absolutos de las columnas
    df['RPM_P'] = df['RPM_P'].abs()
    df['Power [Watts]'] = df['Power [Watts]'].abs()
    df['Angle'] = df['Angle'].abs()
    df['Torque [Nm]'] = df['Torque [Nm]'].abs()

    # Convertimos a Numericos
    df['Sample Number'] = pd.to_numeric(df['Sample Number'])
    df['Tracking Value'] = pd.to_numeric(df['Tracking Value'])
    df['Torque [Nm]'] = pd.to_numeric(df['Torque [Nm]'])
    df['Angle'] = pd.to_numeric(df['Angle'])
    df['RPM_P'] = pd.to_numeric(df['RPM_P'])
    df['Power [Watts]'] = pd.to_numeric(df['Power [Watts]'])

    #Convertimos a Fechas y tiempo
    df['Date'] = pd.to_datetime(df['Date'])

    return df

def ft_engineering(var_cat:list | None=None, var_num:list | None=None, var_ordi:list | None=None, drop_var:list | None=None, quantile:float = 0.05):
    """
    Feature Engineering of dataset

    This function performs a pipeline preprocess:
    1. Columns transformer of numerical, categorical and ordinal features.
    2. Drops innecesary features.
    3. Inputation of null numeric features to mean or median.
    4. Inputation of null categoric features to most frequent or mode .
    5. Mangement of outliers over wished quantile.

    Args:
        var_cats (list): The list of categorical features -> ej... ['id', 'Torque']
        var_num  (list): The list of numerical features -> ej... ['sex', 'type_of_worker']
        var_ordi (list): The list of Ordinal categorical features -> ej... ['education', 'age']
        drop_var (list): The list of features that you wants to drop -> ej... ['date', 'id_client']
        quantile (float): The number of percentil -> ej... 0.05(It´s default)

    Returns:
        Pipeline: A Pipeline that preprocess the data
    """

    preprocessor_sk = ColumnTransformer(
        transformers=[
            ('numerical', StandardScaler(), var_num),
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), var_cat),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), var_ordi)
        ],
        remainder='passthrough'
    )

    Pipeline = Pipe(
        steps= [
            ('drop_features', DropFeatures(features_to_drop = drop_var)),
            ('imputer_numeric', MeanMedianImputer(imputation_method='median', variables= var_num)),
            ('imputer_categorical', CategoricalImputer(imputation_method='frequent', variables= var_cat + var_num)),
            ('outliers', Winsorizer(capping_method='quantiles', tail='right', fold=quantile, variables= var_num)),
            ('preprocessor', preprocessor_sk)
        ]
    )

    return Pipe

def split_to_model(df:pd.DataFrame, target:str, test_size:float = 0.2, random_state:int = 42):

    df = df.dropna(subset=[target])