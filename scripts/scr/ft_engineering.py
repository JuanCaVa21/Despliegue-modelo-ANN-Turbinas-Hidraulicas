import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline as Pipe
from sklearn.model_selection import train_test_split

from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.outliers import Winsorizer
from feature_engine.selection import DropFeatures


def ft_engineering(var_num:list | None=None, var_cat:list | None = None, drop_var:list | None=None, quantile:float=0.05):
    """
    Construye y retorna un Pipeline de preprocesamiento para Feature Engineering.

    Cambios basados en EDA:
        - Reemplazado StandardScaler por MinMaxScaler (debido a distribuciones bimodales).
        - OHE configurado para evitar la trampa de variables ficticias (drop='first').
        - Los pasos se agregan dinámicamente solo si las listas de variables no están vacías.

    Args:
        var_cat (list | None): Variables categóricas nominales para OneHotEncoding (Ej. 'ID_Experimento', 'Frecuencia').
        var_num (list | None): Variables numéricas continuas para MinMaxScaler (Ej. 'Torque', 'RPM').
        drop_var (list | None): Variables altamente correlacionadas a eliminar (Ej. 'Presion (PsiD)', 'Caudal').
        quantile (float): Umbral para el Winsorizer en la cola derecha. Default: 0.05.

    Returns:
        Pipeline: Instancia de sklearn Pipeline lista para hacer `.fit(X_train)`.
    """

    # Configurar el preprocesador de Sklearn
    # MinMaxScaler es mejor para Redes Neuronales y datos No-Normales
    preprocessor_sk = ColumnTransformer(
        transformers=[
            ('numerical', MinMaxScaler(), var_num if var_num else []),
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), var_cat if var_cat else [])
        ],
        remainder='passthrough'
    )

    # Construir los pasos del Pipeline de forma dinámica
    steps = []
    
    if drop_var:
        steps.append(('drop_features', DropFeatures(features_to_drop=drop_var)))
        
    if var_num:
        steps.append(('imputer_numeric', MeanMedianImputer(imputation_method='median', variables=var_num)))
        steps.append(('outliers', Winsorizer(capping_method='quantiles', tail='right', fold=quantile, variables=var_num)))
        
    if var_cat:
        steps.append(('imputer_categorical', CategoricalImputer(imputation_method='frequent', variables=var_cat)))
        
    steps.append(('preprocessor', preprocessor_sk))

    Pipeline = Pipe(steps=steps)

    return Pipeline


def split_to_model(df:pd.DataFrame, target:str, test_size:float | None=0.2, random_state:int | None=42, stratify:bool=True):
    """
    Splits the DataFrame into train and test sets for model training.

    Validates that the target column exists, drops rows where the target is null,
    separates features (X) from the target (y), and performs a stratified or
    non-stratified train/test split.

    Warning:
        Stratification (`stratify=True`) is only valid for classification tasks
        where the target variable is discrete. For regression tasks with continuous
        targets, set `stratify=False` to avoid a ValueError raised by sklearn.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame ready for modeling.
        target (str | None): Name of the target column to predict.
            Raises ValueError if None or not found in `df`.
        test_size (float | None): Proportion of the dataset to allocate to the test set.
            Must be between 0.0 and 1.0. Default: 0.2 (20%).
        random_state (int | None): Random seed for reproducibility. Default: 42.
        stratify (bool): Whether to stratify the split by the target distribution.
            Set to False for regression tasks with continuous targets. Default: True.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            A four-element tuple (X_train, X_test, y_train, y_test) where:
            - X_train (pd.DataFrame): Training feature set.
            - X_test  (pd.DataFrame): Test feature set.
            - y_train (pd.Series): Training target values.
            - y_test  (pd.Series): Test target values.

    Raises:
        ValueError: If `target` is None or not found as a column in `df`.
        ValueError: If `stratify=True` and the target variable is continuous,
            as sklearn cannot stratify splits over non-discrete values.
    """

    if target is None:
        raise ValueError("A target column name must be provided.")

    if target in df.columns:
        df = df.dropna(subset=[target])
        X = df.drop(columns=target).copy()
        y = df[target]
        print(f'Variables successfully separated — X: {len(X)} rows, y: {len(y)} rows')
    else:
        raise ValueError(f'Column ({target}) was not found in the DataFrame.')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None  # Fix: made stratify optional for regression tasks
    )

    print(f'Train split size: {len(X_train)}')
    print(f'Test split size:  {len(X_test)}')

    return X_train, X_test, y_train, y_test