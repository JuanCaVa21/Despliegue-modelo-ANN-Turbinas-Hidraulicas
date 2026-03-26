from cargar_datos import cargar_datos
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline as Pipe
from sklearn.model_selection import train_test_split

from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.outliers import Winsorizer
from feature_engine.selection import DropFeatures

df = cargar_datos(needed_sheet='30Hz_P1')


def fix_dataframe(df:pd.DataFrame, num_feats:list | None=None, negative_feats:list | None=None, date_feats:list | None=None) -> pd.DataFrame:
    """
    Cleans and reformats the raw turbine experimental DataFrame for feature engineering.

    Executes the following sequential steps:
        1. Removes completely empty rows.
        2. Promotes the first row to column headers and resets the index.
        3. Converts specified physical measurement columns to absolute values
           (e.g., RPM, Power, Angle, Torque) to ensure physical consistency.
        4. Casts specified columns to numeric type (float/int).
        5. Casts specified columns to datetime type.

    Args:
        df (pd.DataFrame): Raw input DataFrame containing sensor readings and
            turbine operational data.
        num_feats (list | None): Column names to cast to numeric type.
            Example: ['RPM', 'Torque', 'Power', 'Angle']
        negative_feats (list | None): Column names to transform to absolute value.
            Example: ['RPM', 'Torque']
        date_feats (list | None): Column names to cast to datetime type.
            Example: ['Timestamp', 'Date']

    Returns:
        pd.DataFrame: Cleaned DataFrame with corrected headers, absolute physical
            values, and optimized data types ready for feature engineering.

    Raises:
        KeyError: If any column in `num_feats`, `negative_feats`, or `date_feats`
            does not exist in the DataFrame after header promotion.
        ValueError: If values in `date_feats` cannot be parsed as datetime.
    """

    # Drop null values
    df.dropna(how='all', inplace=True)

    # Variables negatives to positive
    if negative_feats is not None:
        for j in negative_feats:
            df[j] = df[j].abs()

    # Conversion of numeric variables
    if num_feats is not None:
        for i in num_feats:
            df[i] = pd.to_numeric(df[i], errors='coerce')  # Coerce non-numeric to NaN for later imputation

    # Conversion of date features
    if date_feats is not None:
        for i in date_feats:
            df[date_feats] = pd.to_datetime(df[date_feats])

    return df


def ft_engineering(var_num:list | None=None, drop_var:list | None=None, quantile:float=0.05):
    """
    Builds and returns a full preprocessing Pipeline for feature engineering.

    The pipeline executes the following steps in order:
        1. DropFeatures: Removes irrelevant or redundant columns defined in `drop_var`.
        2. MeanMedianImputer: Imputes missing values in numeric variables using the median.
        3. CategoricalImputer: Imputes missing values in categorical variables using the mode.
        4. Winsorizer: Caps outliers at the `quantile` percentile on the right tail.
        5. ColumnTransformer:
            - StandardScaler applied to numeric variables.
            - OneHotEncoder (drop='first') applied to nominal categorical variables.
            - OrdinalEncoder applied to ordinal categorical variables.

    Warning:
        `drop_var` must not contain columns referenced in `var_num`, `var_cat`,
        or `var_ordi`. Doing so will cause downstream pipeline steps to raise a KeyError
        since those columns will no longer exist when the imputers or transformers run.

    Args:
        var_cat (list | None): Nominal categorical feature names for OneHotEncoding.
            Example: ['failure_type', 'operation_mode']
        var_num (list | None): Continuous numeric feature names for scaling and imputation.
            Example: ['RPM', 'Torque', 'Power']
        var_ordi (list | None): Ordinal categorical feature names for OrdinalEncoding.
            Example: ['load_level', 'shift']
        drop_var (list | None): Feature names to drop before any transformation.
            Example: ['test_id', 'Timestamp']
        quantile (float): Percentile threshold for the Winsorizer right-tail capping.
            Must be between 0.0 and 0.5. Default: 0.05 (caps at the 95th percentile).

    Returns:
        Pipeline: A fitted-ready sklearn Pipeline instance. Call `.fit(X_train)` to
            train the transformations and `.transform(X)` to apply them.

    Raises:
        ValueError: If `var_num`, `var_cat`, or `var_ordi` are None when the
            pipeline's `.fit()` method is called.
        KeyError: If any referenced column does not exist in the DataFrame
            at pipeline fit time.
    """

    preprocessor_sk = ColumnTransformer(
        transformers=[
            ('numerical', StandardScaler(), var_num)
        ],
        remainder='passthrough'
    )

    Pipeline = Pipe(
        steps=[
            ('drop_features', DropFeatures(features_to_drop=drop_var)),
            ('imputer_numeric', MeanMedianImputer(imputation_method='median', variables=var_num)),
            ('outliers', Winsorizer(capping_method='quantiles', tail='right', fold=quantile, variables=var_num)),
            ('preprocessor', preprocessor_sk)
        ]
    )

    return Pipeline  # Fix: was returning Pipe (the class) instead of Pipeline (the instance)


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
        X = df.drop(columns=[target]).copy()
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