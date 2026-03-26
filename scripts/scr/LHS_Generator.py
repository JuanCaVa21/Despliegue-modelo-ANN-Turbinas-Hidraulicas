import numpy as np
import pandas as pd
from scipy.stats import qmc
import matplotlib.pyplot as plt
import seaborn as sns

def LHS_DoE_Generation(
        num_of_test:int |None = 600, 
        path_to_export:str |None= 'LHS.csv', 
        name_of_target:str |None = 'Power_[Watts]',
        Atack_Angle:list | None = None, # [degree]
        Output_Angle:list |None = None, # [degree]
        Number_of_blades:list |None = None, # [Int]
        Blade_espesor:list |None=None, # [mm]
        Atack_Border:list |None=None, #[mm]
        Caudal:list |None = None, # [l/s]
        RPM:list |None = None # [RPM]
    ) -> pd.DataFrame:
    """
    Generates a Design of Experiments (DoE) using Latin Hypercube Sampling (LHS).

    This function creates a Latin Hypercube parameter space based on defined
    upper and lower bounds for various turbine parameters. The normalized space
    is scaled to the real limits provided, stored in a Pandas DataFrame along
    with an empty target column, and optionally exported to a CSV file.

    Args:
        num_of_test (int | None, optional): Number of samples to generate. Defaults to 600.
        path_to_export (str | None, optional): File path for exporting the generated DoE data. Defaults to 'LHS.csv'.
        name_of_target (str | None, optional): Name of the target output column to be appended. Defaults to 'Power_[Watts]'.
        Atack_Angle (list | None, optional): List [min, max] for Attack Angle limits [degrees].
        Output_Angle (list | None, optional): List [min, max] for Output Angle limits [degrees].
        Number_of_blades (list | None, optional): List [min, max] for Number of blades limits [int].
        Blade_espesor (list | None, optional): List [min, max] for Blade Thickness limits [mm].
        Atack_Border (list | None, optional): List [min, max] for Attack Border dimensions [mm].
        Caudal (list | None, optional): List [min, max] for flow rate limits [l/s].
        RPM (list | None, optional): List [min, max] for Rotational Speed bounds [RPM].

    Returns:
        pd.DataFrame: A generated sampling DataFrame formatted with parameter columns
            and an empty target column containing np.nan values.
    """

    initial_params = {
        'Atack_Angle': Atack_Angle, # [degree]
        'Output_Angle': Output_Angle, # [degree]
        'Number_of_blades': Number_of_blades, # [Int]
        'Blade_espesor': Blade_espesor, # [mm]
        'Atack_Border': Atack_Border, #[mm]
        'Caudal': Caudal, # [l/s]
        'RPM': RPM # [RPM]
    }

    params_names = list(initial_params.keys()) # Sacamos los nombres de los parametros a modificar
    n_of_params = len(initial_params) # Calculamos la cantidad de parametros 

    # Obtenemos valores min y max de los params
    inf_limits = [initial_params[p][0] for p in params_names]
    sup_limits = [initial_params[p][1] for p in params_names]

    # The sampling of LHS but in 1 and 0
    LHS_sampling = qmc.LatinHypercube(d=n_of_params, rng=42)
    # Now we need to normalisize 
    LHS_norm_samples = LHS_sampling.random(n=num_of_test)

    # Also how the LHS returns a sampling random data between 1/0
    # It's necessary to convert at real data so we use qmc.scale()
    real_sampling = qmc.scale(LHS_norm_samples, inf_limits, sup_limits)

    # To better accesibility convert Numpy Array to Pandas DataFrame
    df_samplings = pd.DataFrame(real_sampling, columns=params_names)
    df_samplings = df_samplings.round(3)

    # In this columns we agregate the data target before compute all script
    df_samplings[name_of_target] = np.nan

    # Saving in a csv
    df_samplings.to_csv(path_to_export, index=False)
    print(f'Generated {num_of_test} samplings')
    print(f'File saved like {path_to_export}')
    print('First 5 rows generates \n')
    print(df_samplings.head(5))

    return df_samplings