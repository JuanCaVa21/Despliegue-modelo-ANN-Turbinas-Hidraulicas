import os
import pandas as pd
from dotenv import load_dotenv

def cargar_datos(needed_sheet:str) -> pd.DataFrame:
    """
    Loads operational data from a specific sheet in the database Excel file.
    
    Retrieves the file path from environment variables and reads the specified 
    sheet. Serves as the entry point for ingesting physical test readings.

    Args:
        needed_sheet (str): Exact name of the Excel sheet to load (e.g., '30Hz_P1').

    Returns:
        pd.DataFrame: DataFrame with the extracted data. Returns an empty 
                      DataFrame if the file path or sheet cannot be loaded.
    """
    
    load_dotenv(dotenv_path='/Users/juanv/Documents/Github/Despliegue-modelo-ANN-Turbinas-Hidraulicas/.env') # Cambia la ruta de tu .env

    # Extraemos la base de datos del .env
    Data = os.getenv('DB_USER')

    # Cargamos el dataset
    try:
        # Creamos el dataframe 
        if Data and needed_sheet is not None:
            df = pd.read_excel(Data, sheet_name=needed_sheet)
            print(f'Cargados correctamente {len(df)} datos')
        else:
            print('No existe el path')
    except Exception as e:
        print(f'No existe la Sheet debes escribir el formato completo {e}')
    
    return df

if __name__ == "__main__":
    df = cargar_datos('30Hz_P1')