import os
import pandas as pd
import psycopg2 as psy
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')

def cargar_datos_from_excel(excel_path:str, sheet_name:str) -> pd.DataFrame:
    """
    Carga datos desde una hoja específica de un archivo Excel.

    Verifica la existencia del archivo en la ruta proporcionada y extrae los 
    datos de la hoja indicada para convertirlos en un DataFrame de Pandas.

    Args:
        excel_path (str): La ruta absoluta o relativa al archivo Excel.
        sheet_name (str): El nombre de la hoja dentro del archivo Excel a leer.

    Returns:
        pd.DataFrame: Un DataFrame con los datos extraídos. Si el archivo no existe
                      o ocurre un error durante la carga, retorna un DataFrame vacío.
    """

    try:
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
        print(f"Cargados {len(df)} datos")
        return df
    except Exception as e:
        print(f"Error cargando {e}")

def cargar_datos_from_supabase(n_data: int, freq: int) -> pd.DataFrame:
    """
    Extrae una muestra aleatoria de datos operativos desde Supabase (PostgreSQL).
    
    Se conecta a la base de datos utilizando las credenciales almacenadas en 
    las variables de entorno (.env). Ejecuta una consulta SQL para obtener datos 
    filtrados por frecuencia experimental y limitados a una cantidad específica.

    Args:
        n_data (int): Cantidad máxima de registros a extraer.
        freq (int): Frecuencia del experimento utilizada para filtrar los datos.

    Returns:
        pd.DataFrame: DataFrame con los datos extraídos. Retorna un DataFrame 
                      vacío en caso de error de conexión o fallo en la consulta.
    """
    
    load_dotenv() # Cambia la ruta de tu .env si es necesario

    # Extraemos la base de datos del .env
    Supabase_conn = {
        'user': os.getenv('DB_USER'),
        'dbname': os.getenv('DB_DATABASE'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT'),
        'sslmode': 'require'
    }

    # Nota: Revisa si el nombre de tu tabla realmente es 'experiemts' o si fue 
    # un error tipográfico y debería ser 'experiments'.
    query_extr = f"""
    SELECT * FROM public.experiemts
    WHERE freq_exp = {freq}
    ORDER BY RANDOM()
    LIMIT (
        SELECT CASE 
            WHEN count(*) > 500 THEN {n_data}
            ELSE count(*) 
        END
        FROM public.experiemts
        WHERE freq_exp = {freq}
    );
    """

    # Intentamos conectar a la base de datos
    try:
        Supa_con = psy.connect(**Supabase_conn)
        print('Conexión a la base de datos exitosa')
    except Exception as e:
        print(f'Error al conectar a la base de datos: {e}')
        return pd.DataFrame()

    # Cargamos el dataset
    df = pd.DataFrame() # Inicializamos un DataFrame vacío por defecto
    try:
        if n_data is not None and freq is not None:
            df = pd.read_sql(query_extr, Supa_con)
            print(f'Cargados {len(df)} datos desde Supabase')
        else:
            print('No se ha especificado la cantidad de datos o la frecuencia')
    except Exception as e:
        print(f'Error al ejecutar la consulta SQL: {e}')
    finally:
        # Es una buena práctica cerrar la conexión si se abrió exitosamente
        if 'Supa_con' in locals() and Supa_con:
            Supa_con.close()
    
    return df