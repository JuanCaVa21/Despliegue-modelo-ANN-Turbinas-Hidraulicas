import pandas as pd
import numpy as np
import re

BASE = "/home/juan21/Documentos/GitHub/Despliegue-modelo-ANN-Turbinas-Hidraulicas/database/raw"

# Paths con los XLS que arroja el sensor 
paths = {
    f"{BASE}/PAT_30Hz_P1.XLS": 'df_30_P1',
    #f"{BASE}/PAT_30Hz_P2.XLS": 'df_30_P2',
    #f"{BASE}/PAT_30Hz_P3.XLS": 'df_30_P3',
    #f"{BASE}/PAT_40Hz_P1.XLS": 'df_40_P1',
    #f"{BASE}/PAT_40Hz_P2.XLS": 'df_40_P2',
    #f"{BASE}/PAT_40Hz_P3.XLS": 'df_40_P3',
    #f"{BASE}/PAT_50Hz_P1.XLS": 'df_50_P1',
    #f"{BASE}/PAT_50Hz_P2.XLS": 'df_50_P2',
    #f"{BASE}/PAT_50Hz_P3.XLS": 'df_50_P3',
    #f"{BASE}/PAT_60Hz_P1.XLS": 'df_60_P1',
    #f"{BASE}/PAT_60Hz_P2.XLS": 'df_60_P2',
    #f"{BASE}/PAT_60Hz_P3.XLS": 'df_60_P3',
}

# Paths donde estan los datos que enlanzan las RPM con caudal
path_models = {
    f"{BASE}/math_models/30 Hz.xlsx": ['30 HZ Prueba 1', '30 HZ Prueba 2', '30 HZ Prueba 3'],
    f"{BASE}/math_models/40 Hz.xlsx": ['40 HZ Prueba 1', '40 HZ Prueba 2', '40 HZ Prueba 3'],
    f"{BASE}/math_models/50 Hz.xlsx": ['50 HZ Prueba 1', '50 HZ Prueba 2', '50 HZ Prueba 3'],
    f"{BASE}/math_models/60 Hz.xlsx": ['60 HZ Prueba 1', '60 HZ Prueba 2', '60 HZ Prueba 3'],
}

# Cada archivo de datos se ajusta con el modelo de su prueba correspondiente.
# Key: data_path  →  Value: (model_path, sheet) que existe en model_registry.
data_to_model = {
    f"{BASE}/PAT_30Hz_P1.XLS": (f"{BASE}/math_models/30 Hz.xlsx", "30 HZ Prueba 1"),
    #f"{BASE}/PAT_30Hz_P2.XLS": (f"{BASE}/math_models/30 Hz.xlsx", "30 HZ Prueba 2"),
    #f"{BASE}/PAT_30Hz_P3.XLS": (f"{BASE}/math_models/30 Hz.xlsx", "30 HZ Prueba 3"),
    #f"{BASE}/PAT_40Hz_P1.XLS": (f"{BASE}/math_models/40 Hz.xlsx", "40 HZ Prueba 1"),
    #f"{BASE}/PAT_40Hz_P2.XLS": (f"{BASE}/math_models/40 Hz.xlsx", "40 HZ Prueba 2"),
    #f"{BASE}/PAT_40Hz_P3.XLS": (f"{BASE}/math_models/40 Hz.xlsx", "40 HZ Prueba 3"),
    #f"{BASE}/PAT_50Hz_P1.XLS": (f"{BASE}/math_models/50 Hz.xlsx", "50 HZ Prueba 1"),
    #f"{BASE}/PAT_50Hz_P2.XLS": (f"{BASE}/math_models/50 Hz.xlsx", "50 HZ Prueba 2"),
    #f"{BASE}/PAT_50Hz_P3.XLS": (f"{BASE}/math_models/50 Hz.xlsx", "50 HZ Prueba 3"),
    #f"{BASE}/PAT_60Hz_P1.XLS": (f"{BASE}/math_models/60 Hz.xlsx", "60 HZ Prueba 1"),
    #f"{BASE}/PAT_60Hz_P2.XLS": (f"{BASE}/math_models/60 Hz.xlsx", "60 HZ Prueba 2"),
    #f"{BASE}/PAT_60Hz_P3.XLS": (f"{BASE}/math_models/60 Hz.xlsx", "60 HZ Prueba 3"),
}

dataframes = {}
model_registry = {}

# Quitamos los espacios de los nombres
def clean_col(name: str) -> str:
    return re.sub(r'\s+', ' ', name.strip())


print("Calculando modelos matemáticos...")

# Calculamos los modelos matematicos diseñados para cada experimentos
for model_path, sheets in path_models.items():
    for sheet in sheets:
        try:
            df_ref = pd.read_excel(model_path, sheet_name=sheet)
            df_ref.columns = [clean_col(c) for c in df_ref.columns]

            if 'RPM' in df_ref.columns and 'Caudal' in df_ref.columns:
                # Usamos una aproximacion polinimica de grado 2
                z = np.polyfit(df_ref['RPM'], df_ref['Caudal'], 2)
                # Guardamos los modelos matematicos para usarlos en cada caso
                model_registry[(model_path, sheet)] = np.poly1d(z)
                print(f"  Modelo ajustado: {sheet}")
        except Exception as e:
            print(f"  Error cargando modelo {sheet} en {model_path}: {e}")

print("Procesando datos experimentales...")

# Comenzamos el arreglo del dataset 
for data_path, df_name in paths.items():
    df = pd.read_excel(data_path) 
    df.columns = [clean_col(c) for c in df.columns]

    # Para poder trackear todas las variables
    df = df.rename(columns={
        'Tracking Value': 'Torque',
        'Sample Number': 'EXP_ID',
        'Time Elapsed': 'Tiempo_Ejecucion',
    })

    # Arreglamos tipos de variables y ademas modificamos valores negaticos y kW por W
    df['Tiempo_Ejecucion'] = pd.to_timedelta(df['Tiempo_Ejecucion'].astype(str))
    df['RPM'] = df['RPM'] * -1
    df['Torque'] = df['Torque'] * -1
    df['Power'] = df['Power'].abs() * 1000

    # Extraer frecuencia del nombre del archivo (ej. PAT_30Hz_P1.XLS → 30)
    freq_match = re.search(r'(\d+)Hz', data_path)
    df['Frecuencia'] = int(freq_match.group(1)) if freq_match else None

    # Calcular Caudal aplicando el polinomio de la prueba que corresponde a este archivo.
    model_key = data_to_model.get(data_path)
    if model_key and model_key in model_registry:
        df['Caudal'] = model_registry[model_key](df['RPM'])
        print(f"  {df_name}: Caudal calculado con modelo '{model_key[1]}'")
    else:
        print(f"  ADVERTENCIA: No se encontró modelo para {data_path}")

    dataframes[df_name] = df

# Guardamos el nuevo dataset ya listo en un csv
output_path = f"{BASE}/../processed/turbina_exp.csv"

# Concatenamos y guardamos el csv
combined = pd.concat(dataframes.values(), ignore_index=True)
combined.to_csv(output_path, index=False)

print(f"CSV guardado en: {output_path}")
