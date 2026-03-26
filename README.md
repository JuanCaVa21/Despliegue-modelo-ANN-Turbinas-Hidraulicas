# Pipeline de Deep Learning para Predicción de Potencia en Turbinas Hidráulicas Banki

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-green?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.x-darkblue?style=flat-square&logo=pandas)
![License](https://img.shields.io/badge/License-MIT-brightgreen?style=flat-square)

---

## 📋 Descripción del Proyecto

Este repositorio implementa un **pipeline de Machine Learning end-to-end** para predecir la potencia generada (en **Watts**) por una turbina hidráulica tipo **Banki** a partir de datos físicos capturados en laboratorio. El modelo integra técnicas avanzadas de preprocesamiento, ingeniería de características y deep learning, optimizadas específicamente para datos tabulares de sensores en sistemas hidráulicos.

### 🎯 Objetivo
Desarrollar un sistema **robusto, reproducible y productivo** que permita predecir la potencia generada por turbinas hidráulicas bajo diferentes condiciones operacionales, facilitando su integración en sistemas de control y optimización energética.

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE DATOS                            │
└─────────────────────────────────────────────────────────────────┘

1. EXTRACCIÓN (cargar_datos.py)
   └─> Excel (.xlsx) via variables de entorno (.env)
       └─> Multiple sheets (30Hz_P1, 30Hz_P2, etc.)

2. LECTURA & VALIDACIÓN (Pandas)
   └─> Lee desde ruta especificada en DB_USER
       └─> Manejo de errores de hojas inexistentes

3. PREPROCESAMIENTO (fix_dataframe)
   └─> Eliminación de filas nulas
   └─> Conversión a valores absolutos (RPM, Torque, Angle, Power)
   └─> Conversión a tipos numéricos estrictos (float64)
   └─> Evita LossySetitemError

4. INGENIERÍA DE CARACTERÍSTICAS (ft_engineering Pipeline)
   ├─> DropFeatures: Quita columnas redundantes dinamicamente
   ├─> MeanMedianImputer: Imputa missing values con mediana
   ├─> Winsorizer: Maneja outliers (percentil 95)
   ├─> StandardScaler: Normaliza variables numéricas
   └─> No estratificación en split (regresión, no clasificación)

5. MODELADO (MLP con TensorFlow/Keras)
   ├─> Arquitectura: [Input] -> Dense(64, relu) -> Dense(32, relu) -> Dense(1)
   ├─> Optimizador: Adam
   ├─> Loss: Mean Squared Error (MSE)
   ├─> Métrica: Mean Absolute Error (MAE)
   └─> CPU-only Mode: CUDA_VISIBLE_DEVICES = '-1'

6. EVALUACIÓN & VISUALIZACIÓN (Seaborn + Matplotlib)
   ├─> Curvas de pérdida (train vs validation)
   ├─> Predicciones vs Valores Reales
   └─> Análisis de residuos

7. INFERENCIA DE PRODUCCIÓN ("Molde de DataFrame")
   └─> Inyectar nuevos escenarios (RPM, Torque, Angle)
       └─> Transformar con Pipeline pre-entrenado
       └─> Predicción directa sin Data Leakage
```

---

## 🏆 Resultados Destacados

| Métrica | Valor | Unidad |
|---------|-------|--------|
| **Error Absoluto Medio (MAE)** en Test | 1.78 | Watts |
| **Epochs de Entrenamiento** | 150 | - |
| **Batch Size** | 32 | muestras |
| **Línea Base (Validación)** | Convergencia estable | Sin overfitting |

### 📊 Hallazgos Clave:
- ✅ **Convergencia estable:** Las curvas de pérdida (train y validation) se desacoplan ligeramente pero no muestran overfitting severo.
- ✅ **Generalización:** El modelo predice potencias futuras con MAE ≤ 1.5 W, aceptable para sistemas de control hidráulico.
- ✅ **Robustez del Pipeline:** Maneja automáticamente hojas Excel con estructuras variables sin `KeyError`.
- ✅ **Integridad de Datos:** Evita Data Leakage mediante aislamiento estricto del target y sin estratificación en regresión.

---

## 🚀 Cómo Replicar el Proyecto

### ✋ Requisitos Previos
- **Python 3.8+**
- **pip** o **conda** package manager
- Una copia de la base de datos Excel (`AFMP_Pruebas_curvas.xlsx`)

### 📦 Instalación Paso a Paso

#### 1️⃣ Clonar el Repositorio
```bash
git clone https://github.com/tu-usuario/Despliegue-modelo-ANN-Turbinas-Hidraulicas.git
cd Despliegue-modelo-ANN-Turbinas-Hidraulicas
```

#### 2️⃣ Crear un Entorno Virtual
```bash
# Con venv (recomendado)
python -m venv .ANN_Env
source .ANN_Env/bin/activate  # En Windows: .ANN_Env\Scripts\activate

# O con conda
conda create -n ANN_Env python=3.10
conda activate ANN_Env
```

#### 3️⃣ Instalar Dependencias
```bash
pip install -r requirements.txt
```

**Dependencias principales:**
- `pandas` - Manipulación de datos
- `numpy` - Computación numérica
- `scikit-learn` - Preprocesamiento y validación
- `feature-engine` - Ingeniería de características avanzada
- `tensorflow` - Training del modelo deep learning
- `matplotlib`, `seaborn` - Visualizaciones
- `python-dotenv` - Gestión de variables de entorno

#### 4️⃣ Configurar Variables de Entorno
Crea un archivo `.env` en la raíz del proyecto:
```bash
# .env
DB_USER=/ruta/absoluta/a/tu/base_datos/AFMP_Pruebas_curvas.xlsx
```

**Ejemplo (Linux/Mac):**
```bash
DB_USER=/home/usuario/datos/AFMP_Pruebas_curvas.xlsx
```

**Ejemplo (Windows):**
```bash
DB_USER=C:\Users\usuario\datos\AFMP_Pruebas_curvas.xlsx
```

#### 5️⃣ Ejecutar el Entrenamiento
```bash
# Navega a la carpeta del modelo MLP
cd scripts/MLP

# Ejecuta el notebook en modo batch (requiere jupyter)
jupyter nbconvert --to notebook --execute MLP_Neuron.ipynb

# O abre el notebook interacticamente
jupyter notebook MLP_Neuron.ipynb
```

**Salida esperada:**
```
Cargados correctamente 663 datos
Número de columnas finales para la Red Neuronal: 3
Train split size: 530
Test split size:  133
Epoch 1/150 [=============>...] - loss: 45.32 - mae: 5.43 - val_loss: 38.21 - val_mae: 4.87
...
Error Absoluto Medio (MAE) en datos de prueba: 2.23 Watts
```

---

## 💡 Cómo Hacer Predicciones Nuevas

El sistema utiliza una técnica de **"molde de DataFrame"** que inyecta nuevos escenarios operacionales sin corromper el pipeline pre-entrenado.

### Ejemplo de Inferencia

```python
import pandas as pd
from scripts.scr.cargar_datos import cargar_datos
from scripts.scr.ft_engineering import ft_engineering, split_to_model
import tensorflow as tf

# ============================================
# 1. CARGAR EL PIPELINE PRE-ENTRENADO
#    (En producción, esto vendría de un archivo .pkl)
# ============================================
df_original = cargar_datos('30Hz_P1')
numerics = ['RPM', 'Torque [Nm]', 'Angle']
Pipeline = ft_engineering(var_num=numerics, drop_var=['RPM_P', 'Power', 'Tracking Value', 'Sample Number'], quantile=0.05)
# ... (entrenar el pipeline con datos reales)

# ============================================
# 2. CARGAR EL MODELO PRE-ENTRENADO
# ============================================
model = tf.keras.models.load_model('path/to/turbina_model.h5')

# ============================================
# 3. CREAR EL "MOLDE" CON NUEVOS DATOS
# ============================================
nuevo_escenario = pd.DataFrame(columns=X_train.columns)
nuevo_escenario.loc[0] = 0.0  # Inicializar con ceros (formato float64)

# Inyectar los parámetros físicos reales
nuevo_escenario.loc[0, 'RPM'] = 850          # RPM de operación
nuevo_escenario.loc[0, 'Torque [Nm]'] = 0.25    # Torque medido
nuevo_escenario.loc[0, 'Angle'] = 75000      # Ángulo de incidencia

# ============================================
# 4. TRANSFORMAR CON EL PIPELINE
# ============================================
nuevo_transformado = Pipeline.transform(nuevo_escenario)

# ============================================
# 5. REALIZAR LA PREDICCIÓN
# ============================================
prediccion = model.predict(nuevo_transformado)
potencia_predicha = prediccion[0][0]

# ============================================
# 6. MOSTRAR RESULTADO
# ============================================
print("=" * 50)
print(" PREDICCIÓN DE POTENCIA")
print("=" * 50)
print(f"RPM:              {nuevo_escenario.loc[0, 'RPM']}")
print(f"Torque [Nm]:      {nuevo_escenario.loc[0, 'Torque [Nm]']}")
print(f"Angle:            {nuevo_escenario.loc[0, 'Angle']}")
print(f"\n🔌 POTENCIA PREDICHA: {potencia_predicha:.2f} Watts")
print("=" * 50)
```

### 🔍 Explicación Detallada

1. **DataFrame "Molde":** Se crea un DataFrame vacío con la misma estructura original, lleno de ceros.
2. **Inyección de Datos:** Los nuevos valores de RPM, Torque y Angle se inyectan en sus columnas correspondientes.
3. **Transformación Consistente:** El Pipeline (ya entrenado) transforma automáticamente los datos de la misma manera que los datos de training.
4. **Predicción:** El modelo predice la potencia basándose en los datos transformados.
5. **Sin Data Leakage:** Al usar el Pipeline pre-entrenado sin re-fitting, se garantiza que no hay fuga de información.

---

## 📚 Estructura del Repositorio

```
Despliegue-modelo-ANN-Turbinas-Hidraulicas/
├── README.md                          ← Este archivo
├── requirements.txt                   ← Dependencias del proyecto
├── dvc.yaml                          ← Configuración de DVC (Data Version Control)
├── .env                              ← Variables de entorno (no incluir en git)
├── .gitignore                        ← Archivos ignorados
│
├── scripts/
│   ├── scr/
│   │   ├── __init__.py
│   │   ├── cargar_datos.py           ← Extrae datos de Excel
│   │   ├── ft_engineering.py         ← Pipeline de preprocesamiento
│   │   └── LHS_Generator.py          ← Generador de diseños (DOE)
│   │
│   ├── MLP/
│   │   ├── MLP_Neuron.ipynb          ← Entrenamiento del modelo
│   │   ├── Params_MLP.ipynb          ← Ajuste de hiperparámetros
│   │   └── data_csv_LHS/             ← Datos de entrada
│   │
│   ├── GNN/
│   │   └── mesh_CAD.ipynb            ← Procesamiento de geometría
│   │
│   └── EDA_Analisis.ipynb            ← Análisis exploratorio

├── database/
│   ├── AFMP_Pruebas_curvas.xlsx.dvc  ← Referencia a datos (DVC)
│   ├── create_database.sql           ← Esquema SQL
│   └── data_scr/
│       └── AFMP_Pruebas curvas_Sheet1.csv

├── docs/
│   ├── architecture.md               ← Detalles técnicos
│   ├── DoE_Report.md                 ← Reporte de Diseño de Experimentos
│   ├── feature_store_report.md       ← Análisis de features
│   └── model_report.md               ← Resultados del modelo

└── CADS_Models/
    └── MESH/                         ← Archivos CAD/Mesh
```

---

## 🔧 Archivos Clave

### 📄 `cargar_datos.py`
Módulo de entrada que:
- Lee archivos Excel desde ruta especificada en `.env`
- Permite seleccionar hojas específicas (ej: '30Hz_P1')
- Maneja excepciones de hojas inexistentes

```python
from scripts.scr.cargar_datos import cargar_datos

df = cargar_datos('30Hz_P1')  # Retorna DataFrame con datos de pruebas
```

### 🔧 `ft_engineering.py`
Implementación de 3 funciones críticas:

1. **`fix_dataframe()`** - Limpieza estructural
   - Elimina filas vacías
   - Convierte a valores absolutos
   - Casting a tipos numéricos

2. **`ft_engineering()`** - Pipeline de features
   - Drop de columnas redundantes
   - Imputación de missing values
   - Manejo de outliers (Winsorizer)
   - Escalado de features (StandardScaler)

3. **`split_to_model()`** - Validación sin Data Leakage
   - Split train/test (80/20)
   - **SIN estratificación** (regresión, no clasificación)
   - Manejo automático de valores NaN en target

---

## ⚙️ Configuración del Modelo

### Arquitectura de la Red Neuronal
```
INPUT (3 features)
       ↓
  Dense(64, relu)
       ↓
  Dense(32, relu)
       ↓
  OUTPUT (1, Watts)

Loss Function:  Mean Squared Error (MSE)
Optimizer:      Adam
Batch Size:     32
Épocas:         150
Validation:     20% del training set
```

### Configuración Computacional
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Fuerza CPU
# Evita cuelgues de memoria GPU con datasets pequeños
# Optimiza TensorFlow para datos tabulares
```

---

## 📈 Métricas de Desempeño

El modelo se evalúa con:
- **MAE (Mean Absolute Error):** Error promedio en Watts
- **MSE (Mean Squared Error):** Penaliza errores grandes
- **Visualización:** Predicciones vs Valores Reales

Todas las métricas se calculan en el **conjunto de prueba (test set)** para garantizar une estimación insesgada del desempeño.

---

## 🐛 Troubleshooting

### ❌ Error: `KeyError: 'Column not found'`
**Causa:** El nombre de la columna en el Excel cambió.  
**Solución:** Actualiza los nombres en `cargar_datos.py` o `ft_engineering.py`

```python
# Verifica las columnas disponibles
print(df.columns.tolist())
```

### ❌ Error: `LossySetitemError`
**Causa:** Tipo de dato incompatible (ej: string en columna numérica).  
**Solución:** Usa `clean_dataframe()` con el tipo correcto:

```python
df_fixed = fix_dataframe(df, num_feats=['RPM', 'Torque [Nm]'], negative_feats=['RPM'])
```

### ❌ Error: `ValueError: stratify=True no permitido en regresión`
**Causa:** Estratificación activa en split de regresión.  
**Solución:** Usa `stratify=False`:

```python
X_train, X_test, y_train, y_test = split_to_model(df, target='Power [Watts]', stratify=False)
```

### ⚠️ Advertencia: `CUDA_VISIBLE_DEVICES` no encontrado
**Causa:** Sistema sin GPU NVIDIA.  
**Solución:** TensorFlow automáticamente usa CPU (comportamiento esperado).

---

## 📖 Documentación Adicional

Consulta los siguientes archivos para detalles profundos:

- **[architecture.md](docs/architecture.md)** - Detalles técnicos y diagrama de datos
- **[feature_store_report.md](docs/feature_store_report.md)** - Análisis exploratorio de features
- **[model_report.md](docs/model_report.md)** - Resultados completos del modelo
- **[DoE_Report.md](docs/DoE_Report.md)** - Diseño de Experimentos

---

## 🤝 Contribuciones

Este proyecto está abierto a mejoras. Para contribuir:

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/mejora`)
3. Commit cambios (`git commit -m "Descripción clara"`)
4. Push a la rama (`git push origin feature/mejora`)
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto está bajo la licencia **MIT**. Consulta [LICENSE](LICENSE) para más detalles.

---

## 📧 Contacto & Soporte

¿Preguntas o problemas?  
- 📧 **Email:** juancvanegas216@gmail.com
- 🐛 **Issues:** [GitHub Issues](https://github.com/JuanCaVa21/Despliegue-modelo-ANN-Turbinas-Hidraulicas/issues)
- 📚 **Documentación Técnica:** Ver carpeta `/docs`

---

## 🙏 Agradecimientos

- **TensorFlow/Keras Team** por framework robusto
- **Scikit-learn & Feature-Engine** por herramientas de calidad industrial
- **Laboratorio de Turbinas Hidráulicas** por datos experimentales

---

**Última actualización:** Marzo 2026  
**Versión:** 1.0.0  
