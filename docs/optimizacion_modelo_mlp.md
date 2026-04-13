# Evolución y Optimización del Modelo MLP para Predicción de Potencia

**Versión:** 2.0.0  
**Fecha:** Abril 2026

---

## 1. Introducción

El objetivo central de este documento es registrar la evolución del modelo de Deep Learning (MLP) para la predicción de potencia generada (en kW) por una turbina hidráulica tipo Banki. El modelo inicial utilizaba como variables de entrada `Torque`, `RPM`, `Caudal` y `Presion` para estimar la potencia.

Si bien el rendimiento inicial fue prometedor, el Análisis Exploratorio de Datos (EDA) y una revisión del pipeline de `Feature Engineering` revelaron oportunidades críticas de mejora para incrementar la precisión y la robustez del sistema predictivo.

## 2. Hallazgos Clave del Análisis Exploratorio (EDA)

El análisis profundo de los datos de entrada arrojó dos hallazgos principales que guiaron la refactorización del modelo:

*   **Distribuciones No Normales:** El uso de gráficos `Raincloud` para visualizar la distribución de las variables de entrada mostró que características clave como `RPM` y `Torque` no seguían una distribución Gaussiana. En su lugar, presentaban comportamientos **bimodales**, lo cual puede afectar negativamente el rendimiento de técnicas de escalado estándar como `StandardScaler`.

*   **Redundancia de Datos y Multicolinealidad:** Un `Heatmap` de correlación entre las variables de entrada identificó una alta correlación positiva entre `Caudal` y `Presion`. Esta multicolinealidad es redundante, introduce ruido en el modelo y puede desestabilizar el proceso de aprendizaje de los pesos de la red neuronal.

## 3. Refactorización del Pipeline de Feature Engineering

Con base en los hallazgos del EDA, se implementaron tres cambios estratégicos en el pipeline de preprocesamiento (`ft_engineering.py`).

### 3.1. Migración a `MinMaxScaler`

Debido a la naturaleza no normal de los datos, se reemplazó `StandardScaler` por `MinMaxScaler`. Este último escala los datos a un rango fijo (generalmente [0, 1]) sin asumir una distribución específica, lo que lo hace más adecuado para las distribuciones bimodales observadas y, en general, más robusto para las redes neuronales.

El cambio se implementó dentro del `ColumnTransformer` del pipeline, como se muestra en el siguiente fragmento de `ft_engineering.py`:

```python
preprocessor_sk = ColumnTransformer(
    transformers=[
        ('numerical', MinMaxScaler(), var_num if var_num else []),
        # ...
    ],
    remainder='passthrough'
)
```

### 3.2. Eliminación de Variables Redundantes

Para mitigar el impacto de la multicolinealidad, se decidió eliminar las variables `Caudal` y `Presion`, que aportaban información redundante. El pipeline se configuró para descartar estas columnas de forma programática utilizando `DropFeatures` de la librería `feature-engine`.

```python
steps = []
if drop_var:
    steps.append(('drop_features', DropFeatures(features_to_drop=drop_var)))
```

### 3.3. División de Datos Estratificada por Experimento

Una división aleatoria simple (`train_test_split`) puede provocar fuga de datos (`data leakage`) si mediciones de un mismo experimento se distribuyen entre los conjuntos de entrenamiento y prueba. Para una validación más robusta que simule condiciones de producción reales, la estrategia de división debe asegurar que todos los datos de un `ID_Experimento` específico permanezcan en un único conjunto (ya sea `train` o `test`).

Esto se logra utilizando un enfoque de validación cruzada por grupos, como `GroupShuffleSplit` de Scikit-learn, que garantiza la integridad de los grupos de datos.

## 4. Proceso de Optimización con Optuna

Para encontrar la arquitectura de red neuronal y los hiperparámetros óptimos, se utilizó **Optuna**, una librería de optimización automática. Se implementó una búsqueda bayesiana (sampler `TPE`) para explorar eficientemente un espacio de búsqueda complejo.

Los hiperparámetros optimizados incluyeron:
*   Número de capas densas.
*   Número de neuronas por capa.
*   Funciones de activación (`tanh`, `swish`, `relu`).
*   Tasa de aprendizaje (`learning rate`) del optimizador Adam.

A continuación, un ejemplo del `objective` function utilizado en el estudio de Optuna:

```python
import optuna
import tensorflow as tf

def objective(trial):
    # Definir el espacio de búsqueda de hiperparámetros
    n_layers = trial.suggest_int('n_layers', 1, 3)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(X_train.shape,)))

    for i in range(n_layers):
        num_hidden = trial.suggest_int(f'n_units_l{i}', 32, 128, log=True)
        activation = trial.suggest_categorical(f'activation_l{i}', ['relu', 'tanh', 'swish'])
        model.add(tf.keras.layers.Dense(num_hidden, activation=activation))
        
    model.add(tf.keras.layers.Dense(1)) # Capa de salida para regresión

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_absolute_error'
    )

    # Entrenar y evaluar el modelo
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, verbose=0)
    val_mae = model.evaluate(X_val, y_val)
    
    return val_mae

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

## 5. Comparativa de Resultados

La refactorización del pipeline y la optimización con Optuna resultaron en una mejora sustancial del rendimiento del modelo, reduciendo el error de predicción en un **72%**.

| Métrica                  | Modelo Base (Antes) | Modelo Optimizado (Después) | Mejora      |
|--------------------------|---------------------|-----------------------------|-------------|
| **Error Absoluto Medio (MAE)** | 1.19 kW             | **0.33 kW**                 | **-72.27%** |

## 6. Arquitectura Final del Modelo

El estudio de Optuna convergió en una arquitectura de red neuronal significativamente más precisa y robusta.

*   **Capa de Entrada:** 3 características (`RPM`, `Torque [Nm]`, `Angle`).
*   **Capa Densa 1:**
    *   **Unidades:** 96
    *   **Activación:** `tanh`
*   **Capa Densa 2:**
    *   **Unidades:** 48
    *   **Activación:** `swish`
*   **Capa de Salida:**
    *   **Unidades:** 1 (predicción de Potencia en kW)
    *   **Activación:** Lineal
*   **Optimizador:** Adam con la tasa de aprendizaje óptima encontrada por Optuna.
*   **Función de Pérdida:** Error Absoluto Medio (MAE).