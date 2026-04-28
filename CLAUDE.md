# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end deep learning pipeline for predicting power output (Watts/kW) from Banki-type hydraulic turbines using sensor data collected in laboratory conditions. The pipeline covers data loading, feature engineering, MLP training, and Optuna-based hyperparameter optimization.

## Development Commands

```bash
# Activate virtual environment
source .venv/bin/activate      # Linux/Mac
# or
source .ANN_Env/bin/activate   # Legacy env name

# Install dependencies
pip install -r requirements.txt

# Run DVC pipeline stages
dvc repro                      # Run full pipeline
dvc run -n carga_y_limpieza    # Run specific stage

# Launch notebooks
jupyter notebook scripts/MLP/MLP_Neuron.ipynb
jupyter notebook scripts/EDA_Analisis.ipynb

# Environment setup
cp .env.example .env           # Configure DB_USER (Excel path), DB_DATABASE, DB_PASSWORD, DB_HOST, DB_PORT
```

## Architecture & Data Flow

```
Excel / Supabase → cargar_datos.py → ft_engineering.py → MLP training → Optuna tuning → Predictions
```

### Core Modules (`scripts/scr/`)

- **`cargar_datos.py`**: Loads data from Excel sheets or Supabase/PostgreSQL. Uses `.env` for credentials. `DB_USER` holds the path to the Excel file.
- **`ft_engineering.py`**: Scikit-learn + feature-engine preprocessing pipeline. Key design: fit only on training data, reuse for inference via "DataFrame molding" (inject new values into a blank DataFrame with the same schema as training data, then transform with the pre-fitted pipeline — never refit).
- **`LHS_Generator.py`**: Latin Hypercube Sampling for Design of Experiments; generates parameter space CSV for turbine simulations.

### Training Notebooks (`scripts/MLP/`)

- **`MLP_Neuron.ipynb`**: Main notebook — contains both the base model and the Optuna-optimized model. Runs CPU-only (`CUDA_VISIBLE_DEVICES = '-1'`). Has a hardcoded path (`/home/asus_juan/...`) that must be updated per machine.
- **`Params_MLP.ipynb`**: Standalone hyperparameter search.

### DVC Pipeline (`dvc.yaml`)

Two stages: `carga_y_limpieza` → `datos_limpios.csv` → `feature_engineering` → `datos_features.csv`. Raw Excel data is tracked via DVC (`.dvc` pointer files), not committed directly.

## Current Optimized Model

```
Input (6 features)
→ Dense(96, tanh) → Dropout(0.003)
→ Dense(48, swish) → Dropout(0.017)
→ Dense(1, linear)   # Output: Power in kW
```

- Optimizer: Adam (lr=0.0026), Loss: MSE, Metric: MAE
- EarlyStopping patience=20, max 200 epochs, batch_size=32
- Achieved ~0.33 kW MAE (72% improvement over base model's 2.23 W)

## Feature Engineering Decisions

- **MinMaxScaler** over StandardScaler — data is bimodal, not Gaussian
- **Winsorization at 95th percentile** for outlier handling
- **Median imputation** for missing values
- Drop highly correlated features (Caudal/Presión have strong correlation)

## Environment Variables (`.env`)

| Variable | Description |
|---|---|
| `DB_USER` | Path to the Excel file (`AFMP_Pruebas_curvas.xlsx`) |
| `DB_DATABASE` | Supabase database name (`PAT_Turbinas`) |
| `DB_PASSWORD` | Supabase password |
| `DB_HOST` | Supabase host (`<project>.supabase.co`) |
| `DB_PORT` | PostgreSQL port (default `5432`) |

## Known Issues

- `scripts/scr/normalize_data.py` is an empty placeholder
- `scripts/MLP/MLP_Neuron.ipynb` has a hardcoded absolute path — update to relative or use `.env`
- `scripts/scr/` lacks `__init__.py` — import as a package requires adding it or using `sys.path`
- Database schema (`database/create_database.sql`) has a typo: `experiemts` → `experiments`
