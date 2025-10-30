# Caso práctico 3 – Solución Agéntica NLP (UASD)

Un sistema de NLP para monitorear en tiempo real plataformas publicaciones y que sea capaz de identificar y
clasificar publicaciones que describen incidentes de tráfico, entender su ubicación geográfica (geolocalización) y
evaluar su severidad a partir del lenguaje utilizado

## Arquitectura
Datos Crudos → Preprocesamiento → Features → XGBoost/LSTM → Predicciones
       ↓              ↓              ↓           ↓           ↓
   Tweets       TF-IDF +       Texto + Geo +   Ensemble     API REST
              Embeddings      Time Series

## Estructura básica

```
caso_practico_3_nlp_uasd/                  # ── project root ────────────────────────────
├── data/                       # ── datasets --------------------------------
│   ├── raw/                    # directory with raw dataset files
├── logs/                  
├── models/                 
├── notebooks/                  # Jupyter EDA
│   └── EDA.ipynb
├── src/trafficincidentdetection/              # ── importable package ----------------------
│   ├── __init__.py
│   ├── 1_data_collector.py         # collect data from Twitter API (minimum Basic Access)
│   ├── 2_model_trainer.py         # train NLP model
│   ├── 3_model_predictor.py       # predict traffic incident
│   ├── __init__.py
│   ├── config.py                   # env handling (pydantic)
│   ├── preprocess.py               # functions to load and preprocess raw data
│   ├── tokenizers.py               # env handling (pydantic)
├── .env.example                # environment variables template (see below)
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── pyproject.toml              # UV deps & tooling
├── README.md
├── uv.lock                     # Locked uv deps

```

## Quick start

**Before starting, you must install [UV](https://docs.astral.sh/uv/getting-started/installation/)**

### Boostrap the app

```bash
git clone <repo-url>
cd macdia-inf8239-casopractico3

# Create and activate a virtual environment
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r pyproject.toml
```

## Como correrlo
```shell
# 1. Corre tu colector (genera datos)
python 1_data_collector.py

# 2. Entrena (solo UNA vez)
uv run python -m spacy download es_core_news_sm
python 2_model_trainer.py

# 3. Predice incidencia de transito de tweets
python 3_model_predictor.py
```
