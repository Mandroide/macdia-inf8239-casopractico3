import os

import pandas as pd

# ==================== CARGA Y PREPROCESAMIENTO ====================
def load_all_data(data_dir: str | os.PathLike[str] = "data"):
    """Carga TODOS tus archivos CSV de tweets"""
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    dfs = []
    for file in all_files:
        if 'tweets_trafico' in file:
            path = os.path.join(data_dir, file)
            df = pd.read_csv(
                path,
                dtype={'tweet_id': str, 'texto': str, 'ubicacion_geo': str},
                parse_dates=['fecha'],
                on_bad_lines='skip'
            )
            dfs.append(df)
            print(f"Cargado {file}: {len(df)} registros")
    return pd.concat(dfs, ignore_index=True)


def preprocess_data(df: pd.DataFrame):
    """Prepara features para ML"""
    df['texto'] = df['texto'].astype(str).fillna('')
    df['ubicacion_geo'] = df['ubicacion_geo'].astype(str).fillna('')

    # 1. Limpiar texto
    df['texto_clean'] = (
        df['texto']
        .str.lower()
        .str.replace(r'http\S+|www\S+|https\S+', '', regex=True)
        .str.replace(r'[^a-záéíóúñü ]', ' ', regex=True)
        .str.strip()
    )

    df['texto_len'] = df['texto_clean'].str.len()
    df['hora'] = pd.to_datetime(df['fecha'], errors='coerce').dt.hour
    df['dia_semana'] = pd.to_datetime(df['fecha'], errors='coerce').dt.dayofweek

    df['en_sd'] = df['ubicacion_geo'].str.contains('Santo Domingo', case=False, na=False).astype(int)
    df['en_santiago'] = df['ubicacion_geo'].str.contains('Santiago', case=False, na=False).astype(int)

    df['palabras_graves'] = (
        df['texto_clean']
        .str.contains('muerto|herido|grave|fatal|accidente', case=False, na=False)
        .astype(int)
    )

    return df