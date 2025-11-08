import pathlib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from src.trafficincidentdetection.config import Settings
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import os
from tokenizers import spacy_tokenizer
from preprocess import preprocess_data, load_all_data


pd.set_option('future.no_silent_downcasting', True)

# ==================== ENTRENAMIENTO ====================

def train_model(df: pd.DataFrame):
    """Entrena el modelo multi-tarea"""

    # Features de texto
    vectorizer = TfidfVectorizer(
        max_features=1000,
        tokenizer=spacy_tokenizer,
        lowercase=True
    )
    text_features = vectorizer.fit_transform(df['texto_clean']).toarray()

    # Features numéricas + geo
    num_features = df[['texto_len', 'hora', 'dia_semana', 'palabras_graves']].values

    # Features finales
    X = np.hstack([text_features, num_features])

    # Targets (multi-output)
    le_tipo = LabelEncoder()
    le_sev = LabelEncoder()

    y_tipo = le_tipo.fit_transform(df['incidente_tipo'].fillna('desconocido'))
    y_sev = le_sev.fit_transform(df['severidad'].fillna('baja'))

    Y = np.column_stack([y_tipo, y_sev])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y[:, 0]
    )

    # Modelo XGBoost Multi-Output
    model = MultiOutputClassifier(
        xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
    )

    model.fit(X_train, y_train)

    # Evaluación
    y_pred = model.predict(X_test)
    acc_tipo = accuracy_score(y_test[:, 0], y_pred[:, 0])
    acc_sev = accuracy_score(y_test[:, 1], y_pred[:, 1])

    print(f"🎯 PRECISIÓN TIPO: {acc_tipo:.1%}")
    print(f"🎯 PRECISIÓN SEVERIDAD: {acc_sev:.1%}")
    print("\n📊 Reporte Detallado:")
    print(classification_report(y_test[:, 0], y_pred[:, 0],
                                target_names=le_tipo.classes_))

    return model, vectorizer, le_tipo, le_sev


# ==================== GUARDAR MODELO ====================

def save_model(model, vectorizer, le_tipo, le_sev):
    joblib.dump({
        'model': model,
        'vectorizer': vectorizer,
        'le_tipo': le_tipo,
        'le_sev': le_sev
    }, pathlib.Path(__file__).parents[2] / 'models/traffic_predictor_v1.pkl')
    print("💾 Modelo guardado: traffic_predictor_v1.pkl")


def main():
    os.makedirs(pathlib.Path(__file__).parents[2]/"models", exist_ok=True)

    print("🚀 ENTRENANDO MODELO DE INCIDENTES VIALES")
    print("=" * 50)
    settings = Settings()
    # 1. Cargar datos
    df = load_all_data(settings.raw_data_dir_path)
    print(f"\n📊 Total registros: {len(df)}")

    # 2. Preprocesar
    df = preprocess_data(df)

    # 3. Entrenar
    model, vectorizer, le_tipo, le_sev = train_model(df)

    # 4. Guardar
    save_model(model, vectorizer, le_tipo, le_sev)


# ==================== EJECUTAR ====================

if __name__ == "__main__":
    main()
