import joblib
import pandas as pd
import numpy as np
import pathlib
import datetime

# Cargar modelo
pipeline = joblib.load(pathlib.Path(__file__).parents[2]/'models/traffic_predictor_v1.pkl')


def predict_incident(texto: str,
                     fecha: float | str | datetime.datetime | datetime.date = None) -> dict:
    """Predice tipo y severidad de un tweet"""

    # Preprocesar
    texto_clean = (
        texto.lower()
        .replace(r'http\S+|www\S+|https\S+', '')
        .replace(r'[^a-záéíóúñü ]', ' ')
        .strip()
    )
    texto_len = len(texto_clean)
    hora = pd.to_datetime(fecha or '2025-01-01', errors='coerce').hour if fecha else 12
    dia_semana = pd.to_datetime(fecha or '2025-01-01', errors='coerce').dayofweek if fecha else 0

    palabras_graves = 1 if any(p in texto_clean for p in ['muerto', 'herido', 'grave', 'fatal', 'accidente']) else 0

    # Vectorizar
    text_features = pipeline['vectorizer'].transform([texto_clean]).toarray()
    num_features = np.array([[texto_len, hora, dia_semana, palabras_graves]])
    X = np.hstack([text_features, num_features])

    # Predecir
    model = pipeline['model']
    pred = model.predict(X)[0]  # [tipo_id, severidad_id]

    probs_per_task = model.predict_proba(X)  # Lista: [probs_tipo, probs_severidad]

    # Asegurarse de que hay 2 clasificadores
    if len(probs_per_task) != 2:
        raise ValueError("Se esperaban 2 tareas en MultiOutputClassifier")

    prob_tipo = probs_per_task[0][0]   # Probabilidades del tipo (para este ejemplo)
    prob_sev = probs_per_task[1][0]    # Probabilidades de severidad

    tipo = pipeline['le_tipo'].inverse_transform([pred[0]])[0]
    severidad = pipeline['le_sev'].inverse_transform([pred[1]])[0]

    return {
        'tipo_incidente': tipo,
        'severidad': severidad,
        'confianza_tipo': float(prob_tipo.max()),
        'confianza_severidad': float(prob_sev.max())
    }


# TEST
print(predict_incident("🚨 ACCIDENTE GRAVE en Autopista Duarte km 45! Heridos reportados"))
# → {'tipo_incidente': 'accidente', 'severidad': 'alta', 'confianza_tipo': 0.92}