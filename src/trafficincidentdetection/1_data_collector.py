import pathlib

import tweepy
import pandas as pd
from src.trafficincidentdetection.config import Settings
from datetime import datetime
import os
import logging
import re

# Cargar configuración
settings = Settings()

# Carpetas
output_dir = settings.raw_data_dir_path
logs_dir = pathlib.Path(__file__).parents[2] / "logs"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Archivo de salida con timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(output_dir, f"tweets_trafico_rd_{timestamp}.csv")
seen_ids_file = os.path.join(output_dir, "seen_tweet_ids.txt")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / "traffic_collection.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Cargar IDs procesados
seen_ids = set()
if os.path.exists(seen_ids_file):
    with open(seen_ids_file, 'r', encoding='utf-8') as f:
        seen_ids = {line.strip() for line in f if line.strip()}
logger.info(f"IDs procesados previos: {len(seen_ids)}")

# ==================== NLP: Patrones de Incidentes ====================

# Tipos de incidentes
INCIDENTES = {
    'accidente': ['accidente', 'choque', 'colisión', 'volcadura'],
    'tapón': ['tapón', 'congestion', 'trancón', 'tráfico lento', 'tráfico denso'],
    'bloqueo': ['vía bloqueada', 'carretera cerrada', 'no pasa', 'bloqueo'],
    'lluvia': ['lluvia fuerte', 'inundación', 'pavimento mojado'],
    'obstrucción': ['árbol caído', 'cable', 'piedra', 'escombro'],
    'manifestación': ['piquete', 'protesta', 'marcha']
}

# Vías principales de RD
VIAS_PRINCIPALES = [
    'autopista duarte', 'autopista las américas', 'autopista 6 de noviembre',
    'avenida 27 de febrero', 'winston churchill', 'nunez de caceres',
    'kennedy', 'lope de vega', 'sanchez', 'independencia',
    'autopista san isidro', 'coral', 'carretera mella'
]

# Ciudades
CIUDADES = ['santo domingo', 'santiago', 'la vega', 'san cristóbal', 'puerto plata', 'la romana']

# Patrón para km
PATRON_KM = r'(km|kilómetro)[\s]*[\d]+'


def detectar_incidente(texto):
    """Detecta tipo de incidente y severidad"""
    texto_lower = texto.lower()
    tipo = 'desconocido'
    severidad = 'baja'

    for cat, palabras in INCIDENTES.items():
        for palabra in palabras:
            if palabra in texto_lower:
                tipo = cat
                if any(p in texto_lower for p in ['muerto', 'herido', 'grave', 'fatal']):
                    severidad = 'alta'
                elif any(p in texto_lower for p in ['lento', 'congestion', 'reten']):
                    severidad = 'media'
                else:
                    severidad = 'media' if cat in ['accidente', 'bloqueo'] else 'baja'
                break
        if tipo != 'desconocido':
            break

    return tipo, severidad


def extraer_ubicacion(texto):
    """Extrae ubicaciones mencionadas en el texto"""
    texto_lower = texto.lower()
    ubicaciones = []

    for via in VIAS_PRINCIPALES:
        if via in texto_lower:
            ubicaciones.append(via.title())
    for ciudad in CIUDADES:
        if ciudad in texto_lower:
            ubicaciones.append(ciudad.title())
    km_match = re.search(PATRON_KM, texto_lower)
    if km_match:
        ubicaciones.append(km_match.group(0).upper())
    if any(p in texto_lower for p in ['frente a', 'cerca de', 'entrada', 'salida']):
        ubicaciones.append("Referencia cercana")

    return ', '.join(ubicaciones) if ubicaciones else None


# ==================== Recolección ====================

tweets = []
new_ids = []

# Cliente con manejo automático de rate limits
client = tweepy.Client(bearer_token=settings.x_bearer_token, wait_on_rate_limit=True)
# --- Consulta optimizada ---
query = (
    '(tapón OR tráfico OR accidente OR choque OR colisión OR volcadura OR '
    '"vía bloqueada" OR "lluvia fuerte" OR trancón OR "carretera mella" OR '
    'kennedy OR duarte OR "las américas" OR coral OR santiago OR '
    '"santo domingo" OR "san cristóbal" OR '
    'avenida OR autopista OR km OR "carretera" OR paso OR retención OR'
    '"27 febrero" OR "nunez caceres" OR "lope vega" OR "6 noviembre") '
    'lang:es '
    '-is:retweet '  # IMPORTANTE: evita duplicados
)
try:
    logger.info(f"🔍 Query: {query}")
    logger.info("Iniciando recolección de tweets...")
    for tweet in tweepy.Paginator(
            client.search_recent_tweets,
            query=query,
            tweet_fields=['text', 'created_at', 'geo', 'lang', 'public_metrics'],
            max_results=100
    ).flatten(limit=5000):  # Límite razonable para Basic Access

        tweet_id = str(tweet.id)
        if tweet_id in seen_ids:
            continue  # Saltar duplicados

        # --- NLP ---
        tipo_incidente, severidad = detectar_incidente(tweet.text)
        ubicacion_texto = extraer_ubicacion(tweet.text)

        # --- Geolocalización ---
        lat, lon = None, None
        if tweet.geo and tweet.geo.get('coordinates'):
            coords = tweet.geo['coordinates']['coordinates']
            lon, lat = coords[0], coords[1]

        tweets.append({
            'tweet_id': tweet_id,
            'texto': tweet.text,
            'fecha': tweet.created_at,
            'incidente_tipo': tipo_incidente,
            'severidad': severidad,
            'ubicacion_texto': ubicacion_texto,
            'url': f"https://twitter.com/i/web/status/{tweet_id}"
        })
        new_ids.append(tweet_id)

        if len(tweets) % 50 == 0:
            logger.info(f"Procesados {len(tweets)} tweets...")

except Exception as e:
    logger.error(f"Error en recolección: {e}")
finally:
    # --- Guardar ---
    if tweets:
        df = pd.DataFrame(tweets)
        df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Guardados {len(df)} tweets en {output_file}")

        # Actualizar IDs
        with open(seen_ids_file, 'a', encoding='utf-8') as f:
            for tid in new_ids:
                f.write(tid + '\n')
        logger.info(f"IDs nuevos registrados: {len(new_ids)}")
    else:
        logger.info("No se encontraron tweets nuevos.")

logger.info("Recolección completada.")
