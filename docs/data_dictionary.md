# Diccionario de datos

| Columna           | Tipo     | Descripción                  | Ejemplo                                               |
|-------------------|----------|------------------------------|-------------------------------------------------------|
| `tweet_id`        | str      | Identificador unico de tweet | `1849274612345679872`                                 |
| `texto`           | str      | Texto completo de tweet      | `"Accidente en la Autopista Duarte km 12 #traficoRD"` |
| `fecha`           | datetime | Fecha de creacion en UTC     | `2025-10-29 22:48:12`                                 |
| `incidente_tipo`  | str      | Tipo de incidente            | `accidente`                                           |
| `severidad`       | str      | Severidad del incidente      | `alta`                                                |
| `ubicacion_texto` | str      | Ubicación del incidente      | `Santo Domingo`                                       |
| `url`             | str      | URL del Tweet                | `https://twitter.com/i/web/status/{tweet_id}`         |
