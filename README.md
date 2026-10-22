# TikTok Profile Scraper (público)

Proyecto simple en Python para extraer información **pública** de una cuenta de TikTok y guardarla en JSON.

## Qué obtiene

- Datos del perfil: username, nickname, bio, verificación y métricas generales.
- Lista de videos visibles: descripción, fecha, vistas, likes, comentarios, compartidos,
  guardados, duración, portada, música, autor y hashtags.
- Clasificación del tipo de contenido, hashtags principales, acumulado de reacciones,
  muestra de comentarios públicos, usuarios comentaristas y sentimiento léxico básico.
- En modo Playwright se realizan desplazamientos automáticos para cargar más publicaciones.
- Si TikTok no incluye la lista de publicaciones en el JSON, se recuperan los enlaces
  y descripciones de las tarjetas visibles en el DOM de Chrome.
- Análisis exploratorio: indicadores Big Five, extracción de entidades, agrupación semántica
  mediante embeddings y gráficos en `output/analysis/`.

## Requisitos

- Python 3.10+

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m playwright install chromium
```

## Uso

```bash
python main.py usuario_tiktok
```

También puedes pasar la URL completa del perfil:

```bash
python main.py https://www.tiktok.com/@usuario_tiktok
```

Con salida personalizada:

```bash
python main.py usuario_tiktok --output output/mi_cuenta.json
```

Al finalizar se imprime una tabla Markdown con los resultados y se generan:

- `output/analysis/analysis.json` y `results.csv` con los resultados reproducibles.
- `big_five_radar.png`, `topic_distribution.png` e `interest_map.png`.
- `report.html`, una página web ordenada como tabla de datos, gráfico de barras
  Big Five y mapa de intereses/conocimiento temático.
- `content_types.png` y `reactions.png` para complementar el análisis de temas y
  reacciones.

El Big Five es un indicador léxico exploratorio del contenido público, no una evaluación
psicológica. Los embeddings multilingües usan `sentence-transformers`; si no está disponible,
se usa un embedding TF-IDF como fallback.

Para guardar solo el perfil sin análisis:

```bash
python main.py usuario_tiktok --skip-analysis
```

Si TikTok bloquea por WAF, el scraper intenta Playwright automáticamente:

```bash
python main.py usuario_tiktok --use-playwright
```

La bandera anterior fuerza Playwright desde el inicio, pero ya no es necesaria
cuando el bloqueo se detecta. Si aparece un desafío visual, abre el navegador:

```bash
python main.py usuario_tiktok --headful --wait-seconds 25
```

Para abrir navegador visible y resolver desafío/login manual:

```bash
python main.py usuario_tiktok --use-playwright --headful --wait-seconds 20
```

En modo `--headful` se abre el Google Chrome instalado en el equipo, no un perfil
temporal de Chromium. La sesión se conserva en `output/chrome-profile`; inicia
sesión en TikTok dentro de esa ventana y reutilízala en las siguientes ejecuciones.
No abras simultáneamente otra instancia usando ese mismo perfil.

Para reutilizar sesión/cookies entre ejecuciones (recomendado):

```bash
python main.py usuario_tiktok --use-playwright --headful --wait-seconds 25 --session-state output/tiktok_session_state.json
```

Después de esa primera ejecución, puedes volver a correr en headless con la misma sesión:

```bash
python main.py usuario_tiktok --use-playwright --session-state output/tiktok_session_state.json
```

## Nota importante

- Este proyecto está diseñado para páginas públicas.
- TikTok puede cambiar su HTML o aplicar bloqueos (WAF). En ese caso se intenta
  Playwright automáticamente y se puede completar el desafío de forma manual con `--headful`.
- Los comentarios individuales solo se extraen cuando TikTok los expone públicamente
  en el HTML/JSON recibido; no se accede a endpoints privados ni se elude autenticación.
- Respeta siempre términos de servicio y legislación aplicable al uso de datos.
