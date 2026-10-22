# TikTok Profile Scraper (contenido público)

Herramienta de línea de comandos en Python para recopilar la información que TikTok
expone públicamente en un perfil y generar un informe exploratorio de sus videos. Los
datos se guardan localmente en JSON, CSV, HTML y gráficos.

> El análisis describe únicamente el contenido público extraído. Las señales semánticas
> Big Five no son una prueba psicológica ni permiten determinar la personalidad real de
> la persona. El mapa temático muestra qué temas aparecen en los videos observados; no
> mide cuánto sabe la persona sobre esos temas.

## Contenido del análisis

### Perfil y publicaciones

Cuando TikTok los expone, se recopilan:

- Nombre de usuario, nickname, biografía, avatar, enlace, idioma, región y verificación.
- Métricas del perfil: seguidores, seguidos, likes y número total de videos.
- Publicaciones visibles: descripción, fecha, visualizaciones, likes, comentarios,
  compartidos, guardados, duración, portada, música, autor y hashtags.
- Comentarios públicos disponibles en el HTML/JSON de las páginas consultadas.

La extracción puede ser parcial: TikTok puede entregar solo las publicaciones que cargó
en la página. Por ejemplo, el perfil puede indicar 397 videos mientras que la sesión solo
permite recopilar 27. El informe indica cuántos se esperaban y cuántos se analizaron; no
interpreta los videos no extraídos.

### Señales Big Five

El análisis crea un embedding multilingüe para el texto de cada video disponible. Usa
descripción, hashtags y título de música, excluyendo títulos genéricos de TikTok. Después
compara esos embeddings con descripciones semánticas de ambos polos de cada dimensión:

- **Apertura:** curiosidad, creatividad e interés por ideas o experiencias nuevas frente
  a preferencia por lo familiar y rutinario.
- **Responsabilidad:** planificación, disciplina y constancia frente a improvisación y
  desorganización.
- **Extraversión:** energía e interacción social frente a reserva y preferencia por la
  soledad.
- **Amabilidad:** empatía, cooperación y consideración frente a confrontación y
  desconfianza.
- **Neuroticismo:** preocupación, tensión e inseguridad frente a calma y estabilidad.

La similitud relativa con ambos polos produce una señal por dimensión. Para facilitar
la lectura conjunta, las cinco señales se normalizan como porcentajes que suman
exactamente **100 %** (con redondeo a una décima). Por tanto, son proporciones relativas
dentro de este análisis: **no** son cinco puntuaciones independientes, percentiles,
probabilidades ni un diagnóstico. Un porcentaje mayor solo indica mayor peso relativo
de esa dimensión frente a las otras en el texto disponible.

Para cada dimensión, el sistema promedia la similitud semántica de los videos con sus dos
descripciones y obtiene una señal bruta entre 0 y 100. Después normaliza las cinco señales
brutas para formar la distribución porcentual. El informe también muestra el total de
palabras analizables, la cantidad de videos de evidencia y un indicador de suficiencia:
`min(1, palabras / 250) × min(1, videos / 50)`. Se requiere además un mínimo de 20 palabras
y una confianza de al menos 0,5; si no se alcanza, las puntuaciones figuran como
«No concluyente». Este indicador es una regla interna de cobertura textual, no una
probabilidad de que la estimación sea correcta. Aun con evidencia suficiente, la
interpretación sigue limitada al contenido público recopilado.

### Intereses y mapa temático

Los embeddings de los videos se agrupan con K-Means. Para cada grupo, el análisis muestra
los términos más representativos y calcula:

`porcentaje del tema = videos asignados al grupo / videos con texto analizable × 100`

Los porcentajes se redondean a una décima y se ajustan para sumar **100 %**. El mapa
`interest_map.png` presenta barras con el porcentaje y número de videos por tema; el
informe incluye además una tabla con esos datos. Los términos son etiquetas automáticas
del contenido (por ejemplo, frases, artistas, letras o nostalgia). El porcentaje mide
la distribución de las publicaciones entre los temas, no el conocimiento ni el nivel
de experiencia de quien publica.

Otros indicadores del informe incluyen hashtags frecuentes, tipos de contenido,
reacciones acumuladas, una muestra de comentarios públicos y un análisis léxico básico
del sentimiento de esos comentarios. El sentimiento se calcula sobre los comentarios,
no sobre la personalidad del autor.

## Requisitos

- Python 3.10 o posterior.
- Google Chrome instalado para el modo de navegador visible (`--headful`).
- Conexión a Internet en el primer análisis para descargar el modelo multilingüe de
  `sentence-transformers` y sus archivos. El modelo queda en caché para ejecuciones
  posteriores.

## Instalación

Desde la carpeta del proyecto, en Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m playwright install chromium
```

Si usas el símbolo del sistema (`cmd.exe`), activa el entorno con:

```bat
.venv\Scripts\activate.bat
```

## Ejecución

Con un nombre de usuario (con o sin `@`):

```bash
python main.py usuario_tiktok
```

También puedes indicar la URL del perfil:

```bash
python main.py https://www.tiktok.com/@usuario_tiktok
```

El comando descarga los datos, los guarda en `output/profile.json`, genera el análisis
y muestra en la terminal una tabla con las señales Big Five y los temas. Si quieres
guardar el JSON en otra ruta:

```bash
python main.py usuario_tiktok --output output/mi_cuenta.json
```

Para guardar solo el perfil, sin análisis ni gráficos:

```bash
python main.py usuario_tiktok --skip-analysis
```

### Navegador y sesión de TikTok

Por defecto, primero se intenta obtener la página directamente y se recurre a Playwright
si TikTok bloquea esa solicitud. Para usar Playwright desde el inicio:

```bash
python main.py usuario_tiktok --use-playwright
```

Si el perfil llega sin videos o aparece un desafío de TikTok, abre Chrome para ver la
página y resolverlo manualmente:

```bash
python main.py usuario_tiktok --use-playwright --headful --wait-seconds 30
```

En modo visible se utiliza Google Chrome y el perfil de navegador persistente
`output/chrome-profile`. Completa el desafío en esa ventana y deja que termine la
ejecución. No abras a la vez otra instancia de Chrome que esté usando ese mismo perfil.

Para guardar y reutilizar el estado de sesión/cookies explícitamente:

```bash
python main.py usuario_tiktok --use-playwright --headful --wait-seconds 30 --session-state output/tiktok_session_state.json
```

En ejecuciones posteriores, se puede intentar reutilizar ese estado en modo headless:

```bash
python main.py usuario_tiktok --use-playwright --session-state output/tiktok_session_state.json
```

La sesión es local. No compartas los archivos de cookies o el perfil del navegador.
No uses estas opciones para acceder a contenido privado ni para eludir controles de
autenticación.

## Opciones de línea de comandos

| Opción | Predeterminado | Descripción |
|---|---|---|
| `username` | — | Usuario de TikTok o URL del perfil. |
| `-o`, `--output` | `output/profile.json` | Ruta del JSON del perfil y videos. |
| `--use-playwright` | Desactivado | Abre el perfil con Playwright desde el inicio. |
| `--headful` | Desactivado | Usa Chrome visible para resolver manualmente un desafío o login. |
| `--wait-seconds` | `10` | Espera tras cargar el perfil cuando se usa Playwright. |
| `--session-state` | `output/tiktok_session_state.json` | Ruta para leer y guardar estado de sesión. |
| `--chrome-profile` | `output/chrome-profile` | Directorio del perfil persistente de Chrome. |
| `--analysis-output` | `output/analysis` | Directorio para resultados, gráficos e informe. |
| `--skip-analysis` | Desactivado | Guarda el perfil sin calcular el análisis. |

## Archivos generados

Al ejecutar el análisis se crean:

| Archivo | Contenido |
|---|---|
| `output/profile.json` | Datos del perfil, publicaciones, comentarios disponibles y estado de extracción. |
| `output/analysis/analysis.json` | Metodología, porcentajes Big Five, temas, métricas y estado de extracción. |
| `output/analysis/results.csv` | Resumen tabular de rasgos, temas, tipos de contenido y reacciones. |
| `output/analysis/report.html` | Informe navegable con datos, tablas, señales Big Five y mapa temático. |
| `output/analysis/big_five_radar.png` | Gráfico de las señales relativas Big Five. |
| `output/analysis/interest_map.png` | Porcentaje y cantidad de videos asignados a cada tema. |
| `output/analysis/topic_distribution.png` | Cantidad de videos por agrupación temática. |
| `output/analysis/content_types.png` | Tipos de contenido detectados. |
| `output/analysis/reactions.png` | Métricas de reacciones disponibles. |

Los resultados y gráficos reflejan únicamente los videos que se hayan extraído. Revisa
el estado de extracción antes de interpretar el informe.

## Limitaciones y uso responsable

- TikTok puede cambiar su página, sus datos o sus medidas antibots (WAF); la extracción
  puede fallar o ser parcial. En algunos casos hace falta completar un desafío en Chrome.
- Solo se analizan las publicaciones visibles que se hayan podido cargar. No se garantiza
  la descarga de todos los videos del perfil.
- La información de comentarios se limita a lo que TikTok expone públicamente en la
  página recibida; no se consultan endpoints privados.
- La agrupación temática depende del texto disponible y del modelo de embeddings. Las
  etiquetas automáticas pueden ser incompletas o poco representativas, especialmente si
  hay pocos videos.
- Big Five y porcentajes temáticos son indicadores exploratorios del contenido, no
  evaluaciones psicológicas ni mediciones de conocimientos.
- Respeta los términos de servicio de TikTok, la legislación aplicable y la privacidad
  de las personas cuyos datos públicos proceses.
