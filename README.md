# TikTok Profile Scraper (público)

Proyecto simple en Python para extraer información **pública** de una cuenta de TikTok y guardarla en JSON.

## Qué obtiene

- Datos del perfil: username, nickname, bio, verificación y métricas generales.
- Lista de videos visibles: descripción, fecha (unix), vistas, likes, comentarios y compartidos.

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

Si TikTok bloquea por WAF, usa Playwright:

```bash
python main.py usuario_tiktok --use-playwright
```

Para abrir navegador visible y resolver desafío/login manual:

```bash
python main.py usuario_tiktok --use-playwright --headful --wait-seconds 20
```

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
- TikTok puede cambiar su HTML o aplicar bloqueos (WAF); si eso pasa, la extracción puede fallar aunque el código esté correcto.
- Respeta siempre términos de servicio y legislación aplicable al uso de datos.
