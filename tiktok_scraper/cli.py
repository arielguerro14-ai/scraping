from __future__ import annotations

import argparse
import json
from pathlib import Path

from .scraper import scrape_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scraping básico de perfil público de TikTok",
    )
    parser.add_argument("username", help="Usuario de TikTok (con o sin @) o URL del perfil")
    parser.add_argument(
        "-o",
        "--output",
        default="output/profile.json",
        help="Ruta del archivo JSON de salida (por defecto: output/profile.json)",
    )
    parser.add_argument(
        "--use-playwright",
        action="store_true",
        help="Usa navegador con Playwright (útil si TikTok bloquea por WAF)",
    )
    parser.add_argument(
        "--headful",
        action="store_true",
        help="Abre navegador visible (no headless) para resolver captcha/login manual",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=10,
        help="Segundos de espera tras cargar la página en Playwright (por defecto: 10)",
    )
    parser.add_argument(
        "--session-state",
        default="output/tiktok_session_state.json",
        help="Ruta para cargar/guardar cookies y sesión de Playwright",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data = scrape_profile(
        args.username,
        use_playwright=args.use_playwright,
        headless=not args.headful,
        wait_after_load_ms=max(args.wait_seconds, 1) * 1000,
        session_state_path=args.session_state if args.use_playwright else None,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Perfil scrapeado correctamente: {output_path}")
    print(f"Videos encontrados: {len(data.get('videos', []))}")


if __name__ == "__main__":
    main()
