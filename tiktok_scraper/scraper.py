from __future__ import annotations

import json
import importlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

PROFILE_DATA_RE = [
    re.compile(r'<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__" type="application/json">(.*?)</script>', re.DOTALL),
    re.compile(r'<script id="SIGI_STATE" type="application/json">(.*?)</script>', re.DOTALL),
]


@dataclass
class TikTokProfileScraper:
    timeout: int = 20

    def scrape(
        self,
        username: str,
        use_playwright: bool = False,
        headless: bool = True,
        wait_after_load_ms: int = 8000,
        session_state_path: str | None = None,
    ) -> dict[str, Any]:
        normalized = self._normalize_username(username)
        if not normalized:
            raise ValueError("Debes indicar un username de TikTok válido.")

        url = f"https://www.tiktok.com/@{normalized}"
        html = self._download_profile_html(
            url,
            use_playwright=use_playwright,
            headless=headless,
            wait_after_load_ms=wait_after_load_ms,
            session_state_path=session_state_path,
        )
        profile_data = self._extract_profile_json(html)

        return {
            "source": url,
            "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
            "profile": self._extract_profile_info(profile_data, normalized),
            "videos": self._extract_videos(profile_data),
        }

    def _download_profile_html(
        self,
        url: str,
        use_playwright: bool,
        headless: bool,
        wait_after_load_ms: int,
        session_state_path: str | None,
    ) -> str:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/128.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
            "Referer": "https://www.tiktok.com/",
        }
        try:
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            html = response.text
        except requests.RequestException as exc:
            if use_playwright:
                return self._download_profile_html_playwright(
                    url,
                    headless=headless,
                    wait_after_load_ms=wait_after_load_ms,
                    session_state_path=session_state_path,
                )
            raise RuntimeError("No se pudo descargar el perfil de TikTok.") from exc

        if self._looks_like_waf_challenge(html):
            if use_playwright:
                html = self._download_profile_html_playwright(
                    url,
                    headless=headless,
                    wait_after_load_ms=wait_after_load_ms,
                    session_state_path=session_state_path,
                )
                if not self._looks_like_waf_challenge(html):
                    return html
                raise RuntimeError(
                    "TikTok sigue devolviendo WAF incluso con navegador. "
                    "Prueba `--headful` y completa manualmente el desafío/login."
                )
            raise RuntimeError(
                "TikTok devolvió una página anti-bot (WAF), no el perfil. "
                "Ejecuta con `--use-playwright` o intenta más tarde."
            )
        return html

    def _download_profile_html_playwright(
        self,
        url: str,
        headless: bool,
        wait_after_load_ms: int,
        session_state_path: str | None,
    ) -> str:
        try:
            sync_api_module = importlib.import_module("playwright.sync_api")
        except ImportError as exc:
            raise RuntimeError(
                "Playwright no está instalado. Ejecuta `pip install -r requirements.txt` "
                "y luego `python -m playwright install chromium`."
            ) from exc

        playwright_timeout_error = getattr(sync_api_module, "TimeoutError")
        sync_playwright = getattr(sync_api_module, "sync_playwright")

        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=headless)
                context_options: dict[str, Any] = {
                    "user_agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/128.0.0.0 Safari/537.36"
                    ),
                    "locale": "es-ES",
                }

                state_path: Path | None = None
                if session_state_path:
                    state_path = Path(session_state_path)
                    if state_path.exists():
                        context_options["storage_state"] = str(state_path)

                context = browser.new_context(**context_options)
                page = context.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=self.timeout * 1000)
                page.wait_for_timeout(wait_after_load_ms)
                html = page.content()

                if state_path is not None:
                    state_path.parent.mkdir(parents=True, exist_ok=True)
                    context.storage_state(path=str(state_path))

                context.close()
                browser.close()
                return html
        except playwright_timeout_error as exc:
            raise RuntimeError(
                "Timeout usando Playwright. Prueba aumentar espera con `--wait-seconds` o usar `--headful`."
            ) from exc

    def _normalize_username(self, value: str) -> str:
        raw_value = value.strip()
        if not raw_value:
            return ""

        if "tiktok.com" in raw_value:
            parsed = urlparse(raw_value)
            path_parts = [part for part in parsed.path.split("/") if part]
            for index, part in enumerate(path_parts):
                if part.startswith("@"):
                    return part.lstrip("@")
                if part == "@" and index + 1 < len(path_parts):
                    return path_parts[index + 1].lstrip("@")

        return raw_value.lstrip("@")

    def _looks_like_waf_challenge(self, html: str) -> bool:
        waf_markers = (
            "SlardarWAF",
            "_wafchallengeid",
            "waforiginalreid",
            "Please wait...",
        )
        return any(marker in html for marker in waf_markers)

    def _extract_profile_json(self, html: str) -> dict[str, Any]:
        for pattern in PROFILE_DATA_RE:
            match = pattern.search(html)
            if not match:
                continue

            raw_json = match.group(1)
            try:
                data = json.loads(raw_json)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                continue

        raise RuntimeError(
            "No se pudo extraer JSON del perfil. TikTok pudo cambiar su HTML o bloquear la petición."
        )

    def _extract_profile_info(self, data: dict[str, Any], fallback_username: str) -> dict[str, Any]:
        candidate = self._find_first_user_object(data)
        if not candidate:
            return {"username": fallback_username}

        stats = candidate.get("stats", {}) if isinstance(candidate.get("stats"), dict) else {}

        return {
            "id": candidate.get("id"),
            "username": candidate.get("uniqueId") or fallback_username,
            "nickname": candidate.get("nickname"),
            "bio": candidate.get("signature"),
            "verified": candidate.get("verified"),
            "following": stats.get("followingCount"),
            "followers": stats.get("followerCount"),
            "likes": stats.get("heartCount"),
            "videos_count": stats.get("videoCount"),
        }

    def _extract_videos(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        videos: list[dict[str, Any]] = []

        item_modules = self._find_key_recursive(data, "ItemModule")
        if isinstance(item_modules, dict):
            for item_id, video in item_modules.items():
                if not isinstance(video, dict):
                    continue
                stats = video.get("stats") if isinstance(video.get("stats"), dict) else {}
                videos.append(
                    {
                        "id": item_id,
                        "description": video.get("desc"),
                        "created_at_unix": video.get("createTime"),
                        "views": stats.get("playCount"),
                        "likes": stats.get("diggCount"),
                        "comments": stats.get("commentCount"),
                        "shares": stats.get("shareCount"),
                    }
                )

        videos.sort(key=lambda row: row.get("created_at_unix") or 0, reverse=True)
        return videos

    def _find_first_user_object(self, node: Any) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if "uniqueId" in node and "nickname" in node:
                return node
            for value in node.values():
                found = self._find_first_user_object(value)
                if found:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = self._find_first_user_object(item)
                if found:
                    return found
        return None

    def _find_key_recursive(self, node: Any, key: str) -> Any:
        if isinstance(node, dict):
            if key in node:
                return node[key]
            for value in node.values():
                found = self._find_key_recursive(value, key)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = self._find_key_recursive(item, key)
                if found is not None:
                    return found
        return None


def scrape_profile(
    username: str,
    use_playwright: bool = False,
    headless: bool = True,
    wait_after_load_ms: int = 8000,
    session_state_path: str | None = None,
) -> dict[str, Any]:
    return TikTokProfileScraper().scrape(
        username,
        use_playwright=use_playwright,
        headless=headless,
        wait_after_load_ms=wait_after_load_ms,
        session_state_path=session_state_path,
    )
