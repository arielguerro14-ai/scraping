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
    re.compile(
        r'<script\b[^>]*\bid=["\']__UNIVERSAL_DATA_FOR_REHYDRATION__["\'][^>]*>(.*?)</script>',
        re.DOTALL | re.IGNORECASE,
    ),
    re.compile(
        r'<script\b[^>]*\bid=["\']SIGI_STATE["\'][^>]*>(.*?)</script>',
        re.DOTALL | re.IGNORECASE,
    ),
    re.compile(
        r'<script\b[^>]*type=["\']application/json["\'][^>]*>(.*?)</script>',
        re.DOTALL | re.IGNORECASE,
    ),
    re.compile(
        r'<script\b[^>]*id=["\']SCRAPER_DOM_VIDEOS["\'][^>]*>(.*?)</script>',
        re.DOTALL | re.IGNORECASE,
    ),
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
        chrome_profile_path: str | None = None,
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
            chrome_profile_path=chrome_profile_path,
        )
        profile_data = self._extract_profile_json(html)
        profile = self._extract_profile_info(profile_data, normalized)
        videos = self._extract_videos(profile_data)
        comments = self._extract_comments(profile_data)
        comments.extend(
            comment
            for video in videos
            for comment in video.get("comments_data", [])
            if comment not in comments
        )

        return {
            "source": url,
            "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
            "profile": profile,
            "videos": videos,
            "comments": comments,
            "extraction": {
                "status": (
                    "complete"
                    if not profile.get("videos_count") or len(videos) >= profile["videos_count"]
                    else "partial"
                ),
                "expected_videos": profile.get("videos_count"),
                "videos_extracted": len(videos),
                "message": (
                    (
                        f"TikTok entregó {len(videos)} de {profile['videos_count']} publicaciones; "
                        "el análisis usa las publicaciones visibles cargadas."
                    )
                    if profile.get("videos_count") and len(videos) < profile["videos_count"]
                    else None
                ),
            },
        }

    def _download_profile_html(
        self,
        url: str,
        use_playwright: bool,
        headless: bool,
        wait_after_load_ms: int,
        session_state_path: str | None,
        chrome_profile_path: str | None,
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
            try:
                return self._download_profile_html_playwright(
                    url,
                    headless=headless,
                    wait_after_load_ms=wait_after_load_ms,
                    session_state_path=session_state_path,
                    chrome_profile_path=chrome_profile_path,
                )
            except RuntimeError:
                if use_playwright:
                    raise
            raise RuntimeError("No se pudo descargar el perfil de TikTok.") from exc

        if self._looks_like_waf_challenge(html):
            try:
                html = self._download_profile_html_playwright(
                    url,
                    headless=headless,
                    wait_after_load_ms=wait_after_load_ms,
                    session_state_path=session_state_path,
                    chrome_profile_path=chrome_profile_path,
                )
            except RuntimeError:
                if use_playwright:
                    raise
                raise RuntimeError(
                    "TikTok devolvió WAF y no se pudo iniciar Playwright. "
                    "Instala Playwright y Chromium con `pip install -r requirements.txt` "
                    "y `python -m playwright install chromium`."
                )
            if not self._looks_like_waf_challenge(html):
                return html
            raise RuntimeError(
                "TikTok sigue devolviendo WAF incluso con navegador. "
                "Ejecuta con `--headful` para completar manualmente el desafío/login "
                "y conserva la sesión con `--session-state`."
            )
        return html

    def _download_profile_html_playwright(
        self,
        url: str,
        headless: bool,
        wait_after_load_ms: int,
        session_state_path: str | None,
        chrome_profile_path: str | None,
    ) -> str:
        try:
            sync_api_module = importlib.import_module("playwright.sync_api")
        except ImportError as exc:
            raise RuntimeError(
                "Playwright no está instalado. Ejecuta `pip install -r requirements.txt` "
                "y luego `python -m playwright install chromium`."
            ) from exc

        playwright_timeout_error = getattr(sync_api_module, "TimeoutError")
        playwright_error = getattr(sync_api_module, "Error")
        sync_playwright = getattr(sync_api_module, "sync_playwright")

        try:
            with sync_playwright() as playwright:
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

                if not headless:
                    profile_path = Path(chrome_profile_path or "output/chrome-profile")
                    profile_path.mkdir(parents=True, exist_ok=True)
                    context_options.pop("storage_state", None)
                    context = playwright.chromium.launch_persistent_context(
                        user_data_dir=str(profile_path),
                        channel="chrome",
                        headless=False,
                        **context_options,
                    )
                    browser = None
                else:
                    browser = playwright.chromium.launch(headless=True)
                    context = browser.new_context(**context_options)
                page = context.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=self.timeout * 1000)
                page.wait_for_timeout(wait_after_load_ms)
                html = page.content()
                if not headless and self._looks_like_waf_challenge(html):
                    print(
                        "TikTok mostró un desafío WAF. Complétalo en Chromium; "
                        "la aplicación esperará hasta 90 segundos."
                    )
                    for _ in range(90):
                        page.wait_for_timeout(1000)
                        html = page.content()
                        if not self._looks_like_waf_challenge(html):
                            break
                if not self._looks_like_waf_challenge(html):
                    for _ in range(4):
                        page.mouse.wheel(0, 1800)
                        page.wait_for_timeout(1200)
                    html = page.content()
                    dom_videos = page.locator('a[href*="/video/"]').evaluate_all(
                        """elements => elements.map(element => ({
                            href: element.href,
                            text: element.innerText || '',
                            label: element.getAttribute('aria-label') || '',
                            title: element.getAttribute('title') || ''
                        }))"""
                    )
                    if dom_videos:
                        dom_videos = self._enrich_video_pages(page, dom_videos)
                        html += (
                            '<script id="SCRAPER_DOM_VIDEOS" type="application/json">'
                            + json.dumps(dom_videos, ensure_ascii=False)
                            + "</script>"
                        )

                if state_path is not None:
                    state_path.parent.mkdir(parents=True, exist_ok=True)
                    context.storage_state(path=str(state_path))

                context.close()
                if browser is not None:
                    browser.close()
                return html
        except playwright_timeout_error as exc:
            raise RuntimeError(
                "Timeout usando Playwright. Prueba aumentar espera con `--wait-seconds` o usar `--headful`."
            ) from exc
        except playwright_error as exc:
            raise RuntimeError(
                "No se pudo iniciar Google Chrome con Playwright. "
                "Cierra otras instancias que usen el perfil del proyecto y verifica "
                "que Chrome esté instalado. Detalle: "
                f"{exc}"
            ) from exc

    def _enrich_video_pages(
        self, page: Any, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in candidates[:100]:
            href = str(candidate.get("href") or "")
            if not href or href in seen:
                continue
            seen.add(href)
            record = dict(candidate)
            try:
                page.goto(href, wait_until="domcontentloaded", timeout=self.timeout * 1000)
                page.wait_for_timeout(900)
                record.update(
                    page.evaluate(
                        """() => {
                            const meta = name => document.querySelector(
                              `meta[name="${name}"], meta[property="${name}"]`
                            )?.content || "";
                            const text = document.body?.innerText || "";
                            const description =
                              document.querySelector('[data-e2e="browse-video-desc"]')?.innerText ||
                              document.querySelector('[data-e2e="video-desc"]')?.innerText ||
                              meta("description") ||
                              meta("og:description") ||
                              "";
                            const comments = Array.from(document.querySelectorAll(
                              '[data-e2e*="comment"], [data-e2e*="Comment"]'
                            )).map(node => node.innerText.trim()).filter(Boolean).slice(0, 50);
                            return {
                              description: description,
                              title: document.title || "",
                              author: meta("author"),
                              cover: meta("og:image"),
                              page_text: text.slice(0, 12000),
                              embedded_data: Array.from(document.scripts)
                                .map(script => script.textContent || "")
                                .filter(value => value.includes('"desc"') || value.includes('"playCount"'))
                                .slice(0, 10),
                              comments: comments
                            };
                        }"""
                    )
                )
            except (
                getattr(importlib.import_module("playwright.sync_api"), "TimeoutError"),
                getattr(importlib.import_module("playwright.sync_api"), "Error"),
            ) as exc:
                record["detail_error"] = str(exc)
            enriched.append(record)
        return enriched

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
        if self._looks_like_waf_challenge(html):
            raise RuntimeError(
                "TikTok mantiene activo el WAF. Completa el desafío en Chromium "
                "y vuelve a ejecutar con `--headful`."
            )
        for pattern in PROFILE_DATA_RE:
            match = pattern.search(html)
            if not match:
                continue

            raw_json = match.group(1)
            try:
                data = json.loads(raw_json)
                if isinstance(data, dict):
                    dom_match = re.search(
                        r'<script\b[^>]*id=["\']SCRAPER_DOM_VIDEOS["\'][^>]*>(.*?)</script>',
                        html,
                        re.DOTALL | re.IGNORECASE,
                    )
                    if dom_match:
                        try:
                            dom_videos = json.loads(dom_match.group(1))
                        except json.JSONDecodeError:
                            dom_videos = []
                        if isinstance(dom_videos, list):
                            data["SCRAPER_DOM_VIDEOS"] = dom_videos
                    return data
            except json.JSONDecodeError:
                continue

        raise RuntimeError(
            "La página cargó, pero TikTok no incluyó datos JSON del perfil. "
            "Comprueba que el desafío WAF haya terminado, espera a que cargue el perfil "
            "y vuelve a ejecutar con `--headful --wait-seconds 60`."
        )

    def _extract_profile_info(self, data: dict[str, Any], fallback_username: str) -> dict[str, Any]:
        candidate = self._find_first_user_object(data) or {}
        user_info = self._find_key_recursive(data, "userInfo")
        if isinstance(user_info, dict):
            user = user_info.get("user")
            if isinstance(user, dict):
                candidate = {**candidate, **user}
            info_stats = user_info.get("stats")
            if isinstance(info_stats, dict):
                candidate = {**candidate, "stats": info_stats}
        if not candidate:
            return {"username": fallback_username}

        stats = candidate.get("stats", {}) if isinstance(candidate.get("stats"), dict) else {}
        bio_link = candidate.get("bioLink")
        if isinstance(bio_link, dict):
            bio_link = bio_link.get("link")

        return {
            "id": candidate.get("id"),
            "username": candidate.get("uniqueId") or fallback_username,
            "nickname": candidate.get("nickname"),
            "bio": candidate.get("signature"),
            "avatar": candidate.get("avatarLarger") or candidate.get("avatarMedium"),
            "bio_link": bio_link,
            "region": candidate.get("region"),
            "language": candidate.get("language"),
            "verified": candidate.get("verified"),
            "following": stats.get("followingCount"),
            "followers": stats.get("followerCount"),
            "likes": stats.get("heartCount") or stats.get("heart"),
            "videos_count": stats.get("videoCount"),
        }

    def _extract_videos(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        videos: list[dict[str, Any]] = []

        item_modules = self._find_key_recursive(data, "ItemModule")
        if isinstance(item_modules, dict):
            for item_id, video in item_modules.items():
                if isinstance(video, dict):
                    videos.append(self._normalize_video(video, str(item_id)))

        for video in self._find_video_objects(data):
            video_id = str(video.get("id") or video.get("video", {}).get("id") or "")
            if video_id and not any(item["id"] == video_id for item in videos):
                videos.append(self._normalize_video(video, video_id))

        dom_videos = data.get("SCRAPER_DOM_VIDEOS")
        if isinstance(dom_videos, list):
            for item in dom_videos:
                if not isinstance(item, dict):
                    continue
                href = str(item.get("href") or "")
                match = re.search(r"/video/(\d+)", href)
                if not match:
                    continue
                video_id = match.group(1)
                if not any(video["id"] == video_id for video in videos):
                    card = self._parse_dom_video(item)
                    videos.append(
                        self._normalize_video(
                            {
                                "id": video_id,
                                "description": card["description"],
                                "page_text": card.get("page_text"),
                                "stats": {"playCount": card["views"]},
                                                "comments": card.get("comments"),
                                                "author": {"uniqueId": card.get("author")},
                                                "music": {"title": card.get("music")},
                                                "video": {"cover": card.get("cover")},
                            },
                            video_id,
                        )
                    )

        videos.sort(key=lambda row: row.get("created_at_unix") or 0, reverse=True)
        return videos

    def _parse_dom_video(self, item: dict[str, Any]) -> dict[str, Any]:
        raw_text = " ".join(
            str(item.get(key) or "") for key in ("text", "label", "title")
        ).strip()
        views_match = re.search(
            r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*([KMB])?(?:\s*(?:vistas?|views?))?",
            raw_text,
            re.IGNORECASE,
        )
        views = 0
        if views_match:
            number = float(views_match.group(1).replace(",", "."))
            multiplier = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}.get(
                (views_match.group(2) or "").lower(),
                1,
            )
            views = int(number * multiplier)
        description = str(item.get("description") or "")
        if not description:
            description = self._description_from_embedded_data(item.get("embedded_data"))
        if not description:
            description = raw_text
        description = re.sub(r"\b(?:Anclado|Pinned)\b", "", description, flags=re.IGNORECASE)
        if views_match:
            description = description.replace(views_match.group(0), "")
        description = " ".join(description.split()).strip()
        return {
            "description": description or None,
            "page_text": item.get("page_text"),
            "embedded_data": item.get("embedded_data"),
            "views": views,
            "comments": item.get("comments") or [],
            "author": item.get("author"),
            "music": item.get("title") or item.get("music"),
            "cover": item.get("cover"),
        }

    def _normalize_video(self, video: dict[str, Any], fallback_id: str) -> dict[str, Any]:
        stats = video.get("stats") if isinstance(video.get("stats"), dict) else {}
        video_info = video.get("video") if isinstance(video.get("video"), dict) else {}
        music = video.get("music") if isinstance(video.get("music"), dict) else {}
        author = video.get("author") if isinstance(video.get("author"), dict) else {}
        description = (
            video.get("desc")
            or video.get("description")
            or self._description_from_embedded_data(video.get("embedded_data"))
        )
        hashtags = [
            item.get("hashtagName") or item.get("name")
            for item in video.get("textExtra", [])
            if isinstance(item, dict) and (item.get("hashtagName") or item.get("name"))
        ]
        comments_data = video.get("comments") if isinstance(video.get("comments"), list) else []
        normalized_comments = [
            {"id": f"dom-{index}", "text": text, "likes": 0}
            for index, text in enumerate(comments_data)
            if isinstance(text, str) and text.strip()
        ]
        return {
            "id": str(video.get("id") or fallback_id),
            "description": description,
            "page_text": video.get("page_text"),
            "embedded_data": video.get("embedded_data"),
            "created_at_unix": video.get("createTime") or video.get("createdAt"),
            "views": stats.get("playCount") or stats.get("views") or self._embedded_number(video, "playCount"),
            "likes": stats.get("diggCount") or stats.get("likeCount") or self._embedded_number(video, "diggCount"),
            "comments": stats.get("commentCount") or self._embedded_number(video, "commentCount"),
            "comments_count": stats.get("commentCount") or self._embedded_number(video, "commentCount"),
            "shares": stats.get("shareCount") or self._embedded_number(video, "shareCount"),
            "saves": stats.get("collectCount") or stats.get("收藏数") or self._embedded_number(video, "collectCount"),
            "duration_seconds": video_info.get("duration") or video.get("duration"),
            "cover": video_info.get("cover") or video.get("cover"),
            "music_title": music.get("title"),
            "music_author": music.get("authorName"),
            "author_username": author.get("uniqueId"),
            "hashtags": hashtags,
            "content_type": self._content_type(video, description, hashtags),
            "comments_data": self._extract_comments(video) or normalized_comments,
        }

    def _embedded_number(self, video: dict[str, Any], key: str) -> int | None:
        scripts = video.get("embedded_data")
        if not isinstance(scripts, list):
            return None
        pattern = re.compile(rf'"{re.escape(key)}"\s*:\s*(\d+)')
        for script in scripts:
            if not isinstance(script, str):
                continue
            match = pattern.search(script)
            if match:
                return int(match.group(1))
        return None

    def _description_from_embedded_data(self, scripts: Any) -> str | None:
        if not isinstance(scripts, list):
            return None
        for script in scripts:
            if not isinstance(script, str):
                continue
            match = re.search(r'"desc"\s*:\s*"((?:\\.|[^"\\])*)"', script)
            if not match:
                continue
            try:
                value = json.loads(f'"{match.group(1)}"')
            except json.JSONDecodeError:
                value = match.group(1)
            value = " ".join(str(value).split()).strip()
            if value:
                return value
        return None

    def _content_type(
        self, video: dict[str, Any], description: str | None, hashtags: list[str]
    ) -> str:
        text = " ".join(
            [str(description or ""), " ".join(str(tag) for tag in hashtags)]
        ).lower()
        if video.get("duetInfo") or "dueto" in text or "duet" in text:
            return "dueto"
        if video.get("stitchInfo") or "stitch" in text:
            return "stitch"
        if any(word in text for word in ("tutorial", "aprende", "tips", "cómo", "como")):
            return "educativo"
        if any(word in text for word in ("receta", "cocina", "comida", "food")):
            return "gastronomía"
        if any(word in text for word in ("música", "musica", "canción", "cover", "lyrics")):
            return "música"
        if any(word in text for word in ("viaje", "viajes", "turismo", "travel")):
            return "viajes"
        if any(word in text for word in ("humor", "comedia", "risa", "meme")):
            return "humor"
        return "general"

    def _extract_comments(self, data: Any) -> list[dict[str, Any]]:
        comments: list[dict[str, Any]] = []
        seen: set[str] = set()

        def visit(node: Any) -> None:
            if isinstance(node, dict):
                text = node.get("text") or node.get("commentText")
                user = node.get("user") if isinstance(node.get("user"), dict) else {}
                is_comment = any(
                    key in node
                    for key in ("cid", "commentText", "digg_count", "diggCount", "create_time")
                )
                if text and isinstance(text, str) and is_comment:
                    comment_id = str(node.get("cid") or node.get("id") or f"{user.get('unique_id', '')}:{text}")
                    if comment_id not in seen:
                        seen.add(comment_id)
                        comments.append(
                            {
                                "id": comment_id,
                                "text": text,
                                "username": user.get("unique_id") or user.get("uniqueId"),
                                "nickname": user.get("nickname"),
                                "likes": node.get("digg_count") or node.get("diggCount") or 0,
                                "created_at_unix": node.get("create_time") or node.get("createTime"),
                            }
                        )
                for value in node.values():
                    visit(value)
            elif isinstance(node, list):
                for item in node:
                    visit(item)

        visit(data)
        return comments

    def _find_video_objects(self, node: Any) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        if isinstance(node, dict):
            if (
                ("desc" in node or "description" in node)
                and ("createTime" in node or "createdAt" in node)
                and ("stats" in node or "video" in node)
            ):
                found.append(node)
            for value in node.values():
                found.extend(self._find_video_objects(value))
        elif isinstance(node, list):
            for item in node:
                found.extend(self._find_video_objects(item))
        return found

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
    chrome_profile_path: str | None = None,
) -> dict[str, Any]:
    return TikTokProfileScraper().scrape(
        username,
        use_playwright=use_playwright,
        headless=headless,
        wait_after_load_ms=wait_after_load_ms,
        session_state_path=session_state_path,
        chrome_profile_path=chrome_profile_path,
    )
