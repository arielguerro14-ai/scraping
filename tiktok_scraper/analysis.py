from __future__ import annotations

import csv
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

TRAIT_ANCHORS = {
    "apertura": (
        "El contenido expresa curiosidad, creatividad, interés por el arte, ideas nuevas y experiencias diferentes.",
        "El contenido prefiere lo familiar, concreto y rutinario, sin explorar ideas nuevas.",
    ),
    "responsabilidad": (
        "El contenido refleja planificación, disciplina, organización, constancia y cumplimiento de metas.",
        "El contenido refleja improvisación, desorganización y poca preocupación por los planes o las obligaciones.",
    ),
    "extraversión": (
        "El contenido muestra energía social, comunicación, interacción con otras personas y entusiasmo.",
        "El contenido muestra reserva, preferencia por la soledad y poca búsqueda de interacción social.",
    ),
    "amabilidad": (
        "El contenido expresa empatía, cooperación, confianza, consideración y preocupación por otras personas.",
        "El contenido expresa confrontación, desconfianza, competitividad y poca consideración por otras personas.",
    ),
    "neuroticismo": (
        "El contenido expresa preocupación, tensión, inseguridad o emociones negativas intensas.",
        "El contenido expresa calma, seguridad emocional y estabilidad ante situaciones difíciles.",
    ),
}
POSITIVE_WORDS = ("amor", "gracias", "excelente", "genial", "me encanta", "bonito", "feliz")
NEGATIVE_WORDS = ("odio", "malo", "triste", "aburrido", "horrible", "decepción", "decepcion")
ENTITY_PATTERN = re.compile(r"(?<!\w)(?:#[\wáéíóúñ]+|@[A-Za-z0-9_.]+)|\b[A-ZÁÉÍÓÚÑ][\wáéíóúñ]{2,}(?:\s+[A-ZÁÉÍÓÚÑ][\wáéíóúñ]{2,})*")
WORD_PATTERN = re.compile(r"[a-záéíóúñü]{3,}", re.IGNORECASE)
DOM_VIEWS_PATTERN = re.compile(
    r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*([KMB])?(?:\s*(?:vistas?|views?))?",
    re.IGNORECASE,
)


def _valid_comments(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        comment
        for comment in data.get("comments", [])
        if isinstance(comment, dict)
        and (
            comment.get("id", "").startswith("c")
            or comment.get("id", "").startswith("dom-")
            or comment.get("likes", 0)
            or comment.get("created_at_unix")
        )
        and not str(comment.get("text") or "").startswith(
            ("pc_web_explorePage_", "Web_explorePage_")
        )
    ]


def _enrich_dom_videos(data: dict[str, Any]) -> list[dict[str, Any]]:
    videos = []
    for original in data.get("videos", []):
        video = dict(original)
        description = str(video.get("description") or "")
        match = DOM_VIEWS_PATTERN.search(description)
        if match and not video.get("views"):
            number = float(match.group(1).replace(",", "."))
            multiplier = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}.get(
                (match.group(2) or "").lower(),
                1,
            )
            video["views"] = int(number * multiplier)
            description = DOM_VIEWS_PATTERN.sub("", description, count=1)
            description = re.sub(r"\b(?:Anclado|Pinned)\b", "", description, flags=re.IGNORECASE)
            video["description"] = " ".join(description.split()).strip() or None
        videos.append(video)
    return videos


def _texts(data: dict[str, Any]) -> list[str]:
    profile = data.get("profile") or {}
    texts = [str(profile.get("bio") or "")]
    texts.extend(_video_texts(data))
    texts.extend(str(comment.get("text") or "") for comment in _valid_comments(data))
    return [text.strip() for text in texts if text.strip()]


def _video_texts(data: dict[str, Any]) -> list[str]:
    texts = []
    for video in data.get("videos", []):
        music_title = video.get("music_title")
        if music_title == "TikTok - Make Your Day":
            music_title = None
        text = " ".join(
            str(value)
            for value in (
                video.get("description"),
                music_title,
                *(video.get("hashtags") or []),
            )
            if value
        ).strip()
        if text:
            texts.append(text)
    return texts


def _content_insights(data: dict[str, Any]) -> dict[str, Any]:
    videos = _enrich_dom_videos(data)
    comments = _valid_comments(data)
    type_counts = Counter(video.get("content_type") for video in videos if video.get("description"))
    profile_bio = str((data.get("profile") or {}).get("bio") or "").lower()
    profile_interests = [
        label
        for label, keywords in (
            ("música y letras", ("música", "musica", "lyrics", "canción")),
            ("nostalgia", ("nostalgia", "recuerdo", "recuerdos")),
            ("colaboraciones", ("colab", "colaboración", "colaboracion")),
        )
        if any(keyword in profile_bio for keyword in keywords)
    ]
    hashtag_counts = Counter(
        hashtag.lower()
        for video in videos
        for hashtag in video.get("hashtags") or []
        if hashtag
    )
    reaction_totals = {
        "views": sum(video.get("views") or 0 for video in videos),
        "likes": sum(video.get("likes") or 0 for video in videos),
        "comments": sum(
            video.get("comments_count")
            if isinstance(video.get("comments_count"), (int, float))
            else video.get("comments") if isinstance(video.get("comments"), (int, float)) else 0
            for video in videos
        ) or None,
        "shares": sum(video.get("shares") or 0 for video in videos) or None,
        "saves": sum(video.get("saves") or 0 for video in videos) or None,
        "comment_likes": sum(comment.get("likes") or 0 for comment in comments) or None,
    }
    comment_texts = [str(comment.get("text") or "").lower() for comment in comments]
    sentiment = {
        "positive": sum(any(word in text for word in POSITIVE_WORDS) for text in comment_texts),
        "negative": sum(any(word in text for word in NEGATIVE_WORDS) for text in comment_texts),
        "neutral": 0,
    }
    sentiment["neutral"] = len(comments) - sentiment["positive"] - sentiment["negative"]
    commenters = Counter(
        comment.get("username") or comment.get("nickname") or "anónimo"
        for comment in comments
    )
    return {
        "content_types": [
            {"type": content_type or "sin clasificar", "videos": count}
            for content_type, count in type_counts.most_common()
        ] or [{"type": "sin descripción disponible", "videos": len(videos)}],
        "profile_interests": profile_interests,
        "top_hashtags": [
            {"hashtag": hashtag, "mentions": count}
            for hashtag, count in hashtag_counts.most_common(15)
        ],
        "reactions": reaction_totals,
        "comment_sentiment": sentiment,
        "top_commenters": [
            {"user": user, "comments": count}
            for user, count in commenters.most_common(10)
        ],
        "comments_sample": comments[:30],
    }


def _entities(texts: list[str]) -> list[dict[str, Any]]:
    counts = Counter(entity.lower() for text in texts for entity in ENTITY_PATTERN.findall(text))
    return [{"entity": entity, "mentions": count} for entity, count in counts.most_common()]


def _big_five(
    texts: list[str], embeddings: np.ndarray, anchor_embeddings: np.ndarray
) -> list[dict[str, Any]]:
    token_count = sum(len(WORD_PATTERN.findall(text)) for text in texts)
    source_count = len(texts)
    results = []
    raw_scores = []
    for index, trait in enumerate(TRAIT_ANCHORS):
        positive_similarity = float(np.mean(embeddings @ anchor_embeddings[index * 2]))
        negative_similarity = float(np.mean(embeddings @ anchor_embeddings[index * 2 + 1]))
        confidence = min(1.0, token_count / 250) * min(1.0, source_count / 50)
        denominator = abs(positive_similarity) + abs(negative_similarity)
        raw_score = (
            50 + 50 * (positive_similarity - negative_similarity) / denominator
            if denominator
            else 50
        )
        raw_scores.append(raw_score)
        sufficient_evidence = confidence >= 0.5 and token_count >= 20
        results.append(
            {
                "trait": trait,
                "score": None,
                "positive_similarity": round(positive_similarity, 3),
                "negative_similarity": round(negative_similarity, 3),
                "evidence_words": token_count,
                "evidence_documents": source_count,
                "confidence": round(confidence, 2),
                "status": "suficiente" if sufficient_evidence else "insuficiente",
            }
        )
    if results and all(item["status"] == "suficiente" for item in results):
        total_score = sum(raw_scores)
        exact_tenths = [
            score / total_score * 1000 if total_score else 1000 / len(results)
            for score in raw_scores
        ]
        tenths = [int(value) for value in exact_tenths]
        remainder = 1000 - sum(tenths)
        remainder_order = sorted(
            range(len(results)),
            key=lambda index: exact_tenths[index] - tenths[index],
            reverse=True,
        )
        for index in remainder_order[:remainder]:
            tenths[index] += 1
        for item, value in zip(results, tenths):
            item["score"] = value / 10
    return results


def _embeddings(texts: list[str]) -> tuple[np.ndarray, str]:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return np.asarray(model.encode(texts, normalize_embeddings=True)), "sentence-transformers"
    except ImportError:
        pass

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=256)
    try:
        matrix = vectorizer.fit_transform(texts).toarray()
    except ValueError:
        matrix = np.zeros((len(texts), 1), dtype=float)
    return normalize(matrix), "tfidf-fallback"


def _topics(texts: list[str], embeddings: np.ndarray) -> list[dict[str, Any]]:
    if len(texts) < 2:
        return [
            {
                "cluster": 0,
                "label": "contenido insuficiente",
                "documents": len(texts),
                "percentage": 100.0 if texts else 0.0,
            }
        ]
    distinct_count = len(np.unique(embeddings, axis=0))
    cluster_count = min(5, max(2, int(np.sqrt(len(texts))), distinct_count))
    cluster_count = min(cluster_count, distinct_count)
    if cluster_count < 2:
        return [
            {
                "cluster": 0,
                "label": "contenido repetido",
                "documents": len(texts),
                "percentage": 100.0,
            }
        ]
    labels = KMeans(n_clusters=cluster_count, random_state=42, n_init=10).fit_predict(embeddings)
    vectorizer = TfidfVectorizer(stop_words=None, max_features=256)
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        topics = [
            {
                "cluster": cluster,
                "label": f"tema {cluster + 1}",
                "documents": int(np.sum(labels == cluster)),
            }
            for cluster in range(cluster_count)
        ]
        return _add_topic_percentages(topics, len(texts))
    terms = np.asarray(vectorizer.get_feature_names_out())
    topics = []
    for cluster in range(cluster_count):
        indexes = np.flatnonzero(labels == cluster)
        weights = matrix[indexes].mean(axis=0).A1
        top_terms = terms[np.argsort(weights)[-5:][::-1]].tolist()
        topics.append(
            {
                "cluster": int(cluster),
                "label": ", ".join(top_terms) or f"tema {cluster + 1}",
                "documents": int(len(indexes)),
            }
        )
    return _add_topic_percentages(topics, len(texts))


def _add_topic_percentages(
    topics: list[dict[str, Any]], document_count: int
) -> list[dict[str, Any]]:
    if not topics or not document_count:
        return topics
    exact_tenths = [
        topic["documents"] / document_count * 1000
        for topic in topics
    ]
    tenths = [int(value) for value in exact_tenths]
    remainder = 1000 - sum(tenths)
    remainder_order = sorted(
        range(len(topics)),
        key=lambda index: exact_tenths[index] - tenths[index],
        reverse=True,
    )
    for index in remainder_order[:remainder]:
        tenths[index] += 1
    for topic, value in zip(topics, tenths):
        topic["percentage"] = value / 10
    return topics


def _write_charts(
    output_dir: Path,
    big_five: list[dict[str, Any]],
    topics: list[dict[str, Any]],
    embeddings: np.ndarray,
    insights: dict[str, Any],
) -> list[str]:
    paths: list[str] = []
    angles = np.linspace(0, 2 * np.pi, len(big_five), endpoint=False).tolist()
    values = [item["score"] if item["score"] is not None else 0 for item in big_five]
    angles += angles[:1]
    values += values[:1]
    figure, axis = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    axis.plot(angles, values, linewidth=2)
    axis.fill(angles, values, alpha=0.25)
    axis.set_ylim(0, 100)
    axis.set_xticks(angles[:-1], [item["trait"].title() for item in big_five])
    axis.set_title("Distribución relativa de señales Big Five (suma 100 %)")
    path = output_dir / "big_five_radar.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    paths.append(str(path))

    content_types = insights["content_types"]
    if content_types:
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.bar(
            [str(item.get("type") or "sin clasificar") for item in content_types],
            [item["videos"] for item in content_types],
            color="#526dca",
        )
        axis.set_title("Tipos de contenido publicados")
        axis.set_ylabel("Videos")
        axis.tick_params(axis="x", rotation=25)
        path = output_dir / "content_types.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        paths.append(str(path))

    reactions = insights["reactions"]
    figure, axis = plt.subplots(figsize=(8, 4))
    reaction_names = [
        name for name in ("views", "likes", "comments", "shares", "saves")
        if reactions[name] is not None
    ]
    axis.bar(reaction_names, [reactions[name] for name in reaction_names], color="#805ad5")
    axis.set_title("Reacciones y alcance acumulado")
    axis.set_ylabel("Cantidad")
    path = output_dir / "reactions.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    if reaction_names:
        paths.append(str(path))
    else:
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar([item["label"] for item in topics], [item["documents"] for item in topics])
    axis.set_ylabel("Publicaciones")
    axis.set_title("Agrupación semántica de temas")
    axis.tick_params(axis="x", rotation=30)
    path = output_dir / "topic_distribution.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    paths.append(str(path))

    if topics:
        ordered_topics = sorted(topics, key=lambda item: item["percentage"])
        figure, axis = plt.subplots(
            figsize=(10, max(4, len(ordered_topics) * 0.9))
        )
        bars = axis.barh(
            [item["label"] for item in ordered_topics],
            [item["percentage"] for item in ordered_topics],
            color="#526dca",
        )
        axis.set_xlim(0, max(100, max(item["percentage"] for item in ordered_topics) + 20))
        axis.set_xlabel("% de videos por tema")
        axis.set_title("Intereses y cobertura temática de los videos")
        axis.set_xticks(np.arange(0, 101, 20))
        axis.bar_label(
            bars,
            labels=[
                f"{item['percentage']:.1f}% ({item['documents']} videos)"
                for item in ordered_topics
            ],
            padding=4,
        )
        figure.tight_layout()
        path = output_dir / "interest_map.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        paths.append(str(path))
    return paths


def analyze_profile(data: dict[str, Any], output_dir: str | Path = "output/analysis") -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    enriched_data = {**data, "videos": _enrich_dom_videos(data)}
    texts = _texts(enriched_data)
    video_texts = _video_texts(enriched_data)
    entities = _entities(texts)
    insights = _content_insights(enriched_data)
    if video_texts:
        anchor_texts = [
            anchor
            for anchors in TRAIT_ANCHORS.values()
            for anchor in anchors
        ]
        all_embeddings, embedding_method = _embeddings([*video_texts, *anchor_texts])
        embeddings = all_embeddings[: len(video_texts)]
        big_five = _big_five(
            video_texts,
            embeddings,
            all_embeddings[len(video_texts) :],
        )
        topics = _topics(video_texts, embeddings)
    else:
        embeddings, embedding_method, topics = np.empty((0, 0)), "not-available", []
        big_five = [
            {
                "trait": trait,
                "score": None,
                "positive_similarity": None,
                "negative_similarity": None,
                "evidence_words": 0,
                "evidence_documents": 0,
                "confidence": 0.0,
                "status": "insuficiente",
            }
            for trait in TRAIT_ANCHORS
        ]
    charts = (
        _write_charts(output_path, big_five, topics, embeddings, insights)
        if video_texts
        else []
    )
    expected_videos = (data.get("profile") or {}).get("videos_count")
    videos_extracted = len(data.get("videos") or [])
    is_partial = bool(expected_videos and videos_extracted < expected_videos)
    extraction = {
        "status": "partial" if is_partial else "complete",
        "expected_videos": expected_videos,
        "videos_extracted": videos_extracted,
        "message": (
            f"TikTok entregó {videos_extracted} de {expected_videos} publicaciones; "
            "el análisis usa las publicaciones visibles cargadas."
            if is_partial
            else None
        ),
    }
    result = {
        "methodology": {
            "big_five": (
                "Distribución relativa normalizada de similitud semántica entre los videos "
                "publicados y descripciones de los cinco rasgos (suma 100 %); no evalúa "
                "la personalidad real de la persona."
            ),
            "entities": "Hashtags, menciones y nombres propios detectados por reglas.",
            "embeddings": embedding_method,
            "semantic_clustering": "KMeans sobre embeddings",
            "topics": (
                "Porcentaje de videos asignados a cada grupo semántico; los términos "
                "representativos describen cobertura del contenido y no miden conocimiento."
            ),
        },
        "big_five": big_five,
        "entities": entities,
        "topics": topics,
        "charts": charts,
        "text_count": len(video_texts),
        "content_insights": insights,
        "extraction": extraction,
    }
    with (output_path / "analysis.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    with (output_path / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["tipo", "nombre", "valor"])
        writer.writerows(
            ("big_five", item["trait"], item["score"] if item["score"] is not None else "no concluyente")
            for item in big_five
        )
        writer.writerows(                (
            "topic",
            item["label"],
            f"{item['percentage']:.1f}% ({item['documents']} videos)",
        )
        for item in topics)
        writer.writerows(
            ("content_type", item["type"], item["videos"])
            for item in insights["content_types"]
        )
        writer.writerows(
            ("reaction", name, value)
            for name, value in insights["reactions"].items()
        )
    result["web_report"] = str(output_path / "report.html")
    write_html_report(enriched_data, result, output_path)
    return result


def format_results_table(result: dict[str, Any]) -> str:
    lines = ["| Tipo | Resultado | Valor |", "|---|---|---:|"]
    for item in result["big_five"]:
        score = f"{item['score']:.1f}%" if item["score"] is not None else "No concluyente"
        lines.append(f"| Big Five | {item['trait']} | {score} |")
    if result["big_five"] and all(item["score"] is not None for item in result["big_five"]):
        total = sum(item["score"] for item in result["big_five"])
        lines.append(f"| Big Five | Total | {total:.1f}% |")
    for item in result["topics"]:
        lines.append(f"| Tema | {item['label']} | {item['documents']} documentos |")
    return "\n".join(lines)


def write_html_report(
    data: dict[str, Any], result: dict[str, Any], output_dir: Path
) -> Path:
    profile = data.get("profile") or {}
    videos = data.get("videos") or []
    extraction = result.get("extraction") or {}
    rows = [
        ("Perfil", "Usuario", profile.get("username")),
        ("Perfil", "Nombre", profile.get("nickname")),
        ("Perfil", "Biografía", profile.get("bio")),
        ("Perfil", "Región", profile.get("region")),
        ("Perfil", "Idioma", profile.get("language")),
        ("Perfil", "Enlace", profile.get("bio_link")),
        ("Perfil", "Verificado", profile.get("verified")),
        ("Perfil", "Siguiendo", profile.get("following")),
        ("Perfil", "Seguidores", profile.get("followers")),
        ("Perfil", "Me gusta", profile.get("likes")),
        ("Perfil", "Videos", profile.get("videos_count")),
    ]
    rows.extend(
        (
            "Video",
            f"{video.get('id') or index + 1}: {video.get('description') or '(sin descripción)'}",
            f"vistas={video.get('views') or 0}, likes={video.get('likes') or 0}, "
            f"comentarios={video.get('comments_count') or video.get('comments') or 0}, "
            f"compartidos={video.get('shares') or 0}, "
            f"guardados={video.get('saves') or 0}, duración={video.get('duration_seconds') or 0}s, "
            f"música={video.get('music_title') or '(sin datos)'}, "
            f"hashtags={', '.join(video.get('hashtags') or []) or '(ninguno)'}",
        )
        for index, video in enumerate(videos)
    )
    comments = _valid_comments(data)
    rows.extend(
        (
            "Comentario",
            f"{comment.get('username') or comment.get('nickname') or 'anónimo'}: "
            f"{comment.get('text') or ''}",
            f"likes={comment.get('likes') or 0}",
        )
        for comment in comments[:50]
    )
    table_rows = "\n".join(
        f"<tr><td>{html.escape(str(kind))}</td>"
        f"<td>{html.escape(str(name or ''))}</td>"
        f"<td>{html.escape(str(value or ''))}</td></tr>"
        for kind, name, value in rows
    )
    bar_rows = []
    for item in result["big_five"]:
        score = float(item["score"] or 0)
        score_label = f"{score:.1f}%" if item["score"] is not None else "No concluyente"
        bar_rows.append(
            f'<div class="bar-row"><div class="trait-name">'
            f'{html.escape(item["trait"].title())}</div>'
            f'<div class="bar-track" role="progressbar" aria-valuenow="{score:.1f}" '
            f'aria-valuemin="0" aria-valuemax="100" aria-label="{html.escape(item["trait"].title())}">'
            f'<div class="bar" style="width:{score:.2f}%"></div></div>'
            f'<div class="score">{score_label}</div></div>'
        )
    big_five_table_rows = []
    for item in result["big_five"]:
        score_text = (
            f"{float(item['score']):.1f}%"
            if item["score"] is not None
            else "No concluyente"
        )
        big_five_table_rows.append(
            f"<tr><td>{html.escape(item['trait'].title())}</td>"
            f"<td>{score_text}</td>"
            f"<td>{item.get('evidence_words', 0)} palabras; "
            f"confianza {item.get('confidence', 0):.0%}</td></tr>"
        )
    big_five_table = "".join(big_five_table_rows)
    big_five_total = (
        f"<tr><th>Total</th><th>{sum(item['score'] for item in result['big_five'] if item['score'] is not None):.1f}%</th>"
        "<th>Distribución relativa</th></tr>"
        if not any(item["score"] is None for item in result["big_five"])
        else ""
    )
    insufficient_big_five = not any(
        item["score"] is not None for item in result["big_five"]
    )
    big_five_notice = (
        '<div class="warning">No hay suficiente texto temático para calcular señales semánticas. '
        "Se necesitan al menos 20 palabras analizables y varias publicaciones extraídas. "
        "Las puntuaciones se muestran como <strong>No concluyente</strong> para evitar "
        "inferencias no sustentadas. Estas señales no representan la personalidad real.</div>"
        if insufficient_big_five
        else ""
    )
    charts_by_name = {Path(path).name: Path(path).name for path in result["charts"]}
    map_image = charts_by_name.get("interest_map.png")
    map_markup = (
        f'<img src="{html.escape(map_image)}" alt="Mapa de intereses y conocimiento temático">'
        if map_image
        else "<p>No hay suficientes textos para construir el mapa temático.</p>"
    )
    topic_rows = "".join(
        f"<tr><td>{html.escape(str(item['label']))}</td>"
        f"<td>{item['percentage']:.1f}%</td>"
        f"<td>{item['documents']}</td></tr>"
        for item in result["topics"]
    ) or "<tr><td colspan='3'>No hay temas identificables en los videos extraídos.</td></tr>"
    insights = result["content_insights"]
    type_rows = "".join(
        f"<tr><td>{html.escape(item['type'])}</td><td>{item['videos']}</td></tr>"
        for item in insights["content_types"]
    ) or "<tr><td colspan='2'>No disponible</td></tr>"
    profile_interest_markup = ", ".join(
        html.escape(str(item)) for item in insights["profile_interests"]
    ) or "No se pudieron inferir intereses adicionales de la biografía."
    reaction_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>"
        f"{value if value is not None else 'No disponible'}</td></tr>"
        for name, value in insights["reactions"].items()
    )
    comment_rows = "".join(
        f"<tr><td>{html.escape(str(item.get('username') or item.get('nickname') or 'anónimo'))}</td>"
        f"<td>{html.escape(str(item.get('text') or ''))}</td><td>{item.get('likes') or 0}</td></tr>"
        for item in insights["comments_sample"]
    ) or "<tr><td colspan='3'>No se encontraron comentarios públicos en la página.</td></tr>"
    chart_names = {Path(path).name for path in result["charts"]}
    content_chart = (
        '<img src="content_types.png" alt="Tipos de contenido">'
        if "content_types.png" in chart_names
        else "<p>No hay tipos de contenido clasificados.</p>"
    )
    reactions_chart = (
        '<img src="reactions.png" alt="Reacciones y alcance">'
        if "reactions.png" in chart_names and any(insights["reactions"].values())
        else "<p>No hay métricas individuales de reacciones disponibles.</p>"
    )
    extraction_notice = ""
    if extraction.get("status") == "partial":
        extraction_notice = (
            '<div class="warning"><strong>Extracción parcial:</strong> '
            f'{html.escape(str(extraction.get("message") or ""))} '
            "Ejecuta Playwright en modo visible, resuelve el WAF y vuelve a generar el informe."
            "</div>"
        )
    report = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Análisis de perfil TikTok</title>
<style>
body {{ font-family: system-ui, sans-serif; background:#f5f7fb; color:#172033; margin:0; }}
main {{ max-width:1100px; margin:0 auto; padding:32px 20px 64px; }}
section {{ background:white; border-radius:14px; padding:24px; margin:20px 0; box-shadow:0 3px 18px #17203314; }}
h1 {{ margin-top:0; }} h2 {{ color:#334e9b; }}
.table-wrap {{ overflow-x:auto; }} table {{ border-collapse:collapse; width:100%; }}
th, td {{ border-bottom:1px solid #e4e8f0; padding:11px; text-align:left; vertical-align:top; }}
th {{ background:#eef2ff; }} .bar-row {{ display:flex; align-items:center; gap:14px; margin:16px 0; }}
.trait-name {{ width:160px; font-weight:600; }} .bar-track {{ flex:1; height:20px; background:#e8ecf5; border-radius:8px; overflow:hidden; }}
.bar {{ background:linear-gradient(90deg,#526dca,#805ad5); height:100%; border-radius:8px; }}
.score {{ width:70px; text-align:right; font-weight:700; color:#334e9b; }}
.columns {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:24px; }}
.columns img {{ max-width:100%; height:auto; margin-top:14px; border-radius:10px; }}
.map {{ text-align:center; }} .map img {{ max-width:100%; height:auto; border:1px solid #e4e8f0; border-radius:10px; }}
.note {{ color:#5e6b85; font-size:.92rem; }}
.warning {{ background:#fff4d6; border-left:5px solid #d69e2e; padding:14px; margin:14px 0; border-radius:6px; }}
</style>
</head>
<body><main>
<h1>Análisis del contenido de TikTok</h1>
<p class="note">Fuente: {html.escape(str(data.get("source") or ""))}</p>
{extraction_notice}
<section><h2>1. Datos extraídos</h2>
<div class="table-wrap"><table><thead><tr><th>Tipo</th><th>Campo</th><th>Valor</th></tr></thead>
<tbody>{table_rows}</tbody></table></div></section>
<section><h2>2. Big Five</h2>
<p class="note">Señales semánticas exploratorias de las descripciones, hashtags y música de los videos extraídos. Los cinco porcentajes se normalizan para sumar 100 % y no representan una evaluación de la personalidad real.</p>
{big_five_notice}
<div class="table-wrap"><table><thead><tr><th>Rasgo</th><th>Distribución (%)</th><th>Evidencia</th></tr></thead>
<tbody>{big_five_table}{big_five_total}</tbody></table></div>
<h3>Distribución relativa de señales (suma 100 %)</h3>
{''.join(bar_rows)}</section>
<section><h2>Intereses, tipos de contenido y reacciones</h2>
<p><strong>Intereses inferidos de la biografía:</strong> {profile_interest_markup}</p>
<div class="columns"><div><h3>Tipos de contenido</h3>
<table><thead><tr><th>Tipo</th><th>Videos</th></tr></thead><tbody>{type_rows}</tbody></table>
{content_chart}</div><div><h3>Reacciones acumuladas</h3>
<table><thead><tr><th>Métrica</th><th>Total</th></tr></thead><tbody>{reaction_rows}</tbody></table>
{reactions_chart}</div></div>
<h3>Comentarios públicos y reacción de los usuarios</h3>
<table><thead><tr><th>Usuario</th><th>Comentario</th><th>Likes</th></tr></thead>
<tbody>{comment_rows}</tbody></table>
</section>
<section class="map"><h2>3. Mapa de intereses y conocimiento temático</h2>
<p class="note">Cada porcentaje muestra qué parte de los videos extraídos corresponde a ese tema; juntos suman 100 %. Los términos del tema describen la cobertura temática observada, no miden cuánto sabe la persona sobre ellos.</p>
{map_markup}
<div class="table-wrap"><table><thead><tr><th>Interés / tema y términos representativos</th><th>Porcentaje de videos</th><th>Videos</th></tr></thead>
<tbody>{topic_rows}</tbody></table></div>
</section>
</main></body></html>"""
    report_path = output_dir / "report.html"
    report_path.write_text(report, encoding="utf-8")
    return report_path
