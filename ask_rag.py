#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from typing import List, Dict, Any, Tuple
from flask import Flask, request, jsonify
from annoy import AnnoyIndex
from openai import OpenAI

# =============================
# Configuración
# =============================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "rag_index")
INDEX_PATH = os.path.join(DATA_DIR, "annoy.index")
META_PATH  = os.path.join(DATA_DIR, "metadata.jsonl")
CONF_PATH  = os.path.join(DATA_DIR, "config.json")

CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4.1")
EMB_MODEL  = os.environ.get("EMB_MODEL", "text-embedding-3-large")

client = OpenAI()  # usa OPENAI_API_KEY del entorno

app = Flask(__name__)

# =============================
# Carga de índice y metadatos
# =============================
ready = False
dim = 0
metric = "angular"
ann: AnnoyIndex | None = None
metadata: List[Dict[str, Any]] = []

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

try:
    # config
    with open(CONF_PATH, "r", encoding="utf-8") as f:
        conf = json.load(f)
    dim = int(conf.get("dim", 3072))
    metric = conf.get("metric", "angular")

    # annoy
    ann = AnnoyIndex(dim, metric)
    ann.load(INDEX_PATH)

    # metadata
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    metadata.append(json.loads(line))
                except Exception:
                    pass

    ready = True
except FileNotFoundError as e:
    app.logger.error(f"[INIT] Faltan archivos del índice: {e}")
except Exception as e:
    app.logger.exception(f"[INIT] Error cargando índice/metadata: {e}")

# =============================
# Utilidades RAG
# =============================
def embed(text: str) -> List[float]:
    """Crea embedding con OpenAI."""
    resp = client.embeddings.create(model=EMB_MODEL, input=text)
    return resp.data[0].embedding

def annoy_search(vec: List[float], k: int) -> Tuple[List[int], List[float]]:
    """Devuelve (ids, distances) desde Annoy."""
    assert ann is not None
    ids, dists = ann.get_nns_by_vector(vec, k, include_distances=True)
    return ids, dists

def build_sources(hits: List[Tuple[int, float]]) -> List[Dict[str, Any]]:
    """Convierte (idx, dist) en objetos fuente con score, título, url, etc."""
    out: List[Dict[str, Any]] = []
    for local_id, dist in hits:
        meta = metadata[local_id] if 0 <= local_id < len(metadata) else {}
        # score simple derivado de la distancia (cuanto menor la distancia, mayor score)
        score = 1.0 / (1.0 + _safe_float(dist, 0.0))
        out.append({
            "id": int(meta.get("id", local_id)),
            "source": meta.get("source") or meta.get("type") or "doc",
            "title": (meta.get("title") or "").strip() or None,
            "url": (meta.get("url") or "").strip() or None,
            "updated_at": meta.get("updated_at"),
            "score": float(score),
        })
    return out

def build_context(hits: List[Tuple[int, float]]) -> str:
    """Concatena los trozos de texto de los top-k documentos para el prompt."""
    chunks: List[str] = []
    for local_id, _ in hits:
        meta = metadata[local_id] if 0 <= local_id < len(metadata) else {}
        text = (meta.get("text") or meta.get("body") or meta.get("chunk") or "").strip()
        title = (meta.get("title") or "").strip()
        if title:
            chunks.append(f"# {title}\n{text}")
        else:
            chunks.append(text)
    # Limita el contexto para no exceder tokens.
    joined = "\n\n---\n\n".join([c for c in chunks if c])
    return joined[:20000]  # corte conservador

def rag_answer(question: str, k: int) -> Tuple[str, List[Dict[str, Any]]]:
    """Búsqueda + generación con contexto."""
    q_vec = embed(question)
    idxs, dists = annoy_search(q_vec, k)
    hits = list(zip(idxs, dists))
    sources = build_sources(hits)
    context = build_context(hits)

    system = (
        "Eres un asistente técnico de soporte para máquinas FLUX (beamo, Beambox, etc.). "
        "Responde en español de forma práctica y paso a paso. "
        "Si no hay suficiente contexto, dilo claramente y sugiere qué comprobar."
    )
    user_content = (
        f"Pregunta del usuario:\n{question}\n\n"
        f"=== CONTEXTO (extractos relevantes) ===\n{context}\n\n"
        "Instrucciones:\n"
        "- Responde con pasos claros.\n"
        "- Cita modelos si procede.\n"
        "- Si hay un artículo oficial útil, menciónalo.\n"
    )

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
    )
    answer = resp.choices[0].message.content.strip()
    return answer, sources

# =============================
# Rutas
# =============================
@app.route("/health", methods=["GET"])
def health():
    if not ready:
        return jsonify({"status": "error", "message": "índice no cargado"}), 500
    return jsonify({"status": "ok", "model": CHAT_MODEL, "emb": EMB_MODEL})

@app.route("/ask", methods=["POST"])
def ask():
    # --- Auth por cabecera ---
    api_key  = request.headers.get("X-API-KEY")
    expected = os.environ.get("API_TOKEN")
    if not expected or api_key != expected:
        return jsonify({"error": "unauthorized"}), 401

    if not ready:
        return jsonify({"error": "index_not_ready"}), 503

    data = request.get_json(silent=True) or {}
    question = data.get("question") or data.get("q") or ""
    k = int(data.get("k") or 6)
    if not question.strip():
        return jsonify({"error": "Falta 'question'"}), 400
    k = max(1, min(k, 20))

    try:
        answer, sources = rag_answer(question.strip(), k)
    except Exception as e:
        app.logger.exception(f"[ASK] Error: {e}")
        return jsonify({"error": "rag_failure", "detail": str(e)}), 500

    # --- construir texto de fuentes ya formateado para Make (evita NaN) ---
    lines: List[str] = []
    for s in sources:
        title = (s.get("title") or "(sin título)").strip()
        src   = (s.get("source") or "doc").strip()
        url   = (s.get("url") or "").strip()
        try:
            score_str = f"{float(s.get('score', 0)):0.3f}"
        except Exception:
            score_str = "0.000"
        line = f"- {src} · {title} · score {score_str}"
        if url:
            line += f" · {url}"
        lines.append(line)
    sources_bulleted = "\n".join(lines) if lines else "Sin fuentes"

    return jsonify({
        "answer": answer,
        "sources": sources,
        "sources_text": sources_bulleted,
        "k": k
    })

# =============================
# Main (local)
# =============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    debug = bool(os.environ.get("FLASK_DEBUG"))
    app.run(host="0.0.0.0", port=port, debug=debug)

