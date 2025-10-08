#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json
from typing import List, Dict, Any
from flask import Flask, request, jsonify
from annoy import AnnoyIndex
from openai import OpenAI
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.environ.get("RAG_DIR", os.path.join(BASE_DIR, "rag_index"))

CONF_PATH = os.path.join(RAG_DIR, "config.json")
ANNOY_PATH = os.path.join(RAG_DIR, "annoy.index")
META_PATH  = os.path.join(RAG_DIR, "metadata.jsonl")

# --- Rutas del índice ---
DATA_DIR   = "rag_index"
INDEX_PATH = os.path.join(DATA_DIR, "annoy.index")
META_PATH  = os.path.join(DATA_DIR, "metadata.jsonl")
CONF_PATH  = os.path.join(DATA_DIR, "config.json")

# --- Modelo LLM (puedes cambiar por GPT-5 si luego tienes acceso) ---
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Carga índice y metadatos una vez ---
with open(CONF_PATH, "r", encoding="utf-8") as f:
    conf = json.load(f)
EMB_MODEL = conf["emb_model"]
EMB_DIM   = conf["emb_dim"]

index = AnnoyIndex(EMB_DIM, "angular")
index.load(INDEX_PATH)

records: Dict[int, Dict[str, Any]] = {}
with open(META_PATH, "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        records[int(rec["idx"])] = rec

def embed_query(text: str):
    resp = client.embeddings.create(model=EMB_MODEL, input=[text])
    return resp.data[0].embedding

def retrieve(text: str, k: int = 6):
    vec = embed_query(text)
    ids, dists = index.get_nns_by_vector(vec, k, include_distances=True)
    results = []
    for i, dist in zip(ids, dists):
        r = records.get(i, {})
        meta = r.get("metadata", {})
        results.append({
            "idx": i,
            "score": float(1.0 / (1.0 + dist)),
            "text": r.get("text", ""),
            "title": meta.get("title"),
            "url": meta.get("url"),
            "updated_at": meta.get("updated_at"),
            "source": meta.get("source"),
            "category": meta.get("category"),
            "section": meta.get("section"),
        })
    return results

def build_prompt(question: str, contexts):
    ctx_blocks = []
    for i, c in enumerate(contexts, 1):
        head = f"[{i}] {c.get('title') or c.get('url') or c.get('source')}"
        meta = []
        if c.get("source"): meta.append(c["source"])
        if c.get("category"): meta.append(c["category"])
        if c.get("section"): meta.append(c["section"])
        if c.get("updated_at"): meta.append(f"updated: {c['updated_at']}")
        meta_str = " • ".join([m for m in meta if m])
        ctx_blocks.append(f"{head}\n{meta_str}\nURL: {c.get('url')}\n\n{c['text']}\n")
    context_text = "\n---\n".join(ctx_blocks[:8])

    system = (
        "Eres asistente técnico de FLUX. Responde conciso y accionable. "
        "CITA SIEMPRE las fuentes usando [n] con TÍTULO y URL. "
        "Prioriza artículos oficiales sobre tickets si hay conflicto. "
        "Si faltan datos, pide aclaraciones. No inventes."
    )
    user = f"Pregunta: {question}\n\nContexto recuperado:\n{context_text}"
    return [
        {"role":"system","content":system},
        {"role":"user","content":user},
    ]

app = Flask(__name__)
API_TOKEN = os.getenv("API_TOKEN", "")

def check_auth(req):
    # Espera el header: X-API-KEY: <tu_token>
    return API_TOKEN and req.headers.get("X-API-KEY") == API_TOKEN

# ...y en el endpoint /ask añade el check:
@app.route("/ask", methods=["POST"])
def ask():
    if not check_auth(request):
        return jsonify({"error": "unauthorized"}), 401
    # (resto de tu código igual)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True) or {}
    q = (data.get("q") or data.get("question") or "").strip()
    k = int(data.get("k") or 6)
    if not q:
        return jsonify({"error":"Falta 'q'"}), 400
    hits = retrieve(q, k=k)
    msgs = build_prompt(q, hits)
    chat = client.chat.completions.create(model=CHAT_MODEL, messages=msgs)
    answer = chat.choices[0].message.content

    sources = []
    for i, h in enumerate(hits, 1):
        sources.append({
            "id": i,
            "title": h.get("title"),
            "url": h.get("url"),
            "updated_at": h.get("updated_at"),
            "source": h.get("source"),
            "score": h.get("score"),
        })
# --- Formateo de fuentes (título, URL y score redondeado) ---
sources_fmt = []
for s in sources:
    sources_fmt.append({
        "id": s.get("id"),
        "source": s.get("source"),
        "score": round(float(s.get("score", 0)), 3),
        "title": s.get("title") or s.get("article_title") or s.get("subject") or None,
        "url": s.get("url") or s.get("article_url") or None,
    })

return jsonify({"answer": answer, "sources": sources_fmt, "k": k})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok","model":CHAT_MODEL,"emb":EMB_MODEL})

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
