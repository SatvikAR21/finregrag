# =============================================================================
# What this file does:
# The FastAPI web server that wraps our FinReg-RAG pipeline and exposes it
# as REST API endpoints that the React frontend can call.
#
# Endpoints:
#   POST /query      — runs a question through the full RAG pipeline
#   GET  /analytics  — returns performance stats from SQLite
#   GET  /health     — simple health check so frontend knows server is running
#   POST /upload     — receives a PDF, runs ingestion pipeline, stores in ChromaDB
#   GET  /documents  — lists all documents currently stored in ChromaDB
#
# Run with: uvicorn api.main:app --reload --port 8000
# =============================================================================

import sys                          # for modifying Python's module search path
import os                           # for file path operations
import time                         # for measuring latency
import sqlite3                      # for reading analytics from SQLite
import json                         # for parsing JSON fields from SQLite
import shutil                       # for copying uploaded files to disk

from fastapi import FastAPI, HTTPException, File, UploadFile   # FastAPI core
from fastapi.middleware.cors import CORSMiddleware              # allows React to call our API
from api.startup import run_startup                  # startup ingestion check

# --- PATH SETUP ---
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "phase1"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "phase2"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "phase3"))

# --- IMPORT OUR PIPELINE ---
from hybrid_retriever import hybrid_search           # BM25 + vector + RRF
from reranker import rerank_chunks                   # Cohere reranking
from citation_enforcer import generate_cited_answer  # cited LLM generation
from prompt_loader import get_model_config           # YAML config
from logger import QueryLogger                       # SQLite + JSONL logger
from pipeline import ingest_document, get_ingested_documents  # upload pipeline

# --- IMPORT OUR DATA MODELS ---
from api.models import (
    QueryRequest, QueryResponse,
    ChunkSource, LatencyBreakdown,
    AnalyticsResponse
)

# --- DATABASE PATH ---
DB_PATH = os.path.join(_PROJECT_ROOT, "logs", "finreg_queries.db")

# --- CREATE FASTAPI APP ---
app = FastAPI(
    title="FinReg-RAG API",
    description="Production RAG API for financial regulatory documents",
    version="1.0.0"
)
# Run startup check — downloads and ingests Basel III if ChromaDB is empty
run_startup()
# --- CORS MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                                 # allows any frontend to call the API    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- INITIALIZE LOGGER ---
logger = QueryLogger()


# =============================================================================
# ENDPOINT 1: Health Check
# =============================================================================
@app.get("/health")
def health_check():
    """Returns OK if the server is running."""
    return {"status": "ok", "message": "FinReg-RAG API is running"}


# =============================================================================
# ENDPOINT 2: Query
# POST /query
# =============================================================================
@app.post("/query", response_model=QueryResponse)
def run_query(request: QueryRequest):
    """
    Runs a question through the complete FinReg-RAG pipeline:
    hybrid retrieval → Cohere reranking → cited generation → logging.
    """
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # --- STAGE 1: HYBRID RETRIEVAL ---
        t0 = time.perf_counter()
        hybrid_chunks = hybrid_search(question, top_k=5)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        # --- STAGE 2: COHERE RERANKING ---
        t1 = time.perf_counter()
        reranked = rerank_chunks(question, hybrid_chunks, top_n=3)
        reranking_ms = (time.perf_counter() - t1) * 1000

        # --- STAGE 3: CITED GENERATION ---
        t2 = time.perf_counter()
        result = generate_cited_answer(question, reranked)
        generation_ms = (time.perf_counter() - t2) * 1000

        total_ms = retrieval_ms + reranking_ms + generation_ms

        # --- STAGE 4: LOGGING ---
        model_config = get_model_config()
        logger.log_query(
            query=question,
            answer=result["answer"],
            chunks_used=result["chunks_used"],
            refused=result["refused"],
            retrieval_latency_ms=retrieval_ms,
            rerank_latency_ms=reranking_ms,
            generation_latency_ms=generation_ms,
            prompt_version=model_config.get("prompt_version", "2.0"),
            model_used=model_config.get("model", "llama-3.3-70b-versatile")
        )

        # --- BUILD SOURCE LIST ---
        sources = []
        for i, chunk in enumerate(result["chunks_used"]):
            source = chunk.get("source", "unknown")
            page = chunk.get("page", "unknown")
            if source == "unknown" and "metadata" in chunk:
                source = chunk["metadata"].get("source", "unknown")
                page = chunk["metadata"].get("page", "unknown")
            text_preview = chunk.get("text", "")[:200]

            sources.append(ChunkSource(
                citation_number=i + 1,
                source=source,
                page=str(page),
                text_preview=text_preview,
                cohere_score=chunk.get("cohere_score")
            ))

        return QueryResponse(
            answer=result["answer"],
            refused=result["refused"],
            sources=sources,
            latency=LatencyBreakdown(
                retrieval_ms=round(retrieval_ms, 0),
                reranking_ms=round(reranking_ms, 0),
                generation_ms=round(generation_ms, 0),
                total_ms=round(total_ms, 0)
            ),
            query=question
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ENDPOINT 3: Analytics
# GET /analytics
# =============================================================================
@app.get("/analytics", response_model=AnalyticsResponse)
def get_analytics():
    """Reads query history and performance stats from SQLite."""
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=404, detail="No query data yet. Run some queries first.")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM queries")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM queries WHERE refused = 1")
        refused_count = cursor.fetchone()[0]

        answered = total - refused_count
        refusal_rate = (refused_count / total * 100) if total > 0 else 0.0

        cursor.execute("""
            SELECT
                AVG(total_latency_ms),
                AVG(retrieval_latency_ms),
                AVG(rerank_latency_ms),
                AVG(generation_latency_ms),
                AVG(top_chunk_score)
            FROM queries WHERE refused = 0
        """)
        row = cursor.fetchone()
        avg_total      = row[0] or 0.0
        avg_retrieval  = row[1] or 0.0
        avg_reranking  = row[2] or 0.0
        avg_generation = row[3] or 0.0
        avg_score      = row[4] or 0.0

        cursor.execute("""
            SELECT query, answer, total_latency_ms, refused, timestamp, top_chunk_score
            FROM queries
            ORDER BY id DESC
            LIMIT 10
        """)
        rows = cursor.fetchall()
        conn.close()

        recent = []
        for row in rows:
            recent.append({
                "query": row[0][:80] + "..." if len(row[0]) > 80 else row[0],
                "answer_preview": row[1][:100] + "..." if len(row[1]) > 100 else row[1],
                "latency_ms": round(row[2] or 0, 0),
                "refused": bool(row[3]),
                "timestamp": row[4][:19].replace("T", " "),
                "cohere_score": round(row[5] or 0, 3) if row[5] else None
            })

        return AnalyticsResponse(
            total_queries=total,
            answered=answered,
            refused=refused_count,
            refusal_rate=round(refusal_rate, 1),
            avg_total_ms=round(avg_total, 0),
            avg_retrieval_ms=round(avg_retrieval, 0),
            avg_reranking_ms=round(avg_reranking, 0),
            avg_generation_ms=round(avg_generation, 0),
            avg_cohere_score=round(avg_score, 4),
            recent_queries=recent
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ENDPOINT 4: Upload Document
# POST /upload
# =============================================================================
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Accepts a PDF upload, runs ingest → chunk → embed pipeline,
    stores chunks in ChromaDB, returns ingestion summary.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    raw_dir   = os.path.join(_PROJECT_ROOT, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    save_path = os.path.join(raw_dir, file.filename)

    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        result = ingest_document(save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    if not result["success"]:
        raise HTTPException(status_code=422, detail=result["message"])

    return result


# =============================================================================
# ENDPOINT 5: List Documents
# GET /documents
# =============================================================================
@app.get("/documents")
def list_documents():
    """Returns all ingested documents and their chunk counts."""
    try:
        docs = get_ingested_documents()
        return {"documents": docs, "total": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))