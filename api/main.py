# =============================================================================
# What this file does:
# The FastAPI web server that wraps our FinReg-RAG pipeline and exposes it
# as REST API endpoints that the React frontend can call.
#
# Endpoints:
#   POST /query      — runs a question through the full RAG pipeline
#   GET  /analytics  — returns performance stats from SQLite
#   GET  /health     — simple health check so frontend knows server is running
#
# Run with: uvicorn api.main:app --reload --port 8000
# =============================================================================

import sys                          # for modifying Python's module search path
import os                           # for file path operations
import time                         # for measuring latency
import sqlite3                      # for reading analytics from SQLite
import json                         # for parsing JSON fields from SQLite

from fastapi import FastAPI, HTTPException   # FastAPI core + error handling
from fastapi.middleware.cors import CORSMiddleware  # allows React to call our API

# --- PATH SETUP ---
# Add project root and phase folders so we can import our pipeline modules
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "phase2"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "phase3"))

# --- IMPORT OUR PIPELINE ---
from hybrid_retriever import hybrid_search           # BM25 + vector + RRF
from reranker import rerank_chunks                   # Cohere reranking
from citation_enforcer import generate_cited_answer  # cited LLM generation
from prompt_loader import get_model_config           # YAML config
from logger import QueryLogger                       # SQLite + JSONL logger

# --- IMPORT OUR DATA MODELS ---
from api.models import (                             # all request/response shapes
    QueryRequest, QueryResponse,
    ChunkSource, LatencyBreakdown,
    AnalyticsResponse
)

# --- DATABASE PATH ---
DB_PATH = os.path.join(_PROJECT_ROOT, "logs", "finreg_queries.db")  # SQLite path

# --- CREATE FASTAPI APP ---
app = FastAPI(                                       # create the FastAPI application
    title="FinReg-RAG API",                          # shown in auto-generated docs
    description="Production RAG API for financial regulatory documents",
    version="1.0.0"                                  # API version
)

# --- CORS MIDDLEWARE ---
# CORS (Cross-Origin Resource Sharing) is a browser security rule.
# Without this, the browser blocks React (port 5173) from calling
# our API (port 8000) because they're on different ports.
# This middleware tells the browser: "yes, React is allowed to call us."
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],          # React dev server address
    allow_credentials=True,                          # allow cookies if needed
    allow_methods=["*"],                             # allow GET, POST, etc.
    allow_headers=["*"],                             # allow all headers
)

# --- INITIALIZE LOGGER ---
logger = QueryLogger()                               # creates DB if not exists


# =============================================================================
# ENDPOINT 1: Health Check
# GET /health
# React calls this on startup to verify the backend is running.
# =============================================================================
@app.get("/health")                                  # GET request to /health
def health_check():
    """Returns OK if the server is running."""
    return {"status": "ok", "message": "FinReg-RAG API is running"}


# =============================================================================
# ENDPOINT 2: Query
# POST /query
# React sends {"question": "..."} and gets back the full RAG response.
# =============================================================================
@app.post("/query", response_model=QueryResponse)    # POST request to /query
def run_query(request: QueryRequest):
    """
    Runs a question through the complete FinReg-RAG pipeline:
    hybrid retrieval → Cohere reranking → cited generation → logging.
    Returns the answer, sources, and latency breakdown as JSON.
    """
    question = request.question.strip()              # clean the input question

    if not question:                                 # reject empty questions
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # --- STAGE 1: HYBRID RETRIEVAL ---
        t0 = time.perf_counter()                     # start retrieval timer
        hybrid_chunks = hybrid_search(question, top_k=5)   # BM25 + vector + RRF
        retrieval_ms = (time.perf_counter() - t0) * 1000  # convert to ms

        # --- STAGE 2: COHERE RERANKING ---
        t1 = time.perf_counter()                     # start reranking timer
        reranked = rerank_chunks(question, hybrid_chunks, top_n=3)  # rerank
        reranking_ms = (time.perf_counter() - t1) * 1000  # convert to ms

        # --- STAGE 3: CITED GENERATION ---
        t2 = time.perf_counter()                     # start generation timer
        result = generate_cited_answer(question, reranked)  # generate answer
        generation_ms = (time.perf_counter() - t2) * 1000  # convert to ms

        total_ms = retrieval_ms + reranking_ms + generation_ms  # total time

        # --- STAGE 4: LOGGING ---
        model_config = get_model_config()            # load YAML config
        logger.log_query(                            # write to SQLite + JSONL
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
        # Extract source info from each chunk for the Sources Panel
        sources = []                                 # empty list to fill
        for i, chunk in enumerate(result["chunks_used"]):   # loop through chunks
            source = chunk.get("source", "unknown")          # get filename
            page = chunk.get("page", "unknown")              # get page number
            if source == "unknown" and "metadata" in chunk:  # check nested metadata
                source = chunk["metadata"].get("source", "unknown")
                page = chunk["metadata"].get("page", "unknown")
            text_preview = chunk.get("text", "")[:200]       # first 200 chars

            sources.append(ChunkSource(              # build ChunkSource object
                citation_number=i + 1,               # [1], [2], [3]
                source=source,                       # document filename
                page=str(page),                      # page as string
                text_preview=text_preview,           # chunk preview
                cohere_score=chunk.get("cohere_score")  # relevance score
            ))

        # --- BUILD AND RETURN RESPONSE ---
        return QueryResponse(
            answer=result["answer"],                 # the cited answer text
            refused=result["refused"],               # True if refused
            sources=sources,                         # list of ChunkSource objects
            latency=LatencyBreakdown(                # latency breakdown
                retrieval_ms=round(retrieval_ms, 0),
                reranking_ms=round(reranking_ms, 0),
                generation_ms=round(generation_ms, 0),
                total_ms=round(total_ms, 0)
            ),
            query=question                           # echo the original question
        )

    except Exception as e:                           # catch any pipeline error
        raise HTTPException(status_code=500, detail=str(e))  # return error to React


# =============================================================================
# ENDPOINT 3: Analytics
# GET /analytics
# React calls this to populate the analytics dashboard tab.
# =============================================================================
@app.get("/analytics", response_model=AnalyticsResponse)  # GET /analytics
def get_analytics():
    """
    Reads query history and performance stats from SQLite.
    Returns aggregated metrics and recent query list for the dashboard.
    """
    if not os.path.exists(DB_PATH):                  # check database exists
        raise HTTPException(status_code=404, detail="No query data yet. Run some queries first.")

    try:
        conn = sqlite3.connect(DB_PATH)              # open SQLite connection
        cursor = conn.cursor()                       # create cursor

        # --- Aggregate stats ---
        cursor.execute("SELECT COUNT(*) FROM queries")
        total = cursor.fetchone()[0]                 # total query count

        cursor.execute("SELECT COUNT(*) FROM queries WHERE refused = 1")
        refused_count = cursor.fetchone()[0]         # refused count

        answered = total - refused_count             # answered count
        refusal_rate = (refused_count / total * 100) if total > 0 else 0.0

        # --- Average latencies ---
        cursor.execute("""
            SELECT
                AVG(total_latency_ms),
                AVG(retrieval_latency_ms),
                AVG(rerank_latency_ms),
                AVG(generation_latency_ms),
                AVG(top_chunk_score)
            FROM queries WHERE refused = 0
        """)
        row = cursor.fetchone()                      # get aggregates row
        avg_total = row[0] or 0.0                    # handle None if no answered queries
        avg_retrieval = row[1] or 0.0
        avg_reranking = row[2] or 0.0
        avg_generation = row[3] or 0.0
        avg_score = row[4] or 0.0

        # --- Recent queries for history table ---
        cursor.execute("""
            SELECT query, answer, total_latency_ms, refused, timestamp, top_chunk_score
            FROM queries
            ORDER BY id DESC
            LIMIT 10
        """)
        rows = cursor.fetchall()                     # get last 10 queries
        conn.close()                                 # close connection

        # Build list of recent query dicts
        recent = []                                  # empty list
        for row in rows:                             # loop through rows
            recent.append({
                "query": row[0][:80] + "..." if len(row[0]) > 80 else row[0],
                "answer_preview": row[1][:100] + "..." if len(row[1]) > 100 else row[1],
                "latency_ms": round(row[2] or 0, 0),
                "refused": bool(row[3]),
                "timestamp": row[4][:19].replace("T", " "),  # clean timestamp
                "cohere_score": round(row[5] or 0, 3) if row[5] else None
            })

        return AnalyticsResponse(                    # return analytics response
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