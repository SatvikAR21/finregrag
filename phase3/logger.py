# =============================================================================
# What this file does:
# Provides a QueryLogger class that records every RAG query to two places:
# 1. A SQLite database (finreg_queries.db) — structured, queryable
# 2. A JSONL file (finreg_queries.jsonl) — one JSON record per line
#
# Every query record captures: timestamp, question, answer, latency per step,
# chunks retrieved, Cohere scores, whether the system refused, and prompt version.
# =============================================================================

import sqlite3        # built-in Python library for SQLite databases
import json           # for writing JSON log records
import os             # for building file paths
import time           # for measuring latency
from datetime import datetime, timezone   # for timestamps

# --- CONFIGURATION ---
# All logs go into a top-level 'logs' folder
LOGS_DIR = os.path.join("logs")                              # logs directory path
DB_PATH = os.path.join(LOGS_DIR, "finreg_queries.db")       # SQLite database file path
JSONL_PATH = os.path.join(LOGS_DIR, "finreg_queries.jsonl") # JSON log file path


def ensure_logs_dir():
    """Creates the logs/ directory if it doesn't already exist."""
    os.makedirs(LOGS_DIR, exist_ok=True)    # create directory, no error if already exists


def init_database():
    """
    Creates the SQLite database and the queries table if they don't exist yet.
    Safe to call multiple times — it only creates if missing.
    """
    ensure_logs_dir()                                        # make sure logs/ folder exists
    conn = sqlite3.connect(DB_PATH)                          # connect to (or create) the database
    cursor = conn.cursor()                                   # create a cursor to run SQL commands

    # Create the main queries table with all the columns we want to track
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            refused INTEGER NOT NULL,
            chunks_retrieved INTEGER NOT NULL,
            top_chunk_score REAL,
            retrieval_latency_ms REAL,
            rerank_latency_ms REAL,
            generation_latency_ms REAL,
            total_latency_ms REAL,
            prompt_version TEXT,
            model_used TEXT,
            chunk_sources TEXT
        )
    """)
    # Column explanations:
    # id — auto-incrementing unique row number
    # timestamp — when the query happened (ISO format string)
    # query — the user's question
    # answer — the LLM's response
    # refused — 1 if system refused to answer, 0 if it answered
    # chunks_retrieved — how many chunks were used
    # top_chunk_score — Cohere relevance score of the best chunk
    # retrieval_latency_ms — how long hybrid search took in milliseconds
    # rerank_latency_ms — how long Cohere reranking took in milliseconds
    # generation_latency_ms — how long LLM generation took in milliseconds
    # total_latency_ms — end-to-end time for the whole query
    # prompt_version — which YAML prompt version was used
    # model_used — which LLM model generated the answer
    # chunk_sources — JSON string listing source docs and pages used

    conn.commit()       # save the table creation to disk
    conn.close()        # close the database connection
    print(f"  ✅ Database initialized at: {DB_PATH}")


class QueryLogger:
    """
    Records every RAG query with full timing and metadata.
    Use this as a context manager or call log_query() directly.
    """

    def __init__(self):
        """Initialize the logger — creates DB and log file if needed."""
        ensure_logs_dir()       # make sure logs/ directory exists
        init_database()         # make sure the SQLite table exists

    def log_query(self,
                  query,
                  answer,
                  chunks_used,
                  refused,
                  retrieval_latency_ms,
                  rerank_latency_ms,
                  generation_latency_ms,
                  prompt_version="2.0",
                  model_used="llama-3.3-70b-versatile"):
        """
        Writes one complete query record to both SQLite and the JSONL file.

        Args:
            query: the user's question string
            answer: the LLM's answer string
            chunks_used: list of chunk dicts that informed the answer
            refused: True if the system refused to answer, False otherwise
            retrieval_latency_ms: time taken by hybrid search in milliseconds
            rerank_latency_ms: time taken by Cohere reranking in milliseconds
            generation_latency_ms: time taken by LLM generation in milliseconds
            prompt_version: version string from YAML config
            model_used: LLM model name string
        """
        # --- Calculate derived fields ---
        total_latency_ms = retrieval_latency_ms + rerank_latency_ms + generation_latency_ms
        timestamp = datetime.now(timezone.utc).isoformat()   # current UTC time as ISO string

        # Get the top Cohere score (best chunk's relevance score)
        top_score = None                                      # default if no chunks
        if chunks_used:                                       # if we have chunks
            scores = [c.get("cohere_score", 0) for c in chunks_used]  # extract all scores
            top_score = max(scores) if scores else None       # take the highest score

        # Build a list of source references for logging
        chunk_sources = []                                    # empty list to fill
        for chunk in chunks_used:                            # loop through used chunks
            source = chunk.get("source", "unknown")          # get source filename
            page = chunk.get("page", "?")                    # get page number
            if source == "unknown" and "metadata" in chunk:  # check nested metadata
                source = chunk["metadata"].get("source", "unknown")
                page = chunk["metadata"].get("page", "?")
            chunk_sources.append(f"{source}:p{page}")        # add "filename:pPage" string
        chunk_sources_str = json.dumps(chunk_sources)        # convert list to JSON string

        # --- Write to SQLite ---
        conn = sqlite3.connect(DB_PATH)                      # open database connection
        cursor = conn.cursor()                               # create cursor
        cursor.execute("""
            INSERT INTO queries (
                timestamp, query, answer, refused,
                chunks_retrieved, top_chunk_score,
                retrieval_latency_ms, rerank_latency_ms,
                generation_latency_ms, total_latency_ms,
                prompt_version, model_used, chunk_sources
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,                    # when
            query,                        # the question
            answer,                       # the answer
            1 if refused else 0,          # refused flag as integer (SQLite has no boolean)
            len(chunks_used),             # number of chunks used
            top_score,                    # best Cohere score
            retrieval_latency_ms,         # retrieval time
            rerank_latency_ms,            # reranking time
            generation_latency_ms,        # generation time
            total_latency_ms,             # total time
            prompt_version,               # YAML prompt version
            model_used,                   # LLM model name
            chunk_sources_str             # sources as JSON string
        ))
        conn.commit()                                        # save to disk
        conn.close()                                         # close connection

        # --- Write to JSONL file ---
        record = {                                           # build the JSON record
            "timestamp": timestamp,
            "query": query,
            "answer": answer,
            "refused": refused,
            "chunks_retrieved": len(chunks_used),
            "top_chunk_score": top_score,
            "retrieval_latency_ms": round(retrieval_latency_ms, 2),
            "rerank_latency_ms": round(rerank_latency_ms, 2),
            "generation_latency_ms": round(generation_latency_ms, 2),
            "total_latency_ms": round(total_latency_ms, 2),
            "prompt_version": prompt_version,
            "model_used": model_used,
            "chunk_sources": chunk_sources
        }

        with open(JSONL_PATH, "a", encoding="utf-8") as f:  # open in APPEND mode
            f.write(json.dumps(record) + "\n")               # write one JSON line

        print(f"  📝 Query logged | Total latency: {total_latency_ms:.0f}ms | "
              f"Refused: {refused} | Chunks: {len(chunks_used)}")


# --- QUICK TEST ---
if __name__ == "__main__":
    print("Initializing QueryLogger...")
    logger = QueryLogger()                                   # creates DB and log file

    print("\nLogging a test query...")
    logger.log_query(                                        # log a fake test record
        query="What is the minimum CET1 capital ratio?",
        answer="The minimum CET1 ratio is 4.5% [2].",
        chunks_used=[{"cohere_score": 0.99, "source": "basel3.pdf", "page": 28}],
        refused=False,
        retrieval_latency_ms=320.5,
        rerank_latency_ms=450.2,
        generation_latency_ms=1200.8
    )

    print("\nReading back from database to verify...")
    conn = sqlite3.connect(DB_PATH)                          # open database
    cursor = conn.cursor()                                   # create cursor
    cursor.execute("SELECT id, timestamp, query, total_latency_ms, refused FROM queries")
    rows = cursor.fetchall()                                 # get all rows
    conn.close()                                             # close connection

    print(f"\nFound {len(rows)} record(s) in database:")
    for row in rows:                                         # print each row
        print(f"  ID:{row[0]} | {row[1][:19]} | '{row[2][:40]}...' | {row[3]:.0f}ms | Refused:{row[4]}")

    print(f"\nJSONL log written to: {JSONL_PATH}")
    print("✅ Logger test complete")