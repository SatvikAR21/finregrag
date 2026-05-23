# =============================================================================
# What this file does:
# Defines Pydantic data models for our FastAPI endpoints.
# Pydantic models are Python classes that describe the exact shape of
# request and response data — like a contract between frontend and backend.
# FastAPI uses these to automatically validate incoming data and serialize
# outgoing data to JSON.
# =============================================================================

from pydantic import BaseModel    # base class for all our data models
from typing import List, Optional # for type hints on lists and optional fields


class QueryRequest(BaseModel):
    """
    Shape of the JSON body sent FROM React TO FastAPI when user asks a question.
    Example: {"question": "What is the CET1 ratio?"}
    """
    question: str                  # the user's regulatory question — required string


class ChunkSource(BaseModel):
    """
    Represents one cited source chunk in the answer.
    Shown in the Sources Panel on the right side of the UI.
    """
    citation_number: int           # the [1], [2], [3] number in the answer
    source: str                    # document filename e.g. "basel3_framework.pdf"
    page: str                      # page number e.g. "28" or "unknown"
    text_preview: str              # first 200 chars of the chunk text
    cohere_score: Optional[float] = None  # relevance score from Cohere (0 to 1)


class LatencyBreakdown(BaseModel):
    """
    Per-stage timing data shown in the latency bar component.
    All values are in milliseconds.
    """
    retrieval_ms: float            # time for hybrid BM25 + vector search
    reranking_ms: float            # time for Cohere reranking
    generation_ms: float           # time for LLM answer generation
    total_ms: float                # sum of all three stages


class QueryResponse(BaseModel):
    """
    Shape of the JSON body sent FROM FastAPI BACK TO React after processing.
    React uses these fields to render the answer, sources, and latency.
    """
    answer: str                    # the full cited answer text
    refused: bool                  # True if system refused to answer
    sources: List[ChunkSource]     # list of cited source chunks
    latency: LatencyBreakdown      # timing breakdown per stage
    query: str                     # echo of the original question


class AnalyticsResponse(BaseModel):
    """
    Shape of the analytics data sent to the dashboard tab.
    Mirrors what our analytics.py reads from SQLite.
    """
    total_queries: int             # total queries logged all time
    answered: int                  # queries that got an answer
    refused: int                   # queries that were refused
    refusal_rate: float            # refused / total as percentage
    avg_total_ms: float            # average end-to-end latency
    avg_retrieval_ms: float        # average retrieval stage latency
    avg_reranking_ms: float        # average reranking stage latency
    avg_generation_ms: float       # average generation stage latency
    avg_cohere_score: float        # average top chunk relevance score
    recent_queries: List[dict]     # last 10 queries for history table