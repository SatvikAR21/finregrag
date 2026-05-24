# =============================================================================
# What this file does:
# Implements Hybrid Retrieval — combining BM25 keyword search with ChromaDB
# vector search using Reciprocal Rank Fusion (RRF) to merge the two ranked
# lists into one superior result list. This version handles ChromaDB IDs in
# the format 'basel3_framework_p64_c0' used by our Phase 1 embed_and_store.py
# =============================================================================

import json                          # for loading our chunks JSON file
import os                            # for building file paths
from rank_bm25 import BM25Okapi      # the BM25 keyword search library
import chromadb                      # our vector database
from sentence_transformers import SentenceTransformer  # to embed the query

# --- CONFIGURATION ---
CHUNKS_PATH = os.path.join("data", "processed", "basel3_framework_chunks.json")  # chunk file
CHROMA_PATH = os.path.join("data", "chromadb")   # persisted vector database folder
COLLECTION_NAME = "finreg_documents"             # confirmed collection name
EMBEDDING_MODEL = "all-MiniLM-L6-v2"            # same model used to build the index
TOP_K = 10                                        # results to retrieve from each method
RRF_K = 60    
# Model loaded lazily on first query — saves memory on startup
_EMBEDDING_MODEL = None


def load_chunks():
    """
    Loads all text chunks from the JSON file we created in Phase 1.
    Returns a list of chunk dictionaries, each with 'text' and metadata.
    """
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:   # open the chunks file
        chunks = json.load(f)                               # parse JSON into a Python list
    return chunks                                           # return the list of chunks


def build_bm25_index(chunks):
    """
    Builds a BM25 index from our chunk texts.
    Tokenizes each chunk into words for keyword matching.
    """
    tokenized_corpus = [
        chunk["text"].lower().split()   # lowercase and split each chunk into words
        for chunk in chunks             # do this for every chunk
    ]
    bm25 = BM25Okapi(tokenized_corpus)  # build the BM25 index
    return bm25                          # return the ready-to-query index


def bm25_search(bm25, query, chunks, top_k=TOP_K):
    """
    Searches the BM25 index for chunks matching the query keywords.
    Returns a list of (chunk_index, score) tuples sorted by score descending.
    """
    tokenized_query = query.lower().split()                 # tokenize the query
    scores = bm25.get_scores(tokenized_query)               # score every chunk
    top_indices = sorted(                                    # sort by score
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]                                                # take top K
    return [(i, scores[i]) for i in top_indices]            # return (index, score) pairs


def vector_search(query, top_k=TOP_K):
    """
    Searches ChromaDB using vector similarity.
    Returns a list of dicts with 'id', 'text', 'metadata', and 'distance'.
    We return full documents here so RRF can work with text directly,
    without needing to map IDs back to our JSON chunks.
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
    _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = _EMBEDDING_MODEL.encode([query]).tolist()    client = chromadb.PersistentClient(path=CHROMA_PATH)    # connect to ChromaDB
    collection = client.get_collection(COLLECTION_NAME)     # get our collection
    results = collection.query(                              # run similarity search
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]     # get text + metadata + scores
    )

    # Build a clean list of result dicts — one per retrieved chunk
    vector_hits = []                                         # empty list to fill
    for i in range(len(results["ids"][0])):                 # loop through each result
        vector_hits.append({                                 # build a dict for each hit
            "id": results["ids"][0][i],                     # the string ID like 'basel3_p64_c0'
            "text": results["documents"][0][i],             # the actual chunk text
            "metadata": results["metadatas"][0][i],         # page number, source, etc.
            "distance": results["distances"][0][i]          # similarity distance score
        })
    return vector_hits                                       # return list of result dicts


def reciprocal_rank_fusion(bm25_results, vector_hits, chunks, top_k=5, k=RRF_K):
    """
    Combines BM25 and vector search results using Reciprocal Rank Fusion.
    
    BM25 results reference chunks by integer index into our JSON list.
    Vector results are full dicts with text already attached.
    
    We use chunk TEXT as the common key to detect overlapping results,
    since the two systems use different ID schemes.
    """
    rrf_scores = {}    # key: chunk text (first 100 chars), value: accumulated RRF score
    chunk_store = {}   # key: chunk text (first 100 chars), value: full chunk dict

    # --- Score BM25 results ---
    for rank, (chunk_idx, _) in enumerate(bm25_results):    # loop with rank position
        chunk = chunks[chunk_idx]                            # get the chunk dict from JSON
        key = chunk["text"][:100]                           # use first 100 chars as unique key
        if key not in rrf_scores:                           # initialize if new
            rrf_scores[key] = 0.0
            chunk_store[key] = chunk                        # store the full chunk
        rrf_scores[key] += 1.0 / (k + rank + 1)            # add RRF score contribution

    # --- Score vector results ---
    for rank, hit in enumerate(vector_hits):                # loop with rank position
        key = hit["text"][:100]                             # same key scheme — first 100 chars
        if key not in rrf_scores:                           # initialize if new
            rrf_scores[key] = 0.0
            chunk_store[key] = {                            # build chunk dict from vector hit
                "text": hit["text"],
                "metadata": hit.get("metadata", {}),
                "source": hit.get("metadata", {}).get("source", "unknown"),
                "page": hit.get("metadata", {}).get("page", "?")
            }
        rrf_scores[key] += 1.0 / (k + rank + 1)            # add RRF score contribution

    # --- Sort by combined RRF score ---
    sorted_keys = sorted(                                    # sort all keys by score
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True                                         # highest first
    )

    # --- Build final result list ---
    top_results = []                                         # final output list
    for key, score in sorted_keys[:top_k]:                  # take top K results
        chunk = chunk_store[key].copy()                     # copy the chunk dict
        chunk["rrf_score"] = round(score, 6)               # attach the RRF score
        chunk["retrieval_method"] = "hybrid_rrf"           # tag retrieval method
        top_results.append(chunk)                           # add to results

    return top_results                                       # return final ranked list


def hybrid_search(query, top_k=5):
    """
    Main function — runs BM25 + vector search, fuses with RRF, returns top chunks.
    Call this from query_v2.py.
    """
    print(f"\n🔍 Running hybrid search for: '{query}'")

    chunks = load_chunks()                                   # load all chunks from JSON
    bm25 = build_bm25_index(chunks)                         # build BM25 index in memory

    print("  → BM25 keyword search running...")
    bm25_results = bm25_search(bm25, query, chunks)         # keyword search

    print("  → Vector similarity search running...")
    vector_hits = vector_search(query)                      # semantic search

    print("  → Fusing results with RRF...")
    top_results = reciprocal_rank_fusion(                   # merge both result lists
        bm25_results, vector_hits, chunks, top_k=top_k
    )

    print(f"  ✅ Retrieved {len(top_results)} chunks via hybrid RRF\n")
    return top_results                                       # return final results


# --- QUICK TEST ---
if __name__ == "__main__":
    test_query = "What is the minimum CET1 capital ratio requirement under Basel III?"
    results = hybrid_search(test_query, top_k=5)

    print("=" * 60)
    print("TOP 5 HYBRID RETRIEVAL RESULTS")
    print("=" * 60)

    for i, chunk in enumerate(results):                      # loop through results
        print(f"\nRank {i+1} | RRF Score: {chunk['rrf_score']}")
        print(f"Method: {chunk['retrieval_method']}")
        print(f"Text preview: {chunk['text'][:200]}...")     # first 200 chars