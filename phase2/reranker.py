# =============================================================================
# What this file does:
# Takes the top chunks from hybrid_retriever.py and re-scores them using
# Cohere's Rerank API — a cross-encoder model that reads the query and each
# chunk together to assign a true relevance score. The output is a re-ordered
# list where the most genuinely relevant chunk is always first.
# =============================================================================

import os                            # for reading environment variables
import cohere                        # the Cohere Python SDK
from dotenv import load_dotenv       # for loading our .env file

load_dotenv()                        # load COHERE_API_KEY from .env into environment

# --- CONFIGURATION ---
COHERE_API_KEY = os.getenv("COHERE_API_KEY")   # read the key from environment
RERANK_MODEL = "rerank-english-v3.0"           # Cohere's best English reranking model
TOP_N = 3                                       # how many chunks to keep after reranking


def rerank_chunks(query, chunks, top_n=TOP_N):
    """
    Re-ranks a list of chunk dicts using Cohere Rerank API.
    
    Args:
        query: the user's question string
        chunks: list of chunk dicts from hybrid_retriever.hybrid_search()
        top_n: how many top chunks to return after reranking
    
    Returns:
        list of chunk dicts, reordered by Cohere relevance score, top_n only
    """
    if not COHERE_API_KEY:                                    # check key exists
        raise ValueError("COHERE_API_KEY not found in .env file")

    if not chunks:                                            # handle empty input
        print("  ⚠️  No chunks to rerank — returning empty list")
        return []

    print(f"  → Sending {len(chunks)} chunks to Cohere Rerank...")  # progress

    co = cohere.ClientV2(COHERE_API_KEY)                     # create Cohere client

    # Extract just the text from each chunk — Cohere only needs the text strings
    documents = [chunk["text"] for chunk in chunks]          # list of text strings

    # Call the Cohere Rerank API
    response = co.rerank(                                    # send rerank request
        model=RERANK_MODEL,                                  # which model to use
        query=query,                                         # the user's question
        documents=documents,                                 # our chunk texts
        top_n=top_n,                                         # how many to return
    )

    # Build reranked result list using the indices Cohere returns
    reranked_chunks = []                                     # empty output list
    for result in response.results:                          # loop through Cohere's results
        original_chunk = chunks[result.index].copy()        # get original chunk by index
        original_chunk["cohere_score"] = round(             # attach Cohere's relevance score
            result.relevance_score, 6
        )
        original_chunk["retrieval_method"] = "hybrid_rrf+cohere_rerank"  # update method tag
        reranked_chunks.append(original_chunk)              # add to output list

    print(f"  ✅ Reranking complete — kept top {len(reranked_chunks)} chunks\n")
    return reranked_chunks                                   # return reranked list


# --- QUICK TEST ---
if __name__ == "__main__":
    # Import hybrid search to get test chunks
    from hybrid_retriever import hybrid_search              # our Phase 2 Step 1 function

    test_query = "What is the minimum CET1 capital ratio requirement under Basel III?"

    print("Step 1: Running hybrid retrieval...")
    hybrid_chunks = hybrid_search(test_query, top_k=5)     # get 5 hybrid chunks

    print("Step 2: Re-ranking with Cohere...")
    reranked = rerank_chunks(test_query, hybrid_chunks, top_n=3)  # rerank to top 3

    print("=" * 60)
    print("TOP 3 RERANKED CHUNKS")
    print("=" * 60)

    for i, chunk in enumerate(reranked):                    # display results
        print(f"\nRank {i+1} | Cohere Score: {chunk['cohere_score']}")
        print(f"RRF Score was: {chunk.get('rrf_score', 'N/A')}")
        print(f"Text preview: {chunk['text'][:250]}...")