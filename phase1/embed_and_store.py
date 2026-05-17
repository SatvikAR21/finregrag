# ==============================================================================
# What this file does:
# This is the Embedding and Storage module for FinReg-RAG.
# It reads the chunks JSON file produced by chunk.py, converts every chunk's
# text into a numerical vector (embedding) using sentence-transformers,
# and stores those vectors — along with the original text and metadata —
# into a ChromaDB vector database saved locally on disk.
# The ChromaDB database is what our retrieval system will search in Step 4.
# ==============================================================================

import json                          # Built-in — for reading our chunks JSON file
import os                            # Built-in — for building file paths
from sentence_transformers import SentenceTransformer  # Converts text to vectors
import chromadb                      # Our local vector database
from chromadb.config import Settings # ChromaDB configuration options


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# The embedding model we use — runs locally, no API key needed
# all-MiniLM-L6-v2 is fast, small (90MB), and produces 384-dimensional vectors
# It's a great balance of speed and quality for our use case
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Where ChromaDB will save its database files on disk
# This folder will be created automatically if it doesn't exist
CHROMADB_PATH = os.path.join("data", "chromadb")

# The name of our collection inside ChromaDB
# A collection is like a table in a regular database — it groups related vectors
COLLECTION_NAME = "finreg_documents"

# How many chunks to embed and store at once (batching)
# Embedding one chunk at a time is slow — we do 32 at once for speed
BATCH_SIZE = 32


def load_chunks(chunks_path: str) -> list:
    """
    Loads the chunks JSON file produced by chunk.py.
    Returns a list of chunk dictionaries.

    chunks_path: full path to the chunks JSON file
    """

    # Verify the file exists before trying to open it
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Chunks file not found at: {chunks_path}")

    # Open and parse the JSON file into a Python list
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    # Report how many chunks were loaded
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    # Return the list of chunk dictionaries
    return chunks


def initialize_embedding_model() -> SentenceTransformer:
    """
    Loads the sentence-transformer embedding model into memory.
    The first time this runs, it downloads the model (~90MB).
    After that, it loads from a local cache instantly.

    Returns the loaded model object.
    """

    # Tell the user what's happening — the first download can take a minute
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    print("(First run downloads ~90MB — this is normal, please wait...)")

    # Load the model — SentenceTransformer handles download + caching automatically
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Confirm the model loaded successfully
    print(f"Embedding model loaded successfully!")
    print(f"Vector dimensions: {model.get_sentence_embedding_dimension()}")

    # Return the loaded model so we can use it to embed text
    return model


def initialize_chromadb() -> tuple:
    """
    Sets up ChromaDB — creates the database directory and collection.
    Returns a tuple of (client, collection) ready to receive vectors.
    """

    # Make sure the ChromaDB directory exists on disk
    os.makedirs(CHROMADB_PATH, exist_ok=True)

    # Create a ChromaDB client that saves data to disk (persistent storage)
    # Without persistence, data would disappear every time you restart Python
    client = chromadb.PersistentClient(path=CHROMADB_PATH)

    # Get or create our collection inside ChromaDB
    # get_or_create_collection is safe — if the collection already exists,
    # it opens it; if not, it creates it fresh
    # metadata={"hnsw:space": "cosine"} tells ChromaDB to use cosine similarity
    # for measuring how close two vectors are — best for text embeddings
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity for search
    )

    # Report the current state of the collection
    print(f"ChromaDB initialized at: {CHROMADB_PATH}")
    print(f"Collection '{COLLECTION_NAME}' has {collection.count()} existing vectors")

    # Return both the client and collection for use in the next function
    return client, collection


def embed_and_store_chunks(chunks: list, model: SentenceTransformer, collection) -> None:
    """
    The core function — takes all chunks, embeds them in batches,
    and stores them in ChromaDB.

    chunks: list of chunk dictionaries from load_chunks()
    model: the loaded SentenceTransformer model
    collection: the ChromaDB collection to store vectors in
    """

    # Get all the chunk IDs already in ChromaDB
    # This lets us skip chunks we've already embedded (avoid duplicates)
    existing_ids = set()

    # Only check for existing IDs if the collection isn't empty
    if collection.count() > 0:
        # Get all existing IDs from the collection
        existing = collection.get(include=[])  # include=[] means only return IDs
        existing_ids = set(existing["ids"])    # Convert to a set for fast lookup
        print(f"Found {len(existing_ids)} existing chunks — will skip duplicates")

    # Filter out chunks that are already in ChromaDB
    # This makes our script safe to run multiple times without creating duplicates
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

    # If all chunks are already stored, nothing to do
    if not new_chunks:
        print("All chunks already in ChromaDB — nothing new to add!")
        return

    # Tell the user how many new chunks we're processing
    print(f"Embedding and storing {len(new_chunks)} new chunks...")
    print(f"Processing in batches of {BATCH_SIZE}...")

    # Track progress
    total_stored = 0

    # Process chunks in batches for speed
    # range(start, stop, step) generates: 0, 32, 64, 96, ...
    for batch_start in range(0, len(new_chunks), BATCH_SIZE):

        # Slice the next batch of chunks from our list
        # batch_start:batch_start+BATCH_SIZE gets the next 32 chunks
        batch = new_chunks[batch_start : batch_start + BATCH_SIZE]

        # Extract just the text from each chunk in this batch
        # This is what we feed to the embedding model
        batch_texts = [chunk["text"] for chunk in batch]

        # Extract the unique IDs for this batch
        # ChromaDB uses these as primary keys to identify each vector
        batch_ids = [chunk["chunk_id"] for chunk in batch]

        # Build metadata dictionaries for this batch
        # ChromaDB stores metadata alongside each vector for retrieval later
        # IMPORTANT: ChromaDB only accepts str, int, float, or bool in metadata
        # We convert page_number to int to be safe
        batch_metadatas = [
            {
                "source":      chunk["source"],           # PDF filename
                "page_number": int(chunk["page_number"]), # Page number as integer
                "chunk_index": int(chunk["chunk_index"]), # Position on page
                "char_count":  int(chunk["char_count"]),  # Length of chunk
                "chunked_at":  chunk["chunked_at"]        # Timestamp string
            }
            for chunk in batch
        ]

        # THE KEY STEP: Convert all texts in this batch to vectors
        # model.encode() returns a numpy array of shape (batch_size, 384)
        # Each row is a 384-dimensional vector for one chunk
        # show_progress_bar=False keeps output clean
        batch_embeddings = model.encode(
            batch_texts,
            show_progress_bar=False,
            convert_to_numpy=True  # Return as numpy array for ChromaDB compatibility
        )

        # Convert numpy array to a plain Python list of lists
        # ChromaDB expects Python lists, not numpy arrays
        batch_embeddings_list = batch_embeddings.tolist()

        # Store everything in ChromaDB in one operation
        # collection.add() takes parallel lists: IDs, embeddings, texts, metadata
        collection.add(
            ids=batch_ids,                    # Unique IDs for each chunk
            embeddings=batch_embeddings_list, # The vectors we just computed
            documents=batch_texts,            # Original text (stored for retrieval)
            metadatas=batch_metadatas         # Source, page number, etc.
        )

        # Update and display progress
        total_stored += len(batch)
        print(f"Stored {total_stored}/{len(new_chunks)} chunks...")

    # Final confirmation
    print(f"\nAll chunks embedded and stored successfully!")
    print(f"ChromaDB collection now contains {collection.count()} total vectors")


def verify_storage(collection) -> None:
    """
    Runs a quick test query to verify our stored vectors work correctly.
    Searches for a simple financial term and prints the top result.

    collection: the ChromaDB collection to test
    """

    print("\n" + "=" * 60)
    print("VERIFICATION — Testing a sample search query")
    print("=" * 60)

    # Initialize a fresh model instance for the test query
    # (In Step 4, this will be done properly — this is just a quick sanity check)
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Convert our test question to a vector
    test_query = "What is the minimum CET1 capital ratio?"
    query_vector = model.encode(test_query).tolist()

    # Search ChromaDB for the 3 closest chunks
    # n_results=3 means return the top 3 matches
    results = collection.query(
        query_embeddings=[query_vector],  # Our query vector (wrapped in a list)
        n_results=3,                      # How many results to return
        include=["documents", "metadatas", "distances"]  # What to include in response
    )

    # Print the results
    print(f"\nQuery: '{test_query}'")
    print(f"\nTop 3 matching chunks:\n")

    # Loop through the results and print each one
    # results["documents"][0] is a list of the matched texts
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],   # List of matched texts
        results["metadatas"][0],   # List of matched metadata dicts
        results["distances"][0]    # List of similarity distances (lower = more similar)
    )):
        print(f"Result {i+1}:")
        print(f"  Source:   {meta['source']} — Page {meta['page_number']}")
        print(f"  Distance: {dist:.4f} (lower = more similar)")
        print(f"  Preview:  {doc[:200]}...")
        print()


def embed_and_store_document(chunks_filename: str) -> None:
    """
    Master function that orchestrates the full embed-and-store pipeline.

    chunks_filename: just the filename, e.g. "basel3_framework_chunks.json"
    """

    # Build the full path to the chunks file
    chunks_path = os.path.join("data", "processed", chunks_filename)

    # Step 1: Load all chunks from JSON
    chunks = load_chunks(chunks_path)

    # Step 2: Load the embedding model
    model = initialize_embedding_model()

    # Step 3: Set up ChromaDB
    client, collection = initialize_chromadb()

    # Step 4: Embed all chunks and store in ChromaDB
    embed_and_store_chunks(chunks, model, collection)

    # Step 5: Run a verification search to confirm everything works
    verify_storage(collection)


# This block runs only when you execute this file directly
if __name__ == "__main__":

    # Print header
    print("=" * 60)
    print("FinReg-RAG — Embedding and Storage")
    print("=" * 60)

    # Run the full pipeline on our Basel III chunks
    embed_and_store_document("basel3_framework_chunks.json")

    # Final success message
    print("\n✅ Embedding and storage complete!")
    print(f"Your ChromaDB vector database is saved at: {CHROMADB_PATH}")
    print("You are now ready for Step 4: Retrieval and Generation")