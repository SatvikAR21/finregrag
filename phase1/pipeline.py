# =============================================================================
# What this file does:
# Provides a single ingest_document() function that runs the complete
# ingestion pipeline on any PDF file:
#   1. Extract text page by page (ingest)
#   2. Split into overlapping chunks (chunk)
#   3. Embed chunks and store in ChromaDB (embed_and_store)
#
# Used by the FastAPI /upload endpoint so the UI can trigger ingestion.
# =============================================================================

import os               # for file path operations
import json             # for saving intermediate JSON files
import sys              # for path manipulation

# --- PATH SETUP ---
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
import fitz                          # PyMuPDF — PDF text extraction
from langchain_text_splitters import RecursiveCharacterTextSplitter  # chunking
import chromadb                      # vector database
from sentence_transformers import SentenceTransformer                # embeddings
# --- CONFIGURATION — must match what Phase 1 used ---
CHROMA_PATH      = os.path.join(_PROJECT_ROOT, "data", "chromadb")
COLLECTION_NAME  = "finreg_documents"    # same collection as Phase 1
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"   # same model as Phase 1
CHUNK_SIZE       = 500                   # same chunk size as Phase 1
CHUNK_OVERLAP    = 50                    # same overlap as Phase 1
PROCESSED_DIR    = os.path.join(_PROJECT_ROOT, "data", "processed")

# Load embedding model once at module level — reused for every upload
_MODEL = None    # loaded lazily on first upload to save startup memory


def ingest_document(pdf_path: str) -> dict:
    """
    Runs the complete ingestion pipeline on a single PDF file.

    Args:
        pdf_path: absolute path to the PDF file on disk

    Returns:
        dict with keys:
            success (bool): True if ingestion completed without error
            filename (str): the PDF filename
            pages (int): number of pages extracted
            chunks (int): number of chunks created and stored
            message (str): human-readable status message
    """
    filename = os.path.basename(pdf_path)         # e.g. "basel3_framework.pdf"
    doc_id   = filename.replace(".pdf", "")       # e.g. "basel3_framework"

    print(f"\n📄 Starting ingestion: {filename}")

    # ── STAGE 1: EXTRACT TEXT FROM PDF ─────────────────────────
    print("  → Stage 1: Extracting text from PDF...")
    try:
        doc = fitz.open(pdf_path)                 # open the PDF with PyMuPDF
    except Exception as e:
        return {
            "success": False, "filename": filename,
            "pages": 0, "chunks": 0,
            "message": f"Failed to open PDF: {str(e)}"
        }

    pages_data = []                               # list to hold page records
    for page_num in range(len(doc)):              # loop through every page
        page = doc[page_num]                      # get the page object
        text = page.get_text()                    # extract raw text from page
        if text.strip():                          # skip blank pages
            pages_data.append({
                "source":   filename,             # which document
                "page":     page_num + 1,         # 1-based page number
                "text":     text,                 # raw extracted text
            })
    doc.close()                                   # close the PDF file

    if not pages_data:                            # nothing was extracted
        return {
            "success": False, "filename": filename,
            "pages": 0, "chunks": 0,
            "message": "No text could be extracted from this PDF. It may be scanned/image-based."
        }

    print(f"     Extracted {len(pages_data)} pages")

    # ── STAGE 2: CHUNK THE EXTRACTED TEXT ──────────────────────
    print("  → Stage 2: Splitting pages into chunks...")
    splitter = RecursiveCharacterTextSplitter(    # create the text splitter
        chunk_size=CHUNK_SIZE,                    # max chars per chunk
        chunk_overlap=CHUNK_OVERLAP,              # overlap between chunks
        separators=["\n\n", "\n", ".", " ", ""],  # split on these in order
    )

    all_chunks = []                               # list to hold all chunks
    for page_record in pages_data:               # loop through each page
        splits = splitter.split_text(page_record["text"])  # split page text
        for i, chunk_text in enumerate(splits):  # loop through splits
            all_chunks.append({
                "text":   chunk_text,            # the chunk text
                "source": page_record["source"], # source filename
                "page":   page_record["page"],   # page number
                "chunk_num": i,                  # chunk index within page
            })

    if not all_chunks:
        return {
            "success": False, "filename": filename,
            "pages": len(pages_data), "chunks": 0,
            "message": "Text was extracted but could not be chunked."
        }

    print(f"     Created {len(all_chunks)} chunks")

    # ── STAGE 3: EMBED AND STORE IN CHROMADB ───────────────────
    print("  → Stage 3: Embedding chunks and storing in ChromaDB...")
    client     = chromadb.PersistentClient(path=CHROMA_PATH)   # connect to ChromaDB
    collection = client.get_or_create_collection(COLLECTION_NAME)  # get or create

    # Build lists for ChromaDB batch insert
    ids        = []   # unique ID per chunk
    documents  = []   # chunk text
    metadatas  = []   # source and page metadata
    embeddings = []   # vector for each chunk

    # Get current count to avoid ID collisions with existing chunks
    existing_count = collection.count()          # how many vectors already stored

    texts_to_embed = [c["text"] for c in all_chunks]             # extract just text
    global _MODEL
    if _MODEL is None:
    _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    vectors = _MODEL.encode(texts_to_embed, show_progress_bar=False).tolist()    for i, (chunk, vector) in enumerate(zip(all_chunks, vectors)):  # pair chunks + vectors
        chunk_id = f"{doc_id}_p{chunk['page']}_c{chunk['chunk_num']}_{existing_count + i}"
        ids.append(chunk_id)
        documents.append(chunk["text"])
        metadatas.append({"source": chunk["source"], "page": str(chunk["page"])})
        embeddings.append(vector)

    # Insert in batches of 100 to avoid memory issues with large PDFs
    batch_size = 100
    for start in range(0, len(ids), batch_size):                 # loop in batches
        end = start + batch_size                                  # batch end index
        collection.add(                                           # add batch to ChromaDB
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )
        print(f"     Stored batch {start//batch_size + 1} ({min(end, len(ids))}/{len(ids)} chunks)")

    print(f"  ✅ Ingestion complete: {len(all_chunks)} chunks stored for {filename}\n")

    return {
        "success":  True,
        "filename": filename,
        "pages":    len(pages_data),
        "chunks":   len(all_chunks),
        "message":  f"Successfully ingested {len(pages_data)} pages and {len(all_chunks)} chunks into ChromaDB."
    }


def get_ingested_documents() -> list:
    """
    Returns a list of all unique documents currently stored in ChromaDB.
    Used by the /documents endpoint to populate the document list in the UI.
    """
    try:
        client     = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_or_create_collection(COLLECTION_NAME)
        total      = collection.count()          # total vectors stored

        if total == 0:                           # nothing stored yet
            return []

        # Fetch all metadata to find unique source documents
        # We fetch in batches to handle large collections
        all_sources = {}                         # dict: filename → chunk count
        batch       = 100
        offset      = 0

        while offset < total:                    # loop until all metadata fetched
            results = collection.get(            # fetch a batch of metadata
                limit=batch,
                offset=offset,
                include=["metadatas"]            # only need metadata, not embeddings
            )
            for meta in results["metadatas"]:   # loop through metadata records
                src = meta.get("source", "unknown")          # get source filename
                all_sources[src] = all_sources.get(src, 0) + 1  # count chunks
            offset += batch                      # move to next batch

        # Build a clean list of document records
        documents = []
        for source, chunk_count in all_sources.items():
            documents.append({
                "filename":    source,
                "chunks":      chunk_count,
                "status":      "ready",
            })

        return documents                         # return list of document dicts

    except Exception as e:
        print(f"Error fetching documents: {e}")
        return []