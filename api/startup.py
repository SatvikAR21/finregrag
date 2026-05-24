# =============================================================================
# What this file does:
# Runs automatically when the Render server starts.
# Checks if ChromaDB has any vectors — if not, downloads Basel III from
# BIS.org and runs the full ingestion pipeline to populate it.
# This ensures the system works on a fresh Render server with no local data.
# =============================================================================

import os        # for file path operations
import sys       # for path manipulation
import urllib.request  # for downloading the PDF

# --- PATH SETUP ---
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "phase1"))

# --- CONFIGURATION ---
BASEL3_URL   = "https://www.bis.org/publ/bcbs189.pdf"   # public BIS URL
RAW_DIR      = os.path.join(_PROJECT_ROOT, "data", "raw")
PDF_PATH     = os.path.join(RAW_DIR, "basel3_framework.pdf")
CHROMA_PATH  = os.path.join(_PROJECT_ROOT, "data", "chromadb")


def needs_ingestion():
    """
    Returns True if ChromaDB is empty or missing — meaning we need to rebuild.
    """
    try:
        import chromadb
        client     = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_or_create_collection("finreg_documents")
        count      = collection.count()
        print(f"  ChromaDB has {count} vectors.")
        return count == 0        # needs ingestion if empty
    except Exception as e:
        print(f"  ChromaDB check failed: {e}")
        return True              # assume needs ingestion on any error


def download_basel3():
    """
    Downloads the Basel III PDF from BIS.org if not already present.
    """
    os.makedirs(RAW_DIR, exist_ok=True)              # create data/raw/ if missing

    if os.path.exists(PDF_PATH):                     # already downloaded
        print(f"  Basel III PDF already exists at {PDF_PATH}")
        return True

    print(f"  Downloading Basel III from {BASEL3_URL}...")
    try:
        urllib.request.urlretrieve(BASEL3_URL, PDF_PATH)  # download to disk
        size_mb = os.path.getsize(PDF_PATH) / 1024 / 1024
        print(f"  Downloaded successfully ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def run_startup():
    """
    Main startup function — called once when the FastAPI server starts.
    Checks if data exists, downloads if needed, ingests if needed.
    """
    print("\n" + "="*50)
    print("  FINREG-RAG STARTUP CHECK")
    print("="*50)

    if not needs_ingestion():                        # ChromaDB already has data
        print("  ✅ ChromaDB is populated. Skipping ingestion.")
        print("="*50 + "\n")
        return

    print("  ⚠️  ChromaDB is empty. Running startup ingestion...")

    # Step 1: Download the PDF
    if not download_basel3():
        print("  ❌ Could not download Basel III PDF. Startup ingestion skipped.")
        print("="*50 + "\n")
        return

    # Step 2: Run ingestion pipeline
    print("  Running ingestion pipeline...")
    try:
        from pipeline import ingest_document         # import our pipeline
        result = ingest_document(PDF_PATH)           # run full pipeline
        if result["success"]:
            print(f"  ✅ Ingestion complete: {result['chunks']} chunks stored.")
        else:
            print(f"  ❌ Ingestion failed: {result['message']}")
    except Exception as e:
        print(f"  ❌ Ingestion error: {e}")

    print("="*50 + "\n")


if __name__ == "__main__":
    run_startup()                                    # allow running directly for testing