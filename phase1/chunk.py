# ==============================================================================
# What this file does:
# This is the Chunking module for FinReg-RAG.
# It reads the JSON file produced by ingest.py (which contains page-by-page text),
# splits each page's text into smaller overlapping chunks using LangChain's
# RecursiveCharacterTextSplitter, attaches rich metadata to every chunk,
# and saves the final list of chunks as a new JSON file in data/processed/.
# The output of this file is the input for Step 3: Embedding.
# ==============================================================================

import json           # Built-in — for reading and writing JSON files
import os             # Built-in — for building file paths
from datetime        import datetime   # Built-in — for timestamps
# ADD THIS LINE INSTEAD:
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==============================================================================
# CONFIGURATION — Change these values to experiment with chunk sizes
# ==============================================================================

CHUNK_SIZE    = 500   # Maximum number of characters in each chunk
CHUNK_OVERLAP = 100   # Number of characters shared between adjacent chunks


def load_ingested_json(json_path: str) -> list:
    """
    Loads the JSON file produced by ingest.py.
    Returns a list of page dictionaries.
    
    json_path: full path to the JSON file (e.g., data/processed/basel3_framework.json)
    """
    
    # Check the file exists before trying to open it
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Ingested JSON not found at: {json_path}")
    
    # Open and read the JSON file
    # encoding='utf-8' handles special financial symbols like § or €
    with open(json_path, 'r', encoding='utf-8') as f:
        pages = json.load(f)  # json.load converts JSON text into a Python list
    
    # Tell the user how many pages were loaded
    print(f"Loaded {len(pages)} pages from {json_path}")
    
    # Return the list of page dictionaries
    return pages


def chunk_pages(pages: list) -> list:
    """
    Takes a list of page dictionaries and splits each page's text into chunks.
    Returns a flat list of chunk dictionaries, each with full metadata.
    
    pages: list of page dictionaries from load_ingested_json
    """
    
    # Initialize LangChain's RecursiveCharacterTextSplitter
    # This splitter tries to cut at: paragraphs → sentences → words → characters
    # in that order of preference, keeping chunks as meaningful as possible
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,        # Maximum characters per chunk
        chunk_overlap=CHUNK_OVERLAP,  # Characters shared between adjacent chunks
        length_function=len,          # Use Python's built-in len() to measure size
        separators=[                  # Try splitting at these boundaries in order:
            "\n\n",                   # 1st choice: paragraph breaks (double newline)
            "\n",                     # 2nd choice: line breaks
            ". ",                     # 3rd choice: end of sentences
            " ",                      # 4th choice: spaces between words
            ""                        # Last resort: split anywhere
        ]
    )
    
    # Create an empty list to collect all chunks from all pages
    all_chunks = []
    
    # Track total chunk count across all pages for unique ID generation
    global_chunk_index = 0
    
    # Loop through every page in our document
    for page in pages:
        
        # Get the text from this page
        # .get() safely retrieves the value — returns "" if key doesn't exist
        page_text = page.get("text", "")
        
        # Skip pages with no text (safety check)
        if not page_text.strip():
            continue
        
        # Use the splitter to cut this page's text into a list of strings
        # split_text() returns a plain list of text strings
        text_chunks = splitter.split_text(page_text)
        
        # Loop through each chunk of text produced for this page
        for chunk_index, chunk_text in enumerate(text_chunks):
            
            # Skip chunks that are too short to be useful (less than 50 characters)
            # These are usually stray headers or page numbers with no real content
            if len(chunk_text.strip()) < 50:
                continue
            
            # Build a unique ID for this chunk
            # Format: sourcename_pPAGENUM_cCHUNKINDEX
            # Example: "basel3_framework_p12_c3"
            source_stem = page.get("source", "unknown").replace(".pdf", "")
            chunk_id = f"{source_stem}_p{page.get('page_number', 0)}_c{chunk_index}"
            
            # Build the complete chunk dictionary with text + all metadata
            chunk_data = {
                "chunk_id":             chunk_id,                    # Unique identifier
                "source":               page.get("source", ""),      # Original PDF filename
                "page_number":          page.get("page_number", 0),  # Page this came from
                "chunk_index":          chunk_index,                  # Position on this page
                "total_chunks_on_page": len(text_chunks),            # How many chunks this page produced
                "text":                 chunk_text.strip(),           # The actual chunk text
                "char_count":           len(chunk_text.strip()),      # Character count of this chunk
                "ingested_at":          page.get("ingested_at", ""), # When the PDF was ingested
                "chunked_at":           datetime.now().isoformat()   # When this chunk was created
            }
            
            # Add this chunk to our master list
            all_chunks.append(chunk_data)
            
            # Increment our global counter
            global_chunk_index += 1
    
    # Print a summary of what we produced
    print(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
    
    # Return the complete list of all chunks
    return all_chunks


def save_chunks_to_json(chunks: list, output_path: str) -> None:
    """
    Saves the list of chunk dictionaries to a JSON file.
    
    chunks: list of chunk dictionaries from chunk_pages()
    output_path: where to save the output JSON file
    """
    
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Open the output file and write the chunks as formatted JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    # Confirm success
    print(f"Saved {len(chunks)} chunks to: {output_path}")


def chunk_document(json_filename: str) -> str:
    """
    Master function that orchestrates the full chunking of one document.
    Loads the ingested JSON, chunks it, and saves the result.
    Returns the path to the saved chunks JSON file.
    
    json_filename: just the filename, e.g. "basel3_framework.json"
    """
    
    # Build the full path to the ingested JSON in data/processed/
    input_path = os.path.join("data", "processed", json_filename)
    
    # Build the output filename by adding "_chunks" before the extension
    # e.g., "basel3_framework.json" → "basel3_framework_chunks.json"
    output_filename = json_filename.replace(".json", "_chunks.json")
    
    # Build the full output path
    output_path = os.path.join("data", "processed", output_filename)
    
    # Step 1: Load the ingested pages from JSON
    pages = load_ingested_json(input_path)
    
    # Step 2: Split all pages into chunks
    chunks = chunk_pages(pages)
    
    # Step 3: Save the chunks to a new JSON file
    save_chunks_to_json(chunks, output_path)
    
    # Return the output path
    return output_path


def print_sample_chunks(chunks_path: str, num_samples: int = 3) -> None:
    """
    Loads a chunks JSON file and prints a few sample chunks so you can
    visually verify the chunking worked correctly.
    
    chunks_path: path to the chunks JSON file
    num_samples: how many sample chunks to print (default 3)
    """
    
    # Load the chunks file
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Print a header
    print("\n" + "=" * 60)
    print(f"SAMPLE CHUNKS (showing {num_samples} of {len(chunks)} total)")
    print("=" * 60)
    
    # Print the first num_samples chunks
    for i, chunk in enumerate(chunks[:num_samples]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"ID:          {chunk['chunk_id']}")
        print(f"Source:      {chunk['source']}")
        print(f"Page:        {chunk['page_number']}")
        print(f"Characters:  {chunk['char_count']}")
        print(f"Text preview: {chunk['text'][:200]}...")  # Show first 200 chars
        print("-" * 40)


# This block runs only when you execute this file directly
if __name__ == "__main__":
    
    # Print a header
    print("=" * 60)
    print("FinReg-RAG — Document Chunking")
    print("=" * 60)
    
    # Run the chunking pipeline on our Basel III document
    output_path = chunk_document("basel3_framework.json")
    
    # Print sample chunks so we can visually verify the output
    print_sample_chunks(output_path, num_samples=3)
    
    # Final success message
    print(f"\n✅ Chunking complete! Output saved to: {output_path}")
    print("You can now open this file to inspect your chunks.")