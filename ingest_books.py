import os
import shutil
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from supabase import create_client, Client
from pypdf import PdfReader

# Load environment variables
load_dotenv()

# 1. Initialize Supabase Client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL or SUPABASE_KEY is missing from .env file.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 2. Initialize Gemini API Client
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is missing from .env file.")

ai_client = genai.Client(api_key=api_key)

# 3. Auto-Detect Supported Embedding Model Name
def resolve_embedding_model() -> str:
    try:
        available_models = [m.name for m in ai_client.models.list()]
        for target in ["text-embedding-004", "gemini-embedding-001"]:
            for m in available_models:
                if target in m:
                    clean_name = m.replace("models/", "")
                    return clean_name
    except Exception:
        pass
    return "text-embedding-004"

EMBED_MODEL = resolve_embedding_model()

# 4. Setup Directories
BOOKS_DIR = Path("books")
COMPLETED_DIR = Path("completed")
COMPLETED_DIR.mkdir(exist_ok=True)


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Generates vector embeddings for a batch of text strings simultaneously."""
    try:
        response = ai_client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts,
            config={"output_dimensionality": 768}
        )
        return [e.values for e in response.embeddings]
    except Exception:
        response = ai_client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts
        )
        return [e.values[:768] for e in response.embeddings]


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif suffix == ".pdf":
        reader = PdfReader(file_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)
    return ""


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += (chunk_size - overlap)

    return chunks


def process_and_ingest():
    files = list(BOOKS_DIR.glob("*.txt")) + list(BOOKS_DIR.glob("*.pdf"))

    if not files:
        print(f"No .txt or .pdf files found in directory: '{BOOKS_DIR.resolve()}'")
        return

    print(f"Found {len(files)} file(s) to process.\n")

    for idx, file_path in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] Processing: {file_path.name}")

        try:
            raw_text = extract_text(file_path)
            if not raw_text.strip():
                print(f"   Skipping empty/unreadable file: {file_path.name}")
                continue

            chunks = chunk_text(raw_text)
            print(f"   -> Generated {len(chunks)} text chunks.")

            successful = 0
            batch_size = 20 # Process 20 chunks at a time for faster ingestion
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                try:
                    # 1. Get embeddings for the entire batch at once
                    vectors = get_embeddings_batch(batch)
                    
                    # 2. Prepare the database rows
                    rows = []
                    for b_idx, text in enumerate(batch):
                        rows.append({
                            "book_title": file_path.stem,
                            "content": text,
                            "embedding": vectors[b_idx]
                        })
                    
                    # 3. Upsert the batch into Supabase
                    supabase.table("hermeneutics_books").upsert(
                        rows,
                        on_conflict="content_hash",
                        ignore_duplicates=True
                    ).execute()
                    
                    successful += len(batch)
                    
                    # 4. Print dynamic real-time progress on the same line
                    print(f"   -> Ingesting... [{successful}/{len(chunks)} chunks]", end="\r", flush=True)
                    
                    time.sleep(0.5) # Small rate-limit breather

                except Exception as batch_err:
                    print(f"\n   [!] Batch error around chunk {i}: {batch_err}")
                    time.sleep(2)

            dest = COMPLETED_DIR / file_path.name
            shutil.move(str(file_path), str(dest))
            print(f"\n   [✓] Successfully ingested ({successful}/{len(chunks)} chunks) -> Moved to completed/\n")

        except Exception as file_err:
            print(f"\n   [✗] File processing error on {file_path.name}: {file_err}\n")


if __name__ == "__main__":
    process_and_ingest()