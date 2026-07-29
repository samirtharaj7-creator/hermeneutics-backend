import os
from dotenv import load_dotenv
from google import genai
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
ai_client = genai.Client(api_key=api_key)


def get_next_study_step(passage: str, current_stage: str):
    query = f"Methodology and procedural steps for {current_stage} when studying {passage}"
    print(f"📖 Passage: {passage}")
    print(f"🎯 Current Stage: {current_stage}\n")

    response = ai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config={"output_dimensionality": 768}
    )
    query_vector = response.embeddings[0].values

    rpc_response = supabase.rpc(
        "match_study_step",
        {
            "query_embedding": query_vector,
            "match_threshold": 0.15,
            "match_count": 4,
            "priority_boost": 2.0  # Increased boost for core guides
        }
    ).execute()

    results = rpc_response.data

    if not results:
        print("❌ No matching steps found.")
        return

    print("--- GUIDANCE TO PRESENT TO USER ---")
    for idx, row in enumerate(results, start=1):
        title = row["book_title"].lower()
        # Flexible check for general-hermeneutics and special-hermeneutics core PDFs
        is_core = ("general" in title or "special" in title) and "hermeneutics" in title
        
        tag = "⭐ [PROCEDURAL GUIDE]" if is_core else "📖 [SUPPORTING TEXT]"
        print(f"\n[{idx}] {tag} From: {row['book_title']}")
        print(f"Content:\n{row['content'][:350]}...\n")


if __name__ == "__main__":
    get_next_study_step(
        passage="Romans 8:1-4", 
        current_stage="Historical and Cultural Context"
    )