import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
import google.generativeai as genai

# Initialize FastAPI App
app = FastAPI(
    title="Hermeneutics Bible Study Assistant API",
    version="1.0.0",
    description="API for multi-step hermeneutical exegesis powered by Supabase pgvector and Gemini."
)

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from local dev servers and production domains
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)

# Load and validate Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_PUBLISHABLE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Warning: Supabase credentials are missing from environment variables.")

if not GEMINI_API_KEY:
    print("Warning: Gemini API Key is missing from environment variables.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize Supabase Client
supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Failed to initialize Supabase client: {e}")

# Data Models
class HistoryItem(BaseModel):
    query: Optional[str] = None
    response: Optional[str] = None

class StudyStepRequest(BaseModel):
    query: str
    history: Optional[List[HistoryItem]] = []

class StudyStepResponse(BaseModel):
    step_title: str
    content: str
    sources: List[str]

# Routes
@app.get("/", tags=["Health Check"])
def read_root():
    return {
        "status": "online",
        "service": "Hermeneutics Bible Study Assistant API",
        "version": "1.0.0"
    }

@app.post("/api/v1/next-step", response_model=StudyStepResponse, tags=["Study Assistant"])
async def generate_next_step(request: StudyStepRequest):
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Query cannot be empty."
        )

    try:
        # Retrieve context from Supabase (if configured)
        retrieved_contexts = []
        sources = []
        
        if supabase:
            # Perform vector search / query against Supabase
            # Adjust table/rpc name based on your schema setup
            try:
                res = supabase.from_("documents").select("content, source").limit(3).execute()
                if res.data:
                    for doc in res.data:
                        retrieved_contexts.append(doc.get("content", ""))
                        if doc.get("source") and doc.get("source") not in sources:
                            sources.append(doc.get("source"))
            except Exception as db_err:
                print(f"Supabase query warning: {db_err}")

        # Construct prompt for Gemini
        context_str = "\n---\n".join(retrieved_contexts) if retrieved_contexts else "No specific document context retrieved."
        
        prompt = f"""You are an expert hermeneutics and biblical exegesis co-pilot.
Provide a clear, structured step-by-step response to help analyze the following query.

User Query: {request.query}

Retrieved Hermeneutical Context:
{context_str}

Format your response as a helpful study guide step.
"""

        # Call Gemini model
        if GEMINI_API_KEY:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            generated_text = response.text if response else "No response generated."
        else:
            generated_text = f"Received query: '{request.query}'. (Gemini API key not configured on backend)."

        return StudyStepResponse(
            step_title=f"Analysis: {request.query[:30]}...",
            content=generated_text,
            sources=sources if sources else ["Hermeneutical Reference Index"]
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response: {str(e)}"
        )