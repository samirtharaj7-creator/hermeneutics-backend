import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from supabase import create_client, Client

load_dotenv()

app = FastAPI(
    title="Hermeneutics Bible Study Assistant API",
    version="1.0.0"
)

# Initialize Clients
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase credentials are missing in environment variables.")

if not api_key:
    raise RuntimeError("Gemini API key is missing in environment variables.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
ai_client = genai.Client(api_key=api_key)


class StudyStepRequest(BaseModel):
    passage: str
    current_stage: str
    user_notes: str = ""


@app.get("/")
async def root():
    return {"message": "Hermeneutics Bible Study Assistant API is running"}


@app.post("/api/v1/next-step")
async def generate_next_step(request: StudyStepRequest):
    try:
        # 1. Embed current stage and passage query
        query_text = f"Methodology and procedural steps for {request.current_stage} when studying {request.passage}"
        
        embed_resp = ai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=query_text,
            config={"output_dimensionality": 768}
        )
        query_vector = embed_resp.embeddings[0].values

        # 2. Retrieve top procedural chunks from Supabase
        rpc_resp = supabase.rpc(
            "match_study_step",
            {
                "query_embedding": query_vector,
                "match_threshold": 0.15,
                "match_count": 4,
                "priority_boost": 2.0
            }
        ).execute()

        retrieved_context = "\n\n".join([
            f"Source: {item['book_title']}\n{item['content']}" 
            for item in rpc_resp.data
        ]) if rpc_resp.data else "No specific procedural guidelines retrieved."

        # 3. Construct Gemini Prompt enforcing procedural biblical guidance
        system_prompt = f"""
You are an expert Hermeneutics Bible Study Guide.
Your purpose is NOT to give a commentary or answer general questions, but to lead the user through the exact procedural step of studying their passage.

Current Study Parameters:
- Passage: {request.passage}
- Current Stage: {request.current_stage}
- User's Current Notes/Observations: {request.user_notes if request.user_notes else 'None provided yet'}

Methodological Guidance Context (Prioritize General & Special Hermeneutics procedures):
{retrieved_context}

Instructions for your response:
1. Briefly state the goal of the current step ({request.current_stage}).
2. Provide 3 concrete, actionable questions/tasks the user must answer about {request.passage} based on the retrieved hermeneutical principles.
3. Instruct the user what to analyze and send back to proceed to the next step.
"""

        # 4. Generate content using gemini-3.5-flash
        response = ai_client.models.generate_content(
            model="gemini-3.5-flash",
            contents=system_prompt
        )

        return {
            "passage": request.passage,
            "stage": request.current_stage,
            "guidance": response.text,
            "sources_used": [item['book_title'] for item in rpc_resp.data] if rpc_resp.data else []
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))