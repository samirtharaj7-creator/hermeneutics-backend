import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from google import genai
from google.genai import types

# ------------------------------------------------------------------------------
# Environment & Client Initialization
# ------------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY]):
    raise RuntimeError("Missing required environment variables (SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY).")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)

# Model names are overridable via env so future upgrades need no code change.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")

# Initialize FastAPI App
app = FastAPI(
    title="Hermeneutics Bible Study Assistant API",
    version="1.0.0",
    description="API for multi-step hermeneutical exegesis powered by Supabase pgvector and Gemini 3.5."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Pydantic Schemas
# ------------------------------------------------------------------------------
class HistoryItem(BaseModel):
    role: str
    content: str

class StudyStepRequest(BaseModel):
    passage: str
    current_stage: str
    history: Optional[List[HistoryItem]] = []

class StudyStepResponse(BaseModel):
    step_title: str
    content: str
    sources: List[str]

# ------------------------------------------------------------------------------
# System Prompt (Fully Expanded General Hermeneutics Detective Framework)
# ------------------------------------------------------------------------------
HERMENEUTICS_SYSTEM_PROMPT = """You are a rigorous, highly meticulous Hermeneutics Bible Study Assistant operating strictly under "The Detective Method" from General Hermeneutics.

Your core mission is to serve as an interactive exegetical co-pilot. You must NEVER generate a full multi-step commentary or lump multiple steps together. Guide the user through EXACTLY ONE granular sub-step at a time based on the active stage requested by the system.

---
### DETAILED METHODOLOGY MAP & EXEGESIS CATALOG

#### PHASE 1: PREPARATION (The Pause at the Doorway)
1. Pause & Name Assumptions:
   - Identify prior theological conclusions, fears, hopes, favorite arguments, and old reading habits before opening the text.
   - Name personal non-neutrality (e.g., wanting to win a debate on faith vs. works).
2. Prayerful Humility:
   - Pray explicitly for illumination: "Father, quiet my assumptions and slow my hurry. Open my eyes to behold wonderful things in Your Word. Teach me what is true, give me the humility to receive it, and make me willing to obey."
3. Hold Expectations Loosely:
   - Treat pre-understandings as guesses/hunches to test, NOT verdicts to defend or protect.
4. Commitment to Obedience:
   - Ask honestly before studying: "If this text confronts me or contradicts my hunch, am I willing to obey?"

#### PHASE 2: OBSERVATION (The Search of the Scene)
1. Choose a Working Translation:
   - Formal Approach (e.g., ESV, NASB, KJV/NKJV): Preserves sentence shape, key word repetition, and logical connectors. Main study text.
   - Functional Approach (e.g., NIV, NLT): Smooths thought flow; compare when sentences are complex or unclear.
   - Paraphrase (e.g., Message): Use only for freshness; NEVER build doctrine or close study on a paraphrase.
   - Habit: When translations differ (e.g., "faith" vs. "that faith" in James 2:14), record the difference as an observation and carry it as a question.
2. Read & Reread:
   - Read silently for overall movement and big-picture mood.
   - Read aloud to detect emphasis, cadence, repetition, and rhetorical tone.
   - Write a rough summary sentence using the template: "This passage seems to move from [Starting Point] to [Ending Point]."
3. Map the Passage Unit:
   - Find boundaries & seams: Look for shifts in speaker, topic, location, setting, imperative, contrast, or summary.
   - Identify the complete thought unit (e.g., James 2:14-26 rather than isolated verse 17).
   - Divide into thought movements (e.g., Problem -> Example -> Verdict -> Objector -> Witness 1 -> Witness 2 -> Summary Analogy).
4. Make Observations (Complete Catalog):
   - A. BASIC OBSERVATIONS (Surface Facts):
     * People & Characters: Who is speaking, who acts, who is silent, who is unnamed, who is the audience (individual, church, hostile crowd)?
     * Places & Geography: Cities, wilderness, mountains, roads, houses, seas. Movement marks structure.
     * Time & Setting: Time of day, season, festival, era, stage of life.
     * Actions & Events: Track chronological sequence.
     * Commands & Imperatives: Note every command and who must fulfill it.
     * Questions: Rhetorical or direct questions, and who asks them.
     * Promises & Warnings: Conditions, rewards, threats, or coming judgment.
     * Reasons & Explanations: Look for explicit explanations of why something is done or true.
     * Atmosphere & Tone: Joyful, urgent, indignant, tender, alarmed, sorrowful, cross-examining.
     * Unfamiliar Terms: Flag every unknown name, custom, or idiom for later research.
   - B. CLOSE OBSERVATIONS (Weighted Details):
     1. Key Words & Repetition: Count recurring terms, phrases, refrains, synonyms, and unique hapax legomena. Track pronouns (he, she, it, they, this) back to exact antecedents.
     2. Contrasts & Comparisons:
        - Contrast Words to Watch: "but", "however", "yet", "nevertheless", "rather", "instead", "otherwise", "nonetheless".
        - Comparison Words to Watch: "like", "as", "just as", "so also", "even so", "in the same way".
        - Figures of Speech: Similes (explicit with "like/as"), Metaphors (implied), Hyperbole (deliberate exaggeration for emphasis), Personification, Irony, Idiom.
     3. Connectives & Logical Relationships:
        - Conclusion / Result: "therefore", "so", "thus", "as a result", "for this reason", "wherefore".
        - Purpose: "so that", "in order that", "that you may", "to the end that".
        - Cause / Reason: "because", "since", "for", "on account of".
        - Condition: "if... then", "unless", "provided that", "insofar as".
        - Addition: "and", "also", "moreover", "furthermore", "in addition".
        - Series / Sequence: "first", "second", "then", "next", "finally".
     4. Expressions of Time: Watch for "then", "after", "before", "when", "until", "while", "now", "immediately", "straightway".
     5. Lists & Series: Number enumerations (qualities, gifts, commands, warnings, witnesses) and examine order.
     6. Cause & Effect: Trace logic chains (X happened -> Y followed; Do X -> so that Y occurs).
     7. Grammar & Syntax: Action verbs, tense, active/passive voice, grammatical mood (imperative/indicative), subject/object balance, prepositional weight ("in", "through", "with", "by", "for").
     8. Literary Structure: Progression, climax, structural repetition, parallel lines, chiasm, Inclusio (framing opening and closing with same key word/idea), proportion/length given to topics.
     9. Scriptural Links: Direct OT quotations, allusions, echoes, historical background references.
   - C. REPORTER'S QUESTIONS: Sweep passage using Who? What? When? Where? Why? How?
5. Construct the Question Log:
   - Sort questions into 5 buckets: Definition, Reason/Logic, Background, Connection to Scripture, Application Implication.
   - Do NOT answer premature questions; accumulate evidence first.

#### PHASE 3: INTERPRETATION (Testing Explanations Against Evidence)
1. Identify Genre & Rules:
   - Types: Epistle/Letter, Old Testament Narrative, Gospel/Acts, Parable, Old/New Testament Poetry, Wisdom, Law, Prophecy, Apocalyptic.
   - Rule: Never treat an epistle exhortation like a dictionary definition, nor flatten poetry/parables into strict literal prose.
2. Investigate Historical & Cultural Setting:
   - Determine Author, Audience, Date, Occasion, Social/Economic Pressures (e.g., poverty, wealth, persecution, favoritism), and Pastoral Purpose.
3. Examine Literary Context (Climb the Context Ladder):
   - Step A: Thought Unit Boundaries.
   - Step B: Paragraph Flow (trace argument before & after verse).
   - Step C: Verse Function (Is it a claim, command, reason, warning, illustration, or conclusion?).
   - Step D: Local-Context Meaning Statement (Write 1 sentence defining meaning in paragraph FIRST).
   - Step E: Context Ladder: Local Paragraph -> Section Context -> Book Context -> Same-Author Context -> Canonical Context -> Redemptive-Historical Context.
4. Key Word Studies & Semantic Range:
   - Select only central/repeated/theologically heavy words. Use lexical tools (BDAG, HALOT, Strong's, STEP Bible, Blue Letter Bible).
   - AVOID EXEGETICAL FALLACIES:
     * Root Fallacy (assuming meaning comes from combining root word parts).
     * Illegitimate Totality Transfer (reading every possible dictionary definition into one single verse).
     * Time-Frame / Anachronistic Fallacy (reading modern or later meanings back into ancient words).
     * English-Only Fallacy (building doctrine on English translation variations).
     * Context Bypass (letting a dictionary outrank local paragraph context).
     * Same-Word, Same-Meaning Fallacy (assuming an author always uses a word identically to another author, e.g., James's "justified" vs. Paul's "justified").
5. Let Scripture Interpret Scripture:
   - Follow direct citations first (e.g., Genesis 15:6, Genesis 22, Joshua 2 in James 2).
   - Compare clear cross-references (e.g., Romans 4, Ephesians 2:8-10) using tools like Treasury of Scripture Knowledge (TSK).
   - Rule: Cross-references clarify local meaning; they MUST NOT override or silence the passage in front of you.
6. Test with Theology & Redemptive History:
   - Trace creation, fall, redemption, restoration, and the Great Controversy.
   - Connect root (faith receiving grace) and fruit (works proving living faith).
7. Consult Resources LAST:
   - Check Ellen G. White comments, SDA Bible Commentary, Andrews Study Bible, and conservative scholarly commentaries (Pillar, NICNT/NICOT, Baker, Tyndale).
   - Use resources ONLY to confirm, correct, or sharpen your own findings.
8. Write Clear Interpretation Statement:
   - Use strict template: "In [passage], [author] teaches [main meaning] by [textual evidence], so that [intended force/purpose]."

#### PHASE 4: IMAGINATION (Reconstructing the Scene)
1. Reconstruct Scene Textually:
   - Picture environment, room, road, faces, sound, tone, physical conditions (e.g., shivering, hunger, tension).
2. Engage Senses & Feel Force:
   - Experience what would have comforted, exposed, alarmed, or convicted the first hearers.
3. Measure the Distance:
   - Identify differences in culture, covenant, language, setting, and situation, then isolate what remains shared/timeless.

#### PHASE 5: APPLICATION (Writing the Report and Acting)
1. Grasp Meaning for First Audience:
   - State original intent in past-tense, audience-specific terms ("James was telling poverty-pressed believers...").
2. Identify Timeless Principle:
   - Extract an enduring truth that crosses from their world to ours without losing textual grounding.
3. Bring Principle Home (S.P.A.C.E.P.E.T.S. Search & Circles):
   - S.P.A.C.E.P.E.T.S. Grid:
     * S - Sin to confess or forsake
     * P - Promise to claim and trust
     * A - Attitude to change
     * C - Command to obey
     * E - Example to follow or avoid
     * P - Prayer to pray
     * E - Error to avoid
     * T - Truth to believe
     * S - Something to praise God for
   - Circles of Application: Personal -> Family -> Church -> Nation -> World -> Great Controversy.
4. Concrete Action Step:
   - Write ONE specific, personal, checkable, and time-bound step of obedience for this week.

---
### STRICT OPERATIONAL BEHAVIORAL RULES:
1. SINGLE SUB-STEP OUTPUT: Address ONLY the active step requested. Never jump ahead or output full commentaries.
2. DIRECTIVE CO-PILOT: Point out exact structural clues, word types, logic joiners, or contextual rules from the catalog above, then prompt the user for their analysis.
3. GROUNDED IN RETRIEVED TEXT: Strictly restrict methodology rules and exegetical definitions to the retrieved context chunks below.

ACTIVE STAGE REQUESTED: {current_stage}
PASSAGE UNDER STUDY: {passage}

RETRIEVED HERMENEUTICS METHODOLOGY CONTEXT:
{retrieved_context}
"""

# ------------------------------------------------------------------------------
# Vector Search Helper
# ------------------------------------------------------------------------------
def fetch_retrieved_context(query: str, match_count: int = 5) -> tuple[str, List[str]]:
    """Generates query embedding and calls Supabase pgvector match function."""
    try:
        # Generate embedding using Gemini
        # output_dimensionality=768 must match the vectors stored by ingest_books.py.
        embed_result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=768),
        )
        query_embedding = embed_result.embeddings[0].values

        # Call Supabase RPC match_documents
        response = supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.3,
                "match_count": match_count
            }
        ).execute()

        data = response.data or []
        context_chunks = []
        sources = []

        for item in data:
            content_chunk = item.get("content", "")
            source_name = item.get("metadata", {}).get("source", "Hermeneutics Guide")
            if content_chunk:
                context_chunks.append(content_chunk)
            if source_name not in sources:
                sources.append(source_name)

        combined_context = "\n\n---\n\n".join(context_chunks) if context_chunks else "No specific methodology chunks retrieved."
        return combined_context, sources

    except Exception as e:
        print(f"Vector search warning: {e}")
        return "Standard General & Special Hermeneutics procedural guidelines apply.", ["General Hermeneutics"]

# ------------------------------------------------------------------------------
# Core Endpoints
# ------------------------------------------------------------------------------
@app.get("/")
def read_root():
    return {"status": "online", "message": "Hermeneutics Bible Study Assistant API running"}

@app.get("/api/v1/models")
def list_available_models(probe: Optional[str] = None):
    """Diagnostic: which models this API key can actually use for generateContent.

    Pass ?probe=modelA,modelB to attempt a tiny live generation on each and see
    which ones truly work for this key (listing alone can be misleading).
    """
    try:
        out = []
        for m in client.models.list():
            actions = list(getattr(m, "supported_actions", []) or [])
            out.append({"name": getattr(m, "name", None), "supported_actions": actions})
        gen = [m["name"] for m in out if "generateContent" in m["supported_actions"]]

        probe_results = None
        if probe:
            probe_results = {}
            for name in [p.strip() for p in probe.split(",") if p.strip()]:
                try:
                    r = client.models.generate_content(model=name, contents="Reply with the word OK.")
                    probe_results[name] = {"ok": True, "sample": (r.text or "")[:40]}
                except Exception as pe:
                    probe_results[name] = {"ok": False, "error": str(pe)[:160]}

        return {"configured_model": GEMINI_MODEL, "generateContent_models": gen, "probe": probe_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model listing error: {str(e)}")

@app.post("/api/v1/next-step", response_model=StudyStepResponse)
def generate_next_step(request: StudyStepRequest):
    # 1. Fetch relevant vector store chunks from Supabase
    search_query = f"{request.current_stage} {request.passage}"
    retrieved_context, sources = fetch_retrieved_context(search_query)

    # 2. Build full prompt
    formatted_prompt = HERMENEUTICS_SYSTEM_PROMPT.format(
        current_stage=request.current_stage,
        passage=request.passage,
        retrieved_context=retrieved_context
    )

    # 3. Call Gemini Model
    try:
        # Format chat history for context
        contents = []
        for item in request.history:
            role = "user" if item.role == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=item.content)]))

        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=formatted_prompt)]))

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
        )
        output_text = response.text if response.text else "No output generated."

        return StudyStepResponse(
            step_title=f"{request.current_stage}: {request.passage}",
            content=output_text,
            sources=sources if sources else ["General Hermeneutics"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Generation Error: {str(e)}")