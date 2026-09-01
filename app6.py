import os
import time
import streamlit as st
import chromadb
from google import genai
from google.genai import types
from PIL import Image

# ---------------------------------------------------------------------------
# 1. PAGE SETUP & GEMINI CLIENT INITIALIZATION
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Socratic AI Tutor", page_icon="📒")
st.title("🧭 Multimodal Socratic STEM Tutor")
st.write("Upload a diagram or type your problem below. I will guide you without spoiling the answer!")

# Load the key from an environment variable or Streamlit secrets — never hardcode
# API keys in source, especially in a repo that's going to be open-sourced.
# Set it before running, e.g.:
#   export GEMINI_API_KEY="your-key-here"          (macOS/Linux)
#   $env:GEMINI_API_KEY="your-key-here"             (Windows PowerShell)
# or add it to .streamlit/secrets.toml as GEMINI_API_KEY = "your-key-here"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    st.error(
        "No Gemini API key found. Set the GEMINI_API_KEY environment variable "
        "or add it to `.streamlit/secrets.toml`, then restart the app."
    )
    st.stop()

MODEL_NAME = "gemini-3.6-flash"

def get_ai_client():
    return genai.Client(api_key=GEMINI_API_KEY)
def generate_with_retry(contents, max_retries=3):
    for attempt in range(max_retries):
        try:
            client = get_ai_client()

            return client.models.generate_content(
                model=MODEL_NAME,
                contents=contents
            )

        except Exception as e:
            error_text = str(e)

            if "503" in error_text or "UNAVAILABLE" in error_text:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue

            raise e

DEFAULT_DISTANCE_THRESHOLD = 0.9  # cosine distance; higher = more permissive
REFUSAL_MESSAGE = (
    "I can only help with concepts covered in my verified course material. "
    "That topic is outside my curriculum, so I cannot assist with it."
)

SYSTEM_RULES = """You are a patient Socratic STEM tutor.

STRICT RULES:
1. NEVER give the final numerical answer.
2. NEVER perform the calculation to the final result.
3. NEVER state the value the student is supposed to find.
4. Do not solve the problem even if it is very easy.
5. Give only a hint about the relevant concept, formula, or first step.
6. Ask exactly ONE question that makes the student take the next step.
7. Use the provided textbook context when it is relevant.
8. Keep the response to 1–2 short sentences.
9. If the student asks directly for an answer, refuse to provide it and guide them toward finding it themselves.

Your response MUST contain:
- One short hint.
- One direct question asking for the student's next step.

Example:
Student: "What is sin 30°?"
Good response: "Use the standard trigonometric values for common angles. What value does your textbook give for sin 30°?"
Bad response: "sin 30° = 1/2."
"""

# ---------------------------------------------------------------------------
# 2. LOCAL KNOWLEDGE BASE (CHROMA DB)
# ---------------------------------------------------------------------------
@st.cache_resource
def setup_database():
    db = chromadb.Client()
    collection = db.get_or_create_collection(
        "curriculum", metadata={"hnsw:space": "cosine"}
    )

    if not os.path.exists("book.txt"):
        return collection, 0, "book.txt not found next to the app script."

    with open("book.txt", "r", encoding="utf-8") as f:
        content = f.read()

    texts, citations = [], []
    if "###" in content:
        for section in content.split("###"):
            section = section.strip()
            if not section:
                continue
            lines = section.split("\n", 1)
            citation = lines[0].strip()
            text = lines[1].strip() if len(lines) > 1 else ""
            if text:
                texts.append(text)
                citations.append(citation)
    else:
        texts = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 30]
        citations = [f"Excerpt {i + 1}" for i in range(len(texts))]

    if not texts:
        return collection, 0, "book.txt was found but no usable chunks could be parsed from it."

    collection.add(
        documents=texts,
        metadatas=[{"citation": c} for c in citations],
        ids=[f"chunk_{i}" for i in range(len(texts))],
    )
    return collection, len(texts), None


collection, chunk_count, load_warning = setup_database()

if load_warning:
    st.warning(
        f"⚠️ {load_warning} With an empty knowledge base, every question will be "
        "marked out of scope. Fix this first — the settings below won't help until "
        "there's content to retrieve."
    )

# ---------------------------------------------------------------------------
# 3. RETRIEVAL SETTINGS (SIDEBAR)
# ---------------------------------------------------------------------------
st.sidebar.header("Retrieval settings")
st.sidebar.caption(f"Loaded {chunk_count} curriculum chunk(s).")
distance_threshold = st.sidebar.slider(
    "Distance threshold (higher = more lenient)",
    min_value=0.0,
    max_value=1.2,
    value=DEFAULT_DISTANCE_THRESHOLD,
    step=0.05,
    help=(
        "Cosine distance between the question and the closest textbook chunk. "
        "If the closest match's distance is ABOVE this value, the tutor refuses "
        "as out of scope. General-purpose embeddings often sit in the 0.6–0.9 "
        "range even for genuinely related text, so 0.7 can be too strict. Watch "
        "the debug panel below while testing and raise this if on-topic "
        "questions keep getting refused."
    ),
)
show_debug = st.sidebar.checkbox("Show retrieval debug info", value=True)

# ---------------------------------------------------------------------------
# 4. CHAT SESSION STATE
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sent_image_id" not in st.session_state:
    st.session_state.last_sent_image_id = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citation"):
            st.caption(f"Source: {message['citation']}")

# ---------------------------------------------------------------------------
# 5. IMAGE UPLOADER SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.header("Problem Image (Optional)")
uploaded_image = st.sidebar.file_uploader(
    "Upload a question, diagram, or handwritten problem",
    type=["png", "jpg", "jpeg"]
)

pil_image = None
current_image_id = None
if uploaded_image:
    pil_image = Image.open(uploaded_image)
    st.sidebar.image(pil_image, caption="Uploaded Problem", use_container_width=True)
    current_image_id = f"{uploaded_image.name}-{uploaded_image.size}"
if uploaded_image:
    image_question = st.sidebar.button("Ask about this image")
else:
    image_question = False

# ---------------------------------------------------------------------------
# 6. USER CHAT INTERACTION
# ---------------------------------------------------------------------------
user_question = st.chat_input("Ask for help on this problem...")

if user_question or image_question:
    if not user_question:
        user_question = "Please help me understand this problem."
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Query ChromaDB vector store (wrapped — some Chroma versions raise on
    # an empty collection instead of returning empty lists)
    try:
        results = collection.query(
            query_texts=[user_question],
            n_results=min(3, chunk_count) if chunk_count else 1,
            include=["documents", "distances", "metadatas"],
        )
        docs = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
    except Exception:
        docs, distances, metadatas = [], [], []

    if show_debug:
        with st.expander("🔍 Debug: retrieval scores", expanded=False):
            if docs:
                for doc, dist, meta in zip(docs, distances, metadatas):
                    flag = "✅ under threshold" if dist <= distance_threshold else "❌ over threshold"
                    st.write(f"**Distance {dist:.3f}** ({flag}) — {meta.get('citation', '?')}")
                    st.caption(doc[:200] + ("..." if len(doc) > 200 else ""))
            else:
                st.write("No chunks were returned — the knowledge base may be empty.")

    # Apply distance threshold guardrail
    if not docs or distances[0] > distance_threshold:
        reply, citation = REFUSAL_MESSAGE, None
    else:
        context_text = "\n\n".join(docs)
        citation = metadatas[0].get("citation", "Curriculum Material")

        is_new_image = pil_image is not None and current_image_id != st.session_state.last_sent_image_id

        message_parts = [
            f"Textbook context for this question:\n{context_text}\n\nStudent question: {user_question}"
        ]
        if is_new_image:
            message_parts.append(pil_image)

        try:
            response = generate_with_retry(message_parts)
            reply = response.text

        except Exception as e:
            reply = "The AI service is temporarily busy. Please try again in a moment."
            citation = None

    st.session_state.last_sent_image_id = current_image_id

    with st.chat_message("assistant"):
        st.markdown(reply)
        if citation:
            st.caption(f"Source: {citation}")

    st.session_state.messages.append({"role": "assistant", "content": reply, "citation": citation})
