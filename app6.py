import os
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

ai_client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-3.6-flash"
DEFAULT_DISTANCE_THRESHOLD = 0.9  # cosine distance; higher = more permissive
REFUSAL_MESSAGE = (
    "I can only help with concepts covered in my verified course material. "
    "That topic is outside my curriculum, so I cannot assist with it."
)

SYSTEM_RULES = """You are a patient Socratic STEM tutor.
Your rules:
1. NEVER reveal the final answer or perform the full calculation for the student.
2. Use the textbook context given to you in each message to guide your response.
   If the context is at least loosely related to the question, use it to point the
   student toward the right concept — do not refuse just because it isn't a perfect
   or complete match. Only say a topic isn't covered if the context is clearly
   unrelated to the question.
3. Respond with exactly:
   - One short sentence pointing to the relevant concept or formula.
   - One direct question asking the student to take the very next step.
4. Keep responses short — 1 to 3 sentences total.
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
if "chat_session" not in st.session_state:
    st.session_state.chat_session = ai_client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_RULES),
    )

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

    # If the student uploaded only an image,
    # create a simple question for the tutor.
    if not user_question:
        user_question = "Please help me understand this problem."

    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    # -------------------------------------------------------
    # IMAGE UNDERSTANDING
    # -------------------------------------------------------
    # If an image is uploaded, first ask Gemini to identify
    # what the image is about. This description is then used
    # to search the verified textbook content.
    # -------------------------------------------------------

    search_query = user_question

    is_new_image = (
        pil_image is not None
        and current_image_id != st.session_state.last_sent_image_id
    )

    if is_new_image:
        try:
            image_analysis_prompt = """
Analyze this student's uploaded problem image.

Identify:
1. The subject/topic shown in the image.
2. The type of problem or concept involved.
3. Important keywords that would help search a textbook.

Return ONLY a short search description.
Do not solve the problem.
Do not give the final answer.
"""

            image_analysis_response = ai_client.models.generate_content(
                model=MODEL_NAME,
                contents=[pil_image, image_analysis_prompt]
            )

            image_description = image_analysis_response.text.strip()

            if image_description:
                search_query = (
                    f"{image_description}\n\n"
                    f"Student question: {user_question}"
                )

        except Exception:
            # If image analysis fails, keep the original
            # text-question retrieval behavior.
            search_query = user_question

    # -------------------------------------------------------
    # QUERY CHROMADB
    # -------------------------------------------------------

    try:
        results = collection.query(
            query_texts=[search_query],
            n_results=min(3, chunk_count) if chunk_count else 1,
            include=["documents", "distances", "metadatas"],
        )

        docs = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []

    except Exception:
        docs, distances, metadatas = [], [], []

    # -------------------------------------------------------
    # DEBUG INFORMATION
    # -------------------------------------------------------

    if show_debug:
        with st.expander("🔍 Debug: retrieval scores", expanded=False):

            if docs:
                for doc, dist, meta in zip(docs, distances, metadatas):

                    flag = (
                        "✅ under threshold"
                        if dist <= distance_threshold
                        else "❌ over threshold"
                    )

                    st.write(
                        f"**Distance {dist:.3f}** "
                        f"({flag}) — "
                        f"{meta.get('citation', '?')}"
                    )

                    st.caption(
                        doc[:200] +
                        ("..." if len(doc) > 200 else "")
                    )

            else:
                st.write(
                    "No chunks were returned — "
                    "the knowledge base may be empty."
                )

    # -------------------------------------------------------
    # CURRICULUM RELEVANCE CHECK
    # -------------------------------------------------------

    if not docs or distances[0] > distance_threshold:

        reply = REFUSAL_MESSAGE
        citation = None

    else:

        context_text = "\n\n".join(docs)

        citation = metadatas[0].get(
            "citation",
            "Curriculum Material"
        )

        # ---------------------------------------------------
        # SEND TEXTBOOK CONTEXT + IMAGE + QUESTION TO GEMINI
        # ---------------------------------------------------

        message_parts = [
            f"""
Textbook context for this question:

{context_text}

Student question:
{user_question}
"""
        ]

        if is_new_image:
            message_parts.append(pil_image)

        try:
            response = (
                st.session_state.chat_session.send_message(
                    message_parts
                )
            )

            reply = response.text

        except Exception as e:
            reply = f"Error generating response: {e}"
            citation = None

    # Remember that this image has already been processed.
    st.session_state.last_sent_image_id = current_image_id

    # -------------------------------------------------------
    # DISPLAY RESPONSE
    # -------------------------------------------------------

    with st.chat_message("assistant"):
        st.markdown(reply)

        if citation:
            st.caption(f"Source: {citation}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply,
            "citation": citation
        }
    )

