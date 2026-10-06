import streamlit as st
import requests
import re
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


# =========================================================
# SETTINGS
# =========================================================

OLLAMA_URL = "http://localhost:11434"

LLM_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
QDRANT_DIR = BASE_DIR / "qdrant_storage"

COLLECTION_NAME = "hackathon_knowledge"


# =========================================================
# OLLAMA EMBEDDING
# =========================================================

def get_embedding(text):

    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": text
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["embeddings"][0]


# =========================================================
# LOAD KNOWLEDGE FILES
# =========================================================

def load_documents():

    documents = []

    for file in DATA_DIR.glob("*.txt"):

        text = file.read_text(encoding="utf-8")

        paragraphs = text.split("\n\n")

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if paragraph:

                documents.append(
                    {
                        "text": paragraph,
                        "source": file.name
                    }
                )

    return documents


# =========================================================
# CREATE / LOAD QDRANT DATABASE
# =========================================================

@st.cache_resource
def create_database():

    client = QdrantClient(
        path=str(QDRANT_DIR)
    )

    collections = client.get_collections().collections

    collection_names = []

    for collection in collections:
        collection_names.append(collection.name)

    if COLLECTION_NAME not in collection_names:

        documents = load_documents()

        if len(documents) == 0:

            st.error(
                "No knowledge files found in data folder."
            )

            st.stop()

        first_embedding = get_embedding(
            documents[0]["text"]
        )

        vector_size = len(first_embedding)

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

        points = []

        for i in range(len(documents)):

            embedding = get_embedding(
                documents[i]["text"]
            )

            points.append(
                PointStruct(
                    id=i,
                    vector=embedding,
                    payload={
                        "text": documents[i]["text"],
                        "source": documents[i]["source"]
                    }
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

    return client


# =========================================================
# SEARCH RAG KNOWLEDGE
# =========================================================

def search_knowledge(client, query):

    query_embedding = get_embedding(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=5
    ).points

    return results


# =========================================================
# ASK LLAMA
# =========================================================

def evaluate_idea(idea, retrieved_results):

    knowledge = ""

    for result in retrieved_results:

        source = result.payload["source"]
        text = result.payload["text"]

        knowledge += (
            "\nSOURCE: "
            + source
            + "\n"
            + text
            + "\n"
        )

    prompt = f"""
You are a professional Hackathon Idea Evaluator AI.

Evaluate the submitted hackathon idea using the retrieved
knowledge from the RAG knowledge base.

IMPORTANT:
- Use retrieved knowledge as the main basis.
- Do not invent facts from the knowledge base.
- Give realistic and practical feedback.
- Scores must be from 1 to 10.
- Explain each score briefly.
- Consider technical limitations.

HACKATHON IDEA:

{idea}


RETRIEVED KNOWLEDGE:

{knowledge}


Evaluate using these criteria:

Problem Relevance
Innovation
Technical Feasibility
Impact
User Experience
Scalability
Sustainability


Use EXACTLY this format:

Problem Relevance: X/10
Innovation: X/10
Technical Feasibility: X/10
Impact: X/10
User Experience: X/10
Scalability: X/10
Sustainability: X/10

STRENGTHS:

- point
- point
- point

WEAKNESSES:

- point
- point
- point

SUGGESTIONS:

- point
- point
- point

FINAL VERDICT:

Give a short overall judgement.
"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=300
    )

    response.raise_for_status()

    return response.json()["response"]


# =========================================================
# GET SCORE
# =========================================================

def get_score(text, criterion):

    pattern = (
        re.escape(criterion)
        + r":\s*(\d+)\s*/\s*10"
    )

    match = re.search(
        pattern,
        text,
        re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    return None


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Hackathon Idea Evaluator AI",
    page_icon="🏆",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .score-box {
        text-align: center;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-bottom: 20px;
    }

    .overall-score {
        font-size: 42px;
        font-weight: 700;
    }

    .decision {
        font-size: 20px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🏆 Hackathon Idea Evaluator AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Evaluate your hackathon idea using LLM + RAG'
    '</div>',
    unsafe_allow_html=True
)


st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🔎 How It Works")

    st.write(
        "1️⃣ Enter your hackathon idea."
    )

    st.write(
        "2️⃣ RAG retrieves relevant knowledge."
    )

    st.write(
        "3️⃣ Llama 3.2 evaluates the idea."
    )

    st.write(
        "4️⃣ Scores and recommendations are generated."
    )

    st.divider()

    st.header("⚙️ Technology")

    st.write("• Llama 3.2 3B")

    st.write("• Nomic Embed")

    st.write("• Qdrant")

    st.write("• Streamlit")

    st.write("• Python")


# =========================================================
# IDEA INPUT
# =========================================================

st.subheader("💡 Enter Your Hackathon Idea")

idea = st.text_area(
    "",
    placeholder=(
        "Example: Our idea is an AI system that "
        "detects potholes using smartphone cameras "
        "and GPS."
    ),
    height=170
)


# =========================================================
# EVALUATE BUTTON
# =========================================================

if st.button(
    "🚀 Evaluate Idea",
    type="primary",
    use_container_width=True
):

    if idea.strip() == "":

        st.warning(
            "Please enter a hackathon idea first."
        )

    else:

        try:

            # -------------------------------------------------
            # RAG DATABASE
            # -------------------------------------------------

            with st.spinner(
                "🔎 Searching knowledge base..."
            ):

                client = create_database()


            # -------------------------------------------------
            # RETRIEVAL
            # -------------------------------------------------

            with st.spinner(
                "📚 Retrieving relevant knowledge..."
            ):

                results = search_knowledge(
                    client,
                    idea
                )


            # -------------------------------------------------
            # LLM
            # -------------------------------------------------

            with st.spinner(
                "🤖 Llama 3.2 is evaluating your idea..."
            ):

                evaluation = evaluate_idea(
                    idea,
                    results
                )


            # -------------------------------------------------
            # GET SCORES
            # -------------------------------------------------

            criteria = [
                "Problem Relevance",
                "Innovation",
                "Technical Feasibility",
                "Impact",
                "User Experience",
                "Scalability",
                "Sustainability"
            ]

            scores = {}

            for criterion in criteria:

                scores[criterion] = get_score(
                    evaluation,
                    criterion
                )


            valid_scores = []

            for score in scores.values():

                if score is not None:

                    valid_scores.append(score)


            # -------------------------------------------------
            # OVERALL SCORE
            # -------------------------------------------------

            if len(valid_scores) > 0:

                overall_score = (
                    sum(valid_scores)
                    / len(valid_scores)
                )

                overall_score = round(
                    overall_score,
                    1
                )

            else:

                overall_score = 0


            # -------------------------------------------------
            # DECISION
            # -------------------------------------------------

            if overall_score >= 8:

                decision = "🟢 Strong Idea"

            elif overall_score >= 6:

                decision = "🟡 Needs Improvement"

            else:

                decision = "🔴 Weak Idea"


            # =================================================
            # RESULT HEADER
            # =================================================

            st.success(
                "Evaluation completed successfully!"
            )


            st.divider()


            # =================================================
            # OVERALL SCORE
            # =================================================

            st.subheader(
                "🏆 Overall Evaluation"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    f"""
                    <div class="score-box">

                    <div>Overall Score</div>

                    <div class="overall-score">
                    {overall_score}/10
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:

                st.markdown(
                    f"""
                    <div class="score-box">

                    <div>Decision</div>

                    <div class="decision">
                    {decision}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # =================================================
            # INDIVIDUAL SCORES
            # =================================================

            st.subheader(
                "📊 Detailed Scores"
            )

            row1 = st.columns(4)

            for i in range(4):

                criterion = criteria[i]

                score = scores[criterion]

                if score is None:
                    value = "N/A"
                else:
                    value = f"{score}/10"

                row1[i].metric(
                    criterion,
                    value
                )


            row2 = st.columns(3)

            for i in range(3):

                criterion = criteria[i + 4]

                score = scores[criterion]

                if score is None:
                    value = "N/A"
                else:
                    value = f"{score}/10"

                row2[i].metric(
                    criterion,
                    value
                )


            st.divider()


            # =================================================
            # AI EVALUATION
            # =================================================

            st.subheader(
                "🤖 AI Evaluation"
            )

            st.markdown(
                evaluation
            )


            # =================================================
            # RAG EVIDENCE
            # =================================================

            st.divider()

            st.subheader(
                "📚 RAG Evidence"
            )

            st.write(
                "The following knowledge was retrieved "
                "from the project knowledge base before "
                "the LLM generated the evaluation."
            )

            for i in range(len(results)):

                source = results[i].payload["source"]

                text = results[i].payload["text"]

                with st.expander(
                    f"📄 Source {i + 1}: {source}"
                ):

                    st.write(text)


        except Exception as e:

            st.error(
                "An error occurred."
            )

            st.write(
                "Please check the error below:"
            )

            st.code(
                str(e)
            )