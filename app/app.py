import streamlit as st
from google import genai
from google.genai import types
from pathlib import Path
import numpy as np
import re


# =========================================================
# SETTINGS
# =========================================================

GEMINI_MODEL = "gemini-2.5-flash"

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


# =========================================================
# GEMINI CLIENT
# =========================================================

if "GEMINI_API_KEY" not in st.secrets:
    st.error("GEMINI_API_KEY is not configured in Streamlit Secrets.")
    st.stop()

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)


# =========================================================
# LOAD KNOWLEDGE
# =========================================================

@st.cache_data
def load_documents():

    documents = []

    for file in DATA_DIR.glob("*.txt"):

        text = file.read_text(encoding="utf-8")

        paragraphs = text.split("\n\n")

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if paragraph:

                documents.append({
                    "text": paragraph,
                    "source": file.name
                })

    return documents


# =========================================================
# SIMPLE RAG RETRIEVAL
# =========================================================

def tokenize(text):

    return set(
        re.findall(
            r"\b[a-zA-Z]{3,}\b",
            text.lower()
        )
    )


def similarity(query, document):

    query_words = tokenize(query)
    document_words = tokenize(document)

    if not query_words or not document_words:
        return 0

    common_words = query_words.intersection(
        document_words
    )

    return len(common_words) / np.sqrt(
        len(query_words) * len(document_words)
    )


def search_knowledge(query):

    documents = load_documents()

    scored_documents = []

    for document in documents:

        score = similarity(
            query,
            document["text"]
        )

        scored_documents.append(
            (
                score,
                document
            )
        )

    scored_documents.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        item[1]
        for item in scored_documents[:5]
    ]


# =========================================================
# GEMINI EVALUATION
# =========================================================

def evaluate_idea(idea, retrieved_documents):

    knowledge = ""

    for document in retrieved_documents:

        knowledge += (
            "\nSOURCE: "
            + document["source"]
            + "\n"
            + document["text"]
            + "\n"
        )

    prompt = f"""
You are a professional Hackathon Idea Evaluator AI.

Evaluate the submitted hackathon idea using the
retrieved knowledge from the RAG knowledge base.

IMPORTANT:
- Use the retrieved knowledge as the main basis.
- Do not invent facts from the knowledge base.
- Give realistic and practical feedback.
- Scores must be from 1 to 10.
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

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text


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
# CSS
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

st.mark