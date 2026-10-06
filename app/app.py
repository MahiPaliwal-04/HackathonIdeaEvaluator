import streamlit as st
from google import genai
from google.genai import types
from pathlib import Path
import numpy as np
import re


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hackathon Idea Evaluator AI",
    page_icon="🚀",
    layout="wide"
)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

GEMINI_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-001"


# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource
def get_gemini_client():

    if "GEMINI_API_KEY" not in st.secrets:
        st.error("Gemini API key is not configured.")
        st.stop()

    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )


client = get_gemini_client()


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

@st.cache_data
def load_knowledge_base():

    chunks = []

    txt_files = list(DATA_DIR.glob("*.txt"))

    for file_path in txt_files:

        text = file_path.read_text(
            encoding="utf-8"
        )

        paragraphs = text.split("\n\n")

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if paragraph:
                chunks.append(
                    {
                        "source": file_path.name,
                        "text": paragraph
                    }
                )

    return chunks


knowledge_base = load_knowledge_base()


# ============================================================
# GENERATE EMBEDDING
# ============================================================

def get_embedding(text):

    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=768
        )
    )

    return np.array(
        result.embeddings[0].values
    )


# ============================================================
# PREPARE KNOWLEDGE BASE EMBEDDINGS
# ============================================================

@st.cache_data
def create_knowledge_embeddings(texts):

    embeddings = []

    for text in texts:

        embedding = get_embedding(text)

        embeddings.append(embedding)

    return np.array(embeddings)


knowledge_texts = [
    item["text"]
    for item in knowledge_base
]

if len(knowledge_texts) == 0:

    st.error(
        "No knowledge base files were found in the data folder."
    )

    st.stop()


knowledge_embeddings = create_knowledge_embeddings(
    knowledge_texts
)


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(query_vector, vectors):

    query_norm = np.linalg.norm(query_vector)

    vector_norms = np.linalg.norm(
        vectors,
        axis=1
    )

    similarities = np.dot(
        vectors,
        query_vector
    ) / (
        vector_norms * query_norm + 1e-10
    )

    return similarities


# ============================================================
# RAG RETRIEVAL
# ============================================================

def retrieve_relevant_knowledge(
    idea,
    top_k=5
):

    query_embedding = get_embedding(
        idea
    )

    similarities = cosine_similarity(
        query_embedding,
        knowledge_embeddings
    )

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append(
            {
                "source": knowledge_base[index]["source"],
                "text": knowledge_base[index]["text"],
                "score": float(similarities[index])
            }
        )

    return results


# ============================================================
# GEMINI EVALUATION
# ============================================================

def evaluate_idea(
    idea,
    retrieved_knowledge
):

    context = ""

    for i, item in enumerate(
        retrieved_knowledge,
        start=1
    ):

        context += f"""
SOURCE {i}: {item["source"]}

{item["text"]}

"""


    prompt = f"""
You are an expert hackathon project evaluator.

Evaluate the following hackathon idea using ONLY the
retrieved knowledge as supporting evidence.

HACKATHON IDEA:
{idea}


RETRIEVED KNOWLEDGE:
{context}


Evaluate the idea on these seven criteria.

1. Problem Relevance
2. Innovation
3. Technical Feasibility
4. Impact
5. User Experience
6. Scalability
7. Sustainability


Give a score from 1 to 10 for every criterion.

Use exactly this format:

Problem Relevance: X/10
Innovation: X/10
Technical Feasibility: X/10
Impact: X/10
User Experience: X/10
Scalability: X/10
Sustainability: X/10

Strengths:
- point
- point
- point

Weaknesses:
- point
- point
- point

Suggestions:
- point
- point
- point

Final Verdict:
Give a short overall judgement of the idea.

Important:
- Be realistic.
- Do not give every criterion the same score.
- Explain the reasoning clearly.
- Consider the retrieved hackathon knowledge.
- Do not invent facts from the knowledge base.
"""


    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text


# ============================================================
# EXTRACT SCORES
# ============================================================

def extract_scores(text):

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

        pattern = (
            re.escape(criterion)
            + r"\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10"
        )

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            scores[criterion] = float(
                match.group(1)
            )

    return scores


# ============================================================
# OVERALL SCORE
# ============================================================

def calculate_overall_score(scores):

    if not scores:
        return 0

    return sum(
        scores.values()
    ) / len(scores)


# ============================================================
# DECISION
# ============================================================

def get_decision(score):

    if score >= 8:
        return "🟢 Strong Idea"

    elif score >= 6:
        return "🟡 Needs Improvement"

    else:
        return "🔴 Weak Idea"


# ============================================================
# USER INTERFACE
# ============================================================

st.title(
    "🚀 Hackathon Idea Evaluator AI"
)

st.markdown(
    """
### AI-powered Hackathon Idea Evaluation using RAG + LLM

Enter your hackathon idea and get an AI-based evaluation
using relevant hackathon judging criteria, successful
project examples and technical feasibility guidelines.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System Information")

    st.write(
        "**LLM:** Gemini"
    )

    st.write(
        "**Embedding Model:** Gemini Embedding"
    )

    st.write(
        "**RAG:** Semantic Vector Retrieval"
    )

    st.write(
        "**Knowledge Base:** Hackathon Guidelines"
    )

    st.write(
        "**Interface:** Streamlit"
    )

    st.write(
        "**Language:** Python"
    )

    st.divider()

    st.info(
        "The system retrieves relevant knowledge "
        "before asking the LLM to evaluate the idea."
    )


# ============================================================
# IDEA INPUT
# ============================================================

st.subheader(
    "💡 Enter Your Hackathon Idea"
)

idea = st.text_area(
    "Hackathon idea",
    placeholder=(
        "Example: Our idea is an AI-based system "
        "that detects potholes using smartphone cameras "
        "and alerts users about damaged roads."
    ),
    height=180,
    label_visibility="collapsed"
)


# ============================================================
# EVALUATE BUTTON
# ============================================================

if st.button(
    "🔍 Evaluate Idea",
    type="primary",
    use_container_width=True
):

    if not idea.strip():

        st.warning(
            "Please enter a hackathon idea first."
        )

        st.stop()


    # --------------------------------------------------------
    # STEP 1: RAG RETRIEVAL
    # --------------------------------------------------------

    with st.spinner(
        "🔎 Retrieving relevant hackathon knowledge..."
    ):

        retrieved_knowledge = (
            retrieve_relevant_knowledge(
                idea,
                top_k=5
            )
        )


    # --------------------------------------------------------
    # STEP 2: GEMINI EVALUATION
    # --------------------------------------------------------

    with st.spinner(
        "🤖 Gemini is evaluating your idea..."
    ):

        evaluation = evaluate_idea(
            idea,
            retrieved_knowledge
        )


    # --------------------------------------------------------
    # STEP 3: EXTRACT SCORES
    # --------------------------------------------------------

    scores = extract_scores(
        evaluation
    )

    overall_score = calculate_overall_score(
        scores
    )

    decision = get_decision(
        overall_score
    )


    # ========================================================
    # OVERALL RESULT
    # ========================================================

    st.divider()

    st.subheader(
        "📊 Overall Evaluation"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Overall Score",
            f"{overall_score:.1f}/10"
        )

    with col2:

        st.metric(
            "Decision",
            decision
        )


    # ========================================================
    # CRITERIA SCORES
    # ========================================================

    st.subheader(
        "📈 Evaluation Criteria"
    )

    score_columns = st.columns(4)

    criteria_order = [
        "Problem Relevance",
        "Innovation",
        "Technical Feasibility",
        "Impact",
        "User Experience",
        "Scalability",
        "Sustainability"
    ]

    for i, criterion in enumerate(
        criteria_order
    ):

        with score_columns[i % 4]:

            if criterion in scores:

                st.metric(
                    criterion,
                    f"{scores[criterion]:.1f}/10"
                )

            else:

                st.metric(
                    criterion,
                    "N/A"
                )


    # ========================================================
    # AI EVALUATION
    # ========================================================

    st.divider()

    st.subheader(
        "🤖 AI Evaluation"
    )

    st.markdown(
        evaluation
    )


    # ========================================================
    # RAG EVIDENCE
    # ========================================================

    st.divider()

    st.subheader(
        "📚 RAG Evidence"
    )

    st.write(
        "The following knowledge was retrieved "
        "from the hackathon knowledge base."
    )


    for i, item in enumerate(
        retrieved_knowledge,
        start=1
    ):

        with st.expander(
            f"Evidence {i} — {item['source']}"
        ):

            st.write(
                item["text"]
            )

            st.caption(
                f"Similarity Score: "
                f"{item['score']:.3f}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Hackathon Idea Evaluator AI | "
    "RAG + LLM + Semantic Retrieval"
)
