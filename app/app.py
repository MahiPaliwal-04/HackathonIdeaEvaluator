import streamlit as st
from google import genai
from google.genai import errors
from pathlib import Path
import numpy as np
import re
import time


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Hackathon Idea Evaluator AI",
    page_icon="💡",
    layout="wide"
)

# Primary + fallback models
GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash"
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# =========================================================
# GEMINI CLIENT
# =========================================================

if "GEMINI_API_KEY" not in st.secrets:
    st.error("GEMINI_API_KEY is not configured in Streamlit Secrets.")
    st.stop()

API_KEY = st.secrets["GEMINI_API_KEY"]

client = genai.Client(api_key=API_KEY)


# =========================================================
# LOAD KNOWLEDGE BASE
# =========================================================

def load_knowledge_base():
    documents = []

    if not DATA_DIR.exists():
        return documents

    for file_path in DATA_DIR.glob("*.txt"):
        try:
            text = file_path.read_text(encoding="utf-8")

            if text.strip():
                documents.append({
                    "name": file_path.name,
                    "text": text
                })

        except Exception as e:
            st.warning(f"Could not read {file_path.name}: {e}")

    return documents


knowledge_base = load_knowledge_base()


# =========================================================
# TEXT PROCESSING
# =========================================================

def tokenize(text):
    text = text.lower()

    words = re.findall(r"\b[a-zA-Z0-9]+\b", text)

    stop_words = {
        "the", "a", "an", "and", "or", "is", "are",
        "to", "of", "in", "on", "for", "with", "using",
        "this", "that", "it", "be", "can", "as", "by",
        "from", "at", "into", "will", "their", "they",
        "should", "have", "has", "than", "also"
    }

    return set(
        word for word in words
        if word not in stop_words and len(word) > 2
    )


# =========================================================
# RAG RETRIEVAL
# =========================================================

def calculate_similarity(query, document):
    query_words = tokenize(query)
    document_words = tokenize(document)

    if not query_words or not document_words:
        return 0.0

    common_words = query_words.intersection(document_words)

    return len(common_words) / len(query_words)


def retrieve_knowledge(query, top_k=5):

    results = []

    for document in knowledge_base:

        score = calculate_similarity(
            query,
            document["text"]
        )

        results.append({
            "name": document["name"],
            "text": document["text"],
            "score": score
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# =========================================================
# GEMINI GENERATION WITH RETRY + FALLBACK
# =========================================================

def generate_with_fallback(prompt):

    last_error = None

    for model in GEMINI_MODELS:

        for attempt in range(3):

            try:

                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )

                if response and response.text:
                    return response.text, model

            except Exception as e:

                last_error = e

                error_text = str(e)

                # Retry temporary server/rate-limit errors
                if (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                    or "500" in error_text
                    or "INTERNAL" in error_text
                ):

                    wait_time = 2 ** attempt

                    time.sleep(wait_time)

                    continue

                # For other errors, move to next model
                break

    raise RuntimeError(
        f"Gemini models are currently unavailable. "
        f"Last error: {last_error}"
    )


# =========================================================
# IDEA EVALUATION
# =========================================================

def evaluate_idea(idea, retrieved_knowledge):

    context_parts = []

    for item in retrieved_knowledge:

        context_parts.append(
            f"""
SOURCE: {item['name']}

{item['text']}
"""
        )

    retrieved_context = "\n".join(context_parts)

    prompt = f"""
You are an expert hackathon project evaluator.

Evaluate the following hackathon idea using the provided
knowledge retrieved from a RAG knowledge base.

HACKATHON IDEA:
{idea}

RETRIEVED KNOWLEDGE:
{retrieved_context}

Evaluate the idea on these seven criteria:

1. Problem Relevance
2. Innovation
3. Technical Feasibility
4. Impact
5. User Experience
6. Scalability
7. Sustainability

Give each score from 1 to 10.

Then provide:

- Overall Score
- Final Verdict
- Strengths
- Weaknesses
- Suggestions for Improvement
- Technical Risks
- Target Users
- Why the idea can perform well in a hackathon

IMPORTANT:
Use the retrieved knowledge as supporting evidence.
Do not invent facts from the knowledge base.
Give practical and realistic evaluation.

Return the answer in exactly this format:

Problem Relevance: X/10
Innovation: X/10
Technical Feasibility: X/10
Impact: X/10
User Experience: X/10
Scalability: X/10
Sustainability: X/10

Overall Score: X/10

Final Verdict:
[Strong / Good / Average / Weak]

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

Technical Risks:
- point
- point

Target Users:
- point
- point

Why It Can Perform Well:
- point
- point
"""

    result, used_model = generate_with_fallback(prompt)

    return result, used_model


# =========================================================
# SCORE EXTRACTION
# =========================================================

def extract_score(text, criterion):

    pattern = rf"{re.escape(criterion)}\s*:\s*(\d+)\s*/\s*10"

    match = re.search(
        pattern,
        text,
        re.IGNORECASE
    )

    if match:
        score = int(match.group(1))

        if 0 <= score <= 10:
            return score

    return None


# =========================================================
# UI
# =========================================================

st.title("💡 Hackathon Idea Evaluator AI")

st.markdown(
    """
### RAG + LLM Based Hackathon Idea Evaluation

Enter your hackathon idea and the system will evaluate it
using relevant hackathon judging criteria, successful ideas,
and technical feasibility guidelines.
"""
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("System Information")

    st.write("**LLM:** Gemini Flash")

    st.write("**RAG:** Knowledge Base Retrieval")

    st.write("**Framework:** Streamlit")

    st.write("**Knowledge Sources:**")

    if knowledge_base:

        for item in knowledge_base:
            st.write(f"• {item['name']}")

    else:

        st.warning("No knowledge files found.")


# =========================================================
# INPUT
# =========================================================

st.subheader("Enter Your Hackathon Idea")

idea = st.text_area(
    "Hackathon Idea",
    height=180,
    placeholder=(
        "Example: An AI-based system that detects potholes "
        "using smartphone cameras and automatically reports "
        "their locations to municipal authorities."
    )
)


# =========================================================
# EVALUATE BUTTON
# =========================================================

if st.button(
    "🚀 Evaluate Idea",
    type="primary",
    use_container_width=True
):

    if not idea.strip():

        st.warning(
            "Please enter a hackathon idea first."
        )

    elif not knowledge_base:

        st.error(
            "Knowledge base is empty. "
            "Please check the data folder."
        )

    else:

        # -------------------------------------------------
        # RAG RETRIEVAL
        # -------------------------------------------------

        with st.spinner(
            "Retrieving relevant hackathon knowledge..."
        ):

            retrieved_knowledge = retrieve_knowledge(
                idea,
                top_k=5
            )

        # -------------------------------------------------
        # DISPLAY RETRIEVED KNOWLEDGE
        # -------------------------------------------------

        with st.expander(
            "📚 RAG Retrieved Knowledge",
            expanded=False
        ):

            for item in retrieved_knowledge:

                st.markdown(
                    f"### {item['name']}"
                )

                st.write(
                    f"Retrieval Score: "
                    f"{item['score']:.2f}"
                )

                st.write(
                    item["text"]
                )

                st.divider()

        # -------------------------------------------------
        # GEMINI EVALUATION
        # -------------------------------------------------

        with st.spinner(
            "Gemini is evaluating your idea..."
        ):

            try:

                evaluation, used_model = evaluate_idea(
                    idea,
                    retrieved_knowledge
                )

            except Exception as e:

                st.error(
                    "Gemini is temporarily unavailable."
                )

                st.info(
                    "Please try again after a short time."
                )

                st.code(
                    str(e)
                )

                st.stop()

        # -------------------------------------------------
        # SUCCESS MESSAGE
        # -------------------------------------------------

        st.success(
            f"Evaluation completed using {used_model}"
        )

        # -------------------------------------------------
        # DISPLAY EVALUATION
        # -------------------------------------------------

        st.subheader("📊 Evaluation Result")

        st.markdown(evaluation)

        # -------------------------------------------------
        # SCORE CARDS
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

            score = extract_score(
                evaluation,
                criterion
            )

            if score is not None:
                scores[criterion] = score

        if scores:

            st.subheader("📈 Score Summary")

            columns = st.columns(4)

            index = 0

            for criterion, score in scores.items():

                with columns[index % 4]:

                    st.metric(
                        criterion,
                        f"{score}/10"
                    )

                index += 1

            overall = sum(scores.values()) / len(scores)

            st.metric(
                "Overall Average",
                f"{overall:.1f}/10"
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Hackathon Idea Evaluator AI | "
    "RAG + LLM + Streamlit"
)