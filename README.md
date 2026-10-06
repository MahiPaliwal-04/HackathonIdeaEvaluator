### 🚀 Live Demo

[Open Hackathon Idea Evaluator](https://mahipaliwal-04-hackathonideaevaluator-appapp-cz5u7d.streamlit.app/)
# 🏆 Hackathon Idea Evaluator AI

An LLM and RAG-based AI system that evaluates hackathon ideas and provides scores, strengths, weaknesses, and practical suggestions.

## 📌 Project Overview

Hackathon participants often have ideas but find it difficult to understand whether their idea is relevant, innovative, technically feasible, and impactful.

Hackathon Idea Evaluator AI solves this problem by combining Retrieval-Augmented Generation (RAG) with a Large Language Model (LLM).

The system retrieves relevant information from a hackathon knowledge base and uses Llama 3.2 to evaluate the submitted idea.

## 🎯 Objectives

- Evaluate hackathon ideas using multiple criteria.
- Retrieve relevant hackathon knowledge using RAG.
- Provide realistic scores and feedback.
- Identify strengths and weaknesses.
- Suggest improvements for the submitted idea.
- Help participants improve their ideas before submission.

## ✨ Features

- 💡 Hackathon idea input
- 🔎 RAG-based knowledge retrieval
- 🤖 LLM-based evaluation
- 📊 Overall score
- 📈 Detailed evaluation scores
- ✅ Strengths identification
- ⚠️ Weakness identification
- 💭 Improvement suggestions
- 📚 RAG evidence display
- 🖥️ Streamlit web interface

## 🧠 Evaluation Criteria

The system evaluates ideas using:

1. Problem Relevance
2. Innovation
3. Technical Feasibility
4. Impact
5. User Experience
6. Scalability
7. Sustainability

## 🔄 How It Works

```text
User enters Hackathon Idea
          ↓
Generate Text Embedding
          ↓
Qdrant Vector Database
          ↓
Retrieve Relevant Knowledge
          ↓
Llama 3.2 LLM
          ↓
Evaluate Hackathon Idea
          ↓
Scores + Strengths + Weaknesses
          ↓
Suggestions + Final Verdict
