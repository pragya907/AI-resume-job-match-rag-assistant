# AI Resume & Job Match RAG Assistant

## Overview

AI-powered Resume and Job Description Matching System built using Python, Streamlit, FAISS, Sentence Transformers, and Gemini LLM integration.

## Features

* Resume vs Job Description Match Score
* Missing Skills Detection
* RAG (Retrieval-Augmented Generation)
* FAISS Vector Search
* Sentence Transformer Embeddings
* AI Interview Preparation Assistant
* Gemini LLM Integration
* Knowledge Base Search

## Tech Stack

* Python
* Streamlit
* Sentence Transformers
* FAISS
* Gemini API
* NumPy
* PyPDF

## Architecture

User Query
↓
Embedding Model
↓
FAISS Vector Search
↓
Relevant Context Retrieval
↓
Gemini LLM
↓
Final Answer

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Future Enhancements

* Multi-PDF Support
* Resume Improvement Suggestions
* Job Recommendation Engine
* Deployment on Render
* PostgreSQL Vector Database
