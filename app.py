import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import os
from dotenv import load_dotenv

try:
    from google import genai
except Exception:
    genai = None

load_dotenv()

st.set_page_config(page_title="AI Resume & Job Match RAG Assistant", layout="wide")

st.title("AI Resume & Job Match RAG Assistant")
st.write("Resume analysis + RAG + FAISS + Gemini LLM")

SKILLS = [
    "python", "machine learning", "deep learning", "generative ai", "llm", "rag",
    "embeddings", "vector database", "faiss", "langchain", "prompt engineering",
    "fastapi", "flask", "api", "docker", "linux", "git", "github",
    "numpy", "pandas", "sql", "postgresql", "tensorflow", "pytorch",
    "nlp", "transformers", "streamlit"
]

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def read_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.lower()

def read_knowledge_base():
    try:
        with open("knowledge_base.txt", "r", encoding="utf-8") as file:
            return file.read().lower()
    except FileNotFoundError:
        return "AI Engineer requires Python, ML, LLM, RAG, FAISS, embeddings, Docker, Linux, GitHub."

def chunk_text(text, chunk_size=100):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def create_faiss_index(chunks, model):
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve_context(query, chunks, index, model, k=3):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

def extract_skills(text):
    found = []
    for skill in SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text.lower()):
            found.append(skill)
    return sorted(list(set(found)))

def calculate_match_score(resume_skills, jd_skills):
    if not jd_skills:
        return 0
    matched = set(resume_skills).intersection(set(jd_skills))
    return round((len(matched) / len(set(jd_skills))) * 100, 2)

def smart_fallback_answer(query, retrieved_chunks):
    q = query.lower()
    context = " ".join(retrieved_chunks)

    if "rag" in q:
        return """
RAG means Retrieval-Augmented Generation.

Simple example:
If a user asks a question, the system first searches relevant information from documents or a knowledge base, then uses that information to generate the answer.

Flow:
User Query → Embedding → FAISS Search → Relevant Context → LLM → Final Answer
"""

    elif "llm" in q:
        return """
LLM means Large Language Model.

It is an AI model trained on large text data to understand and generate human-like answers.
Examples: ChatGPT, Gemini, Claude, LLaMA.
"""

    elif "faiss" in q:
        return "FAISS is used for fast vector similarity search in RAG systems."

    elif "embedding" in q:
        return "Embeddings convert text into numerical vectors so similar meaning can be searched."

    else:
        return context[:1200]

def generate_llm_answer(query, retrieved_chunks):
    api_key = os.getenv("GEMINI_API_KEY")
    context = "\n".join(retrieved_chunks)

    if genai is None:
        st.error("Gemini package not imported. Run: pip install google-genai")
        return smart_fallback_answer(query, retrieved_chunks)

    if not api_key:
        st.error("GEMINI_API_KEY not found in .env file.")
        return smart_fallback_answer(query, retrieved_chunks)

    prompt = f"""
You are an AI career assistant.

Use the given context and answer the question in simple words with example.

Context:
{context}

Question:
{query}

Answer:
"""

    try:
        st.info("Using Gemini LLM...")
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:
        st.error(f"Gemini Error: {e}")
        return smart_fallback_answer(query, retrieved_chunks)

tab1, tab2, tab3 = st.tabs([
    "Resume Match Analysis",
    "RAG Search Assistant",
    "Interview Prep"
])

with tab1:
    st.header("Resume and Job Description Match Analysis")

    resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
    jd_file = st.file_uploader("Upload Job Description PDF", type=["pdf"])

    if resume_file and jd_file:
        resume_text = read_pdf(resume_file)
        jd_text = read_pdf(jd_file)

        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(jd_text)

        matched_skills = sorted(list(set(resume_skills).intersection(set(jd_skills))))
        missing_skills = sorted(list(set(jd_skills) - set(resume_skills)))
        score = calculate_match_score(resume_skills, jd_skills)

        st.success("Resume and Job Description processed successfully!")

        col1, col2, col3 = st.columns(3)
        col1.metric("Match Score", f"{score}%")
        col2.metric("Matched Skills", len(matched_skills))
        col3.metric("Missing Skills", len(missing_skills))

        st.subheader("Matched Skills")
        st.write(", ".join(matched_skills) if matched_skills else "No matching skills found.")

        st.subheader("Missing Skills")
        st.write(", ".join(missing_skills) if missing_skills else "No missing skills found.")

with tab2:
    st.header("RAG Search Assistant")

    source_option = st.radio("Choose knowledge source", ["Knowledge Base", "Uploaded Resume + JD"])
    query = st.text_input("Ask anything about AI Engineer preparation or uploaded documents")

    embedding_model = load_embedding_model()

    if source_option == "Knowledge Base":
        source_text = read_knowledge_base()
    else:
        if "resume_text" in locals() and "jd_text" in locals():
            source_text = resume_text + "\n" + jd_text
        else:
            source_text = ""

    if st.button("Search"):
        if query.strip() == "":
            st.warning("Please enter a query.")
        elif source_text.strip() == "":
            st.warning("Please upload Resume and Job Description first.")
        else:
            chunks = chunk_text(source_text)
            index = create_faiss_index(chunks, embedding_model)
            retrieved_chunks = retrieve_context(query, chunks, index, embedding_model)

            answer = generate_llm_answer(query, retrieved_chunks)

            st.subheader("AI Answer")
            st.write(answer)

            with st.expander("Retrieved Context"):
                for chunk in retrieved_chunks:
                    st.write(chunk)

with tab3:
    st.header("AI Engineer Interview Prep")

    st.subheader("Project Explanation")
    st.write("""
I built an AI Resume & Job Match RAG Assistant using Python, Sentence Transformers, FAISS, and Gemini LLM.
It analyzes resumes and job descriptions, calculates match score, detects missing skills,
retrieves relevant context using FAISS, and generates final answers using Gemini.
""")

    st.subheader("Architecture")
    st.code("""
Resume PDF + Job Description PDF / Knowledge Base
        ↓
Text Extraction
        ↓
Text Chunking
        ↓
Sentence Transformer Embeddings
        ↓
FAISS Vector Search
        ↓
Retrieved Context
        ↓
Gemini LLM
        ↓
Final Answer
""")