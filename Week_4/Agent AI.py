import os
import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI
import requests
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
from typing import Any, List, Tuple

# Load .env keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

client = OpenAI()

# === Load PDFs & create index ===
@st.cache_resource
def load_pdfs_and_create_index(pdf_paths: List[str]) -> Tuple[List[str], faiss.IndexFlatL2, SentenceTransformer]:
    docs = []
    for path in pdf_paths:
        reader = PdfReader(path)
        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )
        docs.append(text)

    chunks: List[str] = []
    for doc in docs:
        for i in range(0, len(doc), 500):
            chunk = doc[i : i + 500]
            if chunk.strip():
                chunks.append(chunk)

    if not chunks:
        st.error("❌ No text extracted from PDFs. Please check the files.")
        st.stop()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return chunks, index, model

# === Retrieve ===
def retrieve(query: str, chunks: List[str], index: faiss.IndexFlatL2, model: SentenceTransformer, top_k: int = 3) -> List[str]:
    q_emb = model.encode(query)
    D, I = index.search(q_emb.reshape(1, -1), top_k)
    return [chunks[i] for i in I[0]]

# === Verifier ===
def is_answer_sufficient(query: str, answer: str) -> str:
    prompt = f"""
Question: {query}
Retrieved Answer: {answer}

Is the retrieved answer sufficient, accurate, and complete? Reply with YES or NO and a short reason.
"""
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content

# === SerpAPI Fallback ===
def serp_search(query: str) -> str:
    url = "https://serpapi.com/search.json"
    params = {
        "q": query,
        "api_key": SERPAPI_KEY
    }
    resp = requests.get(url, params=params)
    results = resp.json()
    organic = results.get("organic_results", [])
    snippets = []
    for r in organic:
        snippet = r.get("snippet") or r.get("title")
        if snippet:
            snippets.append(snippet)

    if snippets:
        return "\n".join(snippets)
    else:
        fallback_prompt = f"""
The web search for the question "{query}" returned no relevant results.
Please answer the question directly.
"""
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": fallback_prompt}]
        )
        return resp.choices[0].message.content

# === Final Orchestrator ===
def answer_query(query: str, chunks: List[str], index: faiss.IndexFlatL2, model: SentenceTransformer) -> str:
    retrieved = retrieve(query, chunks, index, model)
    combined = "\n\n".join(retrieved)
    verdict = is_answer_sufficient(query, combined)

    if "YES" in verdict.upper():
        return f"""
✅ **Answer:**  
{combined}

📝 **Verifier says:** {verdict}
"""
    else:
        web_ans = serp_search(query)
        return f"""
🔍 **Answer From Web:**  
{web_ans}

📝 **Verifier says:** {verdict}
"""

# === Streamlit UI ===
st.title("📚 Agentic RAG QA")

uploaded = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded:
    st.write("✅ Files uploaded.")
    os.makedirs("temp", exist_ok=True)
    pdf_paths = []
    for f in uploaded:
        path = os.path.join("temp", f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        pdf_paths.append(path)

    chunks, index, model = load_pdfs_and_create_index(pdf_paths)
    st.success(f"Indexed {len(chunks)} chunks from {len(pdf_paths)} PDFs.")

    query = st.text_input("🔎 Ask a question:")
    if query:
        with st.spinner("🤖 Thinking..."):
            answer = answer_query(query, chunks, index, model)
        st.markdown(answer)
