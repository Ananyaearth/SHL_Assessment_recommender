import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Configure Gemini API key (from Streamlit secrets)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAISS index and documents
faiss_index = faiss.read_index("faiss_index.bin")
with open("documents.pkl", "rb") as f:
    index_to_doc = pickle.load(f)

# Search function
def search_documents(query, top_k=10):
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = faiss_index.search(query_embedding, top_k)
    return [index_to_doc[i] for i in indices[0]]

# Prompt + Gemini generation
def ask_rag_question(query):
    context_chunks = search_documents(query)
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are an expert in HR assessments. Based on the context below, identify all assessments relevant to the user's request.

### Instructions:
- Present your answer in a markdown table with **these columns**:
  | Assessment Name (with link) | Remote Testing Support | Adaptive/IRT Support | Duration & Test Type | Why Recommended/Not Recommended |
- Carefully review **all the provided context chunks** and extract multiple assessments if applicable.
- **Do not make up any data** — only use what's in the context.
- **Do not hallucinate or assume** Remote Testing or Adaptive/IRT support if it is not explicitly mentioned.

### Context:
{context}

### Question:
{query}

### Answer:
"""
    response = genai.generate_content(prompt)
    return response.text

# Streamlit UI
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")
st.title("🔍 SHL Assessment Recommender (RAG + FAISS + Gemini)")

query = st.text_input("Enter your hiring requirement (e.g., Java developer with business skills)...")

if query:
    with st.spinner("Finding the most suitable assessments..."):
        result = ask_rag_question(query)
        st.markdown(result, unsafe_allow_html=True)
