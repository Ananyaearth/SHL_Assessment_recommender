import os
import streamlit as st
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Configure Gemini API key
genai.configure(api_key="AIzaSyASKTzSNuMbJMdZWr81Xuw2hS1Poe3acZo")  # replace with st.secrets later
model_gemini = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load CSV
df = pd.read_csv("shl_catalog_detailed.csv")  # ensure full filename with .csv
documents = df["description"].tolist()
ids = [str(i) for i in range(len(documents))]

# Generate embeddings
embeddings = model.encode(documents, convert_to_numpy=True).astype("float32")

# Initialize FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Search function using FAISS
def search_documents(query, top_k=10):
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(query_embedding, top_k)
    return [documents[i] for i in I[0]]

# Gemini response generator
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
    response = model_gemini.generate_content(prompt)
    return response.text

# Streamlit UI
st.set_page_config(page_title="SHL Catalog Assistant", layout="wide")
st.title("🧠 SHL Assessment Recommendation System")

query = st.text_area("Enter your hiring requirement or question:", height=150)

if st.button("Search & Recommend"):
    if not query.strip():
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Processing..."):
            output = ask_rag_question(query)
            st.markdown(output)
