import os
import streamlit as st
import pandas as pd
import google.generativeai as genai
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import chromadb
from sentence_transformers import SentenceTransformer

# Configure Gemini API key
genai.configure(api_key="AIzaSyASKTzSNuMbJMdZWr81Xuw2hS1Poe3acZo")  # replace with st.secrets or env variable later
model_gemini = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load CSV
df = pd.read_csv("your_data.csv")  # <- Replace with your actual filename
documents = df["description"].tolist()  # or whichever column has text
ids = [str(i) for i in range(len(documents))]

# In-memory Chroma client
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="shl_data")

# Add documents to collection
collection.add(documents=documents, ids=ids)

# Search function
def search_documents(query, top_k=10):
    query_embedding = model.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    return results['documents'][0]

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
