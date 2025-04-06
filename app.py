import os
import streamlit as st
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer

# --- 1. Use environment variable or secret for API Key ---
genai.configure(api_key=st.secrets["AIzaSyASKTzSNuMbJMdZWr81Xuw2hS1Poe3acZo"])  # Set this in Streamlit Cloud secrets
model_gemini = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")

# --- 2. Use Hugging Face cache path (required by Streamlit) ---
os.environ["HF_HOME"] = "./hf_cache"  # persistent cache to avoid download errors
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- 3. Use a relative path for Chroma DB ---
CHROMA_PATH = "chroma_database"  # path inside repo, not /content
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="shl_data")

# --- 4. Search similar documents ---
def search_documents(query, top_k=10):
    query_embedding = model.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    return results['documents'][0]

# --- 5. Generate response from Gemini ---
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

# --- 6. Streamlit UI ---
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
