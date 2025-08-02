# The sqlite3 monkey-patch for deployment
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import os
import requests
import zipfile

# --- Startup Script to Fetch Database ---
DB_DIR = "db"
DB_ZIP_PATH = "db.zip"
DB_DOWNLOAD_URL = "https://github.com/SakshamVats/semantic-arxiv-search/releases/download/v1.0/db.zip" 

if not os.path.exists(DB_DIR):
    with st.spinner("Downloading database... This may take a moment."):
        r = requests.get(DB_DOWNLOAD_URL)
        with open(DB_ZIP_PATH, "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(DB_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(".")
    os.remove(DB_ZIP_PATH)
    st.success("Database downloaded and ready!")
    st.rerun()

# --- Main App ---
st.set_page_config(page_title="Semantic arXiv Search", page_icon="🔬", layout="wide")

@st.cache_resource
def load_model_and_db():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name="arxiv_papers")
    return model, collection

if os.path.exists(DB_DIR):
    MODEL, COLLECTION = load_model_and_db()

    # --- Updated Sidebar ---
    with st.sidebar:
        st.header("About This Project")
        st.write("""
        This app overcomes the limitations of traditional keyword search. Instead of just matching words, it uses a Sentence Transformer model to understand the *semantic meaning* of your query.

        It finds research papers whose summaries are contextually most similar to what you're asking for, providing more relevant and nuanced results.
        """)
        st.write("---")
        st.header("Tech Stack")
        st.markdown("""
        - **Model:** `all-MiniLM-L6-v2`
        - **Database:** ChromaDB (Vector Store)
        - **Framework:** Streamlit
        - **Core Libraries:** PyTorch, Pandas
        """)

    # --- Main Page ---
    st.title("🔬 Semantic arXiv Search Engine")

    with st.form(key="search_form"):
        query = st.text_input("Enter your research query")
        submit_button = st.form_submit_button(label="Search")

    if submit_button and query:
        with st.spinner("Searching..."):
            results = COLLECTION.query(
                query_embeddings=[MODEL.encode(query).tolist()],
                n_results=10
            )
        
        st.subheader("Search Results")
        for i, metadata in enumerate(results['metadatas'][0]):
            st.markdown(f"**{i + 1}. {metadata['title']}**")
            with st.expander("Show Summary"):
                st.write(metadata['summary'])
            st.link_button("PDF Link 🔗", metadata['pdf_url'])
            st.divider()
