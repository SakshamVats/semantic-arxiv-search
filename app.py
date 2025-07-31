__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

# --- Page Config ---
st.set_page_config(
    page_title="Semantic arXiv Search",
    page_icon="🔬",
    layout="wide"
)

# --- Add Custom Color ---
st.markdown("""
<style>
    /* This changes the primary color of sliders, buttons, etc. */
    :root {
        --primary-color: #4F8BF9;
    }
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# --- Model and DB Loading ---
@st.cache_resource
def load_model_and_db():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="db")
    collection = client.get_collection(name="arxiv_papers")
    return model, collection

MODEL, COLLECTION = load_model_and_db()

# --- Sidebar ---
with st.sidebar:
    st.header("About")
    st.write("""
    This app performs semantic search on a dataset of recent arXiv papers. 
    It uses a Sentence Transformer model to understand the meaning behind your query 
    and finds the most relevant research papers from a ChromaDB vector store.
    """)
    st.header("Tech Stack")
    st.markdown("""
    - Streamlit
    - Sentence-Transformers
    - ChromaDB
    - Pandas
    """)

# --- Main Page ---
st.title("🔬 Semantic arXiv Search Engine")

# --- Search Form ---
with st.form(key="search_form"):
    query = st.text_input(
        "Enter your research query:",
        placeholder="e.g., Using transformers for time-series forecasting"
    )
    submit_button = st.form_submit_button(label="Search")

# --- Search Results Logic ---
if submit_button and query:
    with st.spinner("Searching for relevant papers..."):
        results = COLLECTION.query(
            query_embeddings=[MODEL.encode(query).tolist()],
            n_results=10
        )

    st.subheader("Search Results")
    distances = results['distances'][0]

    min_dist = min(distances) if distances else 0
    max_dist = max(distances) if distances else 1
    normalized_distances = [(dist - min_dist) / (max_dist - min_dist) for dist in distances] if (max_dist - min_dist) > 0 else [0 for _ in distances]

    for i, (metadata, norm_dist) in enumerate(zip(results['metadatas'][0], normalized_distances)):
        relevance = max(0, 1 - norm_dist) * 100

        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.markdown(f"**{i+1}. {metadata['title']}**")
        with col2:
            st.link_button("PDF Link 🔗", metadata['pdf_url'], type="secondary", use_container_width=True)
        
        # Using st.metric for a nicer look
        st.metric(label="Relevance Score", value=f"{relevance:.1f}%")

        with st.expander("Show Summary"):
            st.write(metadata['summary'])