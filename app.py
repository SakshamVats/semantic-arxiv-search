import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_model_and_db():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="db")
    collection = client.get_collection(name="arxiv_papers")
    return model, collection

MODEL, COLLECTION = load_model_and_db()

st.set_page_config(page_title="Semantic arXiv Search", page_icon="🔬", layout="wide")
st.header("Semantic arXiv Search Engine")

query = st.text_input("Enter your research query:", placeholder="e.g., Using transformers for time-series forecasting")

if query:
    query_embedding = MODEL.encode(query).tolist()

    with st.spinner("Searching for relevant papers..."):
        results = COLLECTION.query(
            query_embeddings=[query_embedding],
            n_results=10
        )

    st.subheader("Search Results")
    for i, metadata in enumerate(results['metadatas'][0]):
        st.markdown(f"**{i+1}. {metadata['title']}**")
        st.write(metadata['summary'])
        st.markdown(f"[Link to PDF]({metadata['pdf_url']})")
        st.divider()