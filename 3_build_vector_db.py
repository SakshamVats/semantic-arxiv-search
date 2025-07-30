import chromadb
import numpy as np
import pandas as pd
from tqdm import tqdm

client = chromadb.PersistentClient(path="db")
collection = client.get_or_create_collection(name="arxiv_papers")

df = pd.read_csv('data/arxiv_papers.csv')
embeddings = np.load('data/embeddings.npy')

ids = [str(i) for i in df.index]
metadatas = df[['title', 'summary', 'pdf_url']].to_dict(orient='records')

batch_size = 500
for i in tqdm(range(0, len(ids), batch_size), desc="Adding embeddings to ChromaDB"):
    collection.add(
        embeddings=embeddings[i:i+batch_size].tolist(),
        metadatas=metadatas[i:i+batch_size],
        ids=ids[i:i+batch_size]
    )

print(f"\nSuccessfully built vector DB with {collection.count()} entries.")