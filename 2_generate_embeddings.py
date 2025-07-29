# 2_generate_embeddings.py
import pandas as pd                                 
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import os

os.makedirs('data', exist_ok=True)

# Use CUDA if available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the pre-trained model all-MiniLM-L6-v2
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
df = pd.read_csv('data/arxiv_papers.csv')

# Generate embeddings for the summaries
embeddings = model.encode(df['summary'].tolist(), show_progress_bar=True, convert_to_numpy=True)

# Save the embeddings to a file
np.save('data/embeddings.npy', embeddings)
print(f"\nEmbeddings generated with shape {embeddings.shape} and saved.")