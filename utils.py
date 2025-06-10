# utils.py

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model and data once
model = SentenceTransformer("all-MiniLM-L6-v2")
INDEX_PATH = "vector_store/index.faiss"
TEXTS_PATH = "vector_store/texts.pkl"

# Load index and texts
index = faiss.read_index(INDEX_PATH)
with open(TEXTS_PATH, "rb") as f:
    texts = pickle.load(f)

def search_faiss(query, top_k=5):
    """Search the FAISS index for relevant chunks."""
    query_embedding = model.encode([query])[0]
    D, I = index.search(np.array([query_embedding]), top_k)
    results = [texts[i] for i in I[0] if i != -1]
    return results
