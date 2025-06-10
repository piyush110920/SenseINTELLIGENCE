# index_data.py
import os
import fitz  # PyMuPDF
import pickle
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
INDEX_PATH = "vector_store/index.faiss"
TEXTS_PATH = "vector_store/texts.pkl"

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = []

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".pdf"):
        doc = fitz.open(os.path.join(DATA_DIR, filename))
        for page in doc:
            text = page.get_text().strip()
            if text:
                texts.append(text)

embeddings = model.encode(texts)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

os.makedirs("vector_store", exist_ok=True)
faiss.write_index(index, INDEX_PATH)

with open(TEXTS_PATH, "wb") as f:
    pickle.dump(texts, f)

print("✅ FAISS index and texts saved.")
