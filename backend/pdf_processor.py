import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Initialize the sentence transformer model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_and_split_pdfs(pdf_dir):
    """
    Load PDFs from directory, extract text by page,
    and return a list of text chunks and their metadata (source filenames).
    """
    texts = []
    metadata = []
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            doc_path = os.path.join(pdf_dir, filename)
            try:
                doc = fitz.open(doc_path)
                for page_num, page in enumerate(doc):
                    text = page.get_text().strip()
                    if text:
                        texts.append(text)
                        metadata.append({
                            "source": filename,
                            "page": page_num + 1  # Optional: store page number
                        })
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
    return texts, metadata

def build_vector_store(texts, metadata):
    """
    Create embeddings for text chunks, build FAISS index,
    and save both index and metadata to disk.
    """
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # L2 distance index
    
    # Add embeddings to FAISS index
    index.add(np.array(embeddings).astype('float32'))
    
    # Ensure directory exists
    os.makedirs("vectorstore", exist_ok=True)
    
    # Save metadata (list of dicts)
    with open("vectorstore/texts.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    # Save FAISS index
    faiss.write_index(index, "vectorstore/index.faiss")

def load_vector_store():
    """
    Load FAISS index and metadata from disk.
    Returns:
        index: FAISS index
        metadata: list of dicts with source info
    """
    index_path = "vectorstore/index.faiss"
    metadata_path = "vectorstore/texts.pkl"
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError("Vector store files not found. Please run build_vector_store first.")
    
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata
