import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
from pdf_processor import load_vector_store
from config import GEMINI_API_KEY

# Configure Gemini API key once
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model (Gemini Pro)
model = genai.GenerativeModel(model_name='models/gemini-1.5-pro-latest')


# Initialize sentence transformer embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index and metadata ONCE, cache globally to avoid reloading every request
index, metadata = load_vector_store()

def retrieve_context(query, top_k=5):
    """
    Given a query string, embed it and retrieve top_k relevant chunks from FAISS index.
    Returns list of context strings.
    """
    query_vector = embed_model.encode([query])
    query_vector = np.array(query_vector).astype('float32')
    D, I = index.search(query_vector, top_k)
    context = []
    for idx in I[0]:
        if idx < len(metadata):
            if isinstance(metadata[idx], dict):
                source = metadata[idx].get("source", "Unknown source")
            else:
                source = metadata[idx]  # if it's a string, just use it directly
            context.append(f"{source}:\n...")
    return context


def generate_answer(query):
    """
    Given a user query, retrieve relevant context and generate an answer using Gemini.
    """
    context_chunks = retrieve_context(query)
    context_text = "\n\n".join(context_chunks)
    
    prompt = f"""You are senseINTELLIGENCE, a helpful assistant for Senselive Technology.
Use the following context from company PDFs to answer concisely:

Context:
{context_text}

Question: {query}
Answer:"""
    
    # Call Gemini's generate_content API
    response = model.generate_content(prompt)
    return response.text.strip()
