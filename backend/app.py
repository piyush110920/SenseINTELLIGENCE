from flask import Flask, request, jsonify
from flask_cors import CORS

# Import your chatbot and pdf processor modules
from chatbot import generate_answer
from pdf_processor import load_and_split_pdfs, build_vector_store

app = Flask(__name__)
CORS(app)  # Enable CORS so frontend can call API

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    response = generate_answer(user_input)
    return jsonify({"reply": response})

@app.route("/build", methods=["POST"])
def build_embeddings():
    # This triggers loading and processing PDFs to build the vector store
    texts, metadata = load_and_split_pdfs("data/documents")  # Adjust path as needed
    build_vector_store(texts, metadata)
    return jsonify({"message": "Embeddings built successfully."})

@app.route("/")
def home():
    return "SenseINTELLIGENCE backend is running."

if __name__ == "__main__":
    app.run(debug=True)
