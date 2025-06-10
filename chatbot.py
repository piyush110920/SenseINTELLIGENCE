import ollama
from utils import search_faiss  # Your vector search function
#from clean_utils import clean_text  # If you use it
import re

# ✅ Option 1: Static answers override
OVERRIDE_ANSWERS = {
    "contact": "You can contact Senselive Technology at support@senselive.com or call +91 9604070622.",
    "phone number": "Senselive Technology's phone number is +91 9604070622.",
    "email": "You can email Senselive at support@senselive.com.",
    "address": "Senselive Technology is located at 268, BHAMTEE COLONEY, NAGPUR, NAGPUR, Maharashtra, India - 440022.",
    "website": "Our website is info@senselive.io"
}

# ✅ Option 2: Trusted info injection
trusted_info = """
Company: Senselive Technology
Email: support@senselive.com
Phone: +91 9604070622
Address: 268, BHAMTEE COLONEY, NAGPUR, NAGPUR, Maharashtra, India - 440022.
Website: info@senselive.io
"""

def check_for_override(query):
    for key in OVERRIDE_ANSWERS:
        if re.search(rf'\b{key}\b', query.lower()):
            return OVERRIDE_ANSWERS[key]
    return None

def format_prompt(query, context):
    return f"""You are SENSEintelligence, a helpful AI assistant for Senselive Technology.

IMPORTANT: Only refer to the official company info when asked about contact details.

Official Company Info:
{trusted_info}

Relevant Information:
{context}

User Query:
{query}

Answer:"""

def chatbot_response(query):
    # Check if question is one of the static overrides
    override = check_for_override(query)
    if override:
        return override

    # If not, perform vector search
    relevant_chunks = search_faiss(query)
    context = "\n".join(relevant_chunks)

    # Build final prompt
    prompt = format_prompt(query, context)

    # Call TinyLLaMA or your current LLM
    response = ollama.chat(
        model="tinyllama",  # or whatever model you're using
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response['message']['content']
