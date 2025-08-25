import google.generativeai as genai

def generate_embedding(code_snippet):
    """Generate embedding for a code snippet."""
    response = genai.embed_content(
        model="models/embedding-001",
        content=code_snippet,
        task_type="retrieval_document"
    )
    estimated_tokens = len(code_snippet.split())
    print(f"Estimated tokens for embedding: {estimated_tokens}")
    return response['embedding']