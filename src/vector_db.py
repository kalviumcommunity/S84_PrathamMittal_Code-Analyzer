import faiss
import numpy as np
from embeddings import generate_embedding

def create_vector_db(code_snippets):
    """Create a FAISS index with embeddings of code snippets."""
    dimension = 768  # Gemini embedding-001 dimension
    index = faiss.IndexFlatL2(dimension)
    embeddings = [generate_embedding(code) for code in code_snippets]
    embeddings = np.array(embeddings, dtype=np.float32)
    index.add(embeddings)
    return index, embeddings