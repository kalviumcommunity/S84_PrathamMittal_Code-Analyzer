import google.generativeai as genai
import numpy as np

def log_tokens(response, system_prompt, user_prompt):
    """Log the number of tokens used in a Gemini API call."""
    total_tokens = len(system_prompt.split()) + len(user_prompt.split()) + len(response.text.split())
    print(f"Estimated tokens used in API call: {total_tokens}")
    return total_tokens

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def l2_distance(vec1, vec2):
    """Calculate L2 (Euclidean) distance between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.linalg.norm(vec1 - vec2)

def dot_product_similarity(vec1, vec2):
    """Calculate dot product similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2)