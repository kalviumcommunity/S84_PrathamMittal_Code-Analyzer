import os
from dotenv import load_dotenv
import google.generativeai as genai
from prompts import get_system_prompt, get_zero_shot_prompt, get_one_shot_prompt, get_multi_shot_prompt, get_dynamic_prompt, get_cot_prompt
from evaluation import run_evaluation
from utils import log_tokens, cosine_similarity, l2_distance, dot_product_similarity
from hyperparams import analyze_with_temperature, analyze_with_top_p, analyze_with_top_k, analyze_with_stop_sequence, analyze_with_structured_output, analyze_with_function_calling
from embeddings import generate_embedding
from vector_db import create_vector_db

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def analyze_code(code_snippet, prompt_type="zero_shot"):
    """Analyze code using the specified prompt type."""
    prompt_functions = {
        "zero_shot": get_zero_shot_prompt,
        "one_shot": get_one_shot_prompt,
        "multi_shot": get_multi_shot_prompt,
        "dynamic": get_dynamic_prompt,
        "cot": get_cot_prompt
    }
    
    if prompt_type not in prompt_functions:
        raise ValueError(f"Invalid prompt type: {prompt_type}")
    
    prompt = prompt_functions[prompt_type](code_snippet)
    system_prompt = get_system_prompt()
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([system_prompt, prompt])
    
    log_tokens(response, system_prompt, prompt)
    return response.text

if __name__ == "__main__":
    # Sample code snippet with an error
    sample_code = """
    def factorial(n):
        if n == 0:
            return 1
        return n * factorial(n - 1
    """
    
    # Test different prompt types (Tasks 3-7)
    prompt_types = ["zero_shot", "one_shot", "multi_shot", "dynamic", "cot"]
    for prompt_type in prompt_types:
        print(f"\n{prompt_type.replace('_', ' ').title()} Result:")
        print(analyze_code(sample_code, prompt_type))
    
    # Test temperature (Task 10)
    print("\nTemperature (0.7) Result:")
    print(analyze_with_temperature(sample_code, "zero_shot", temperature=0.7))
    
    # Test top_p (Task 11)
    print("\nTop P (0.9) Result:")
    print(analyze_with_top_p(sample_code, "zero_shot", top_p=0.9))
    
    # Test top_k (Task 12)
    print("\nTop K (40) Result:")
    print(analyze_with_top_k(sample_code, "zero_shot", top_k=40))
    
    # Test stop sequence (Task 13)
    print("\nStop Sequence (###) Result:")
    print(analyze_with_stop_sequence(sample_code, "zero_shot", stop_sequence="###"))
    
    # Test structured output (Task 14)
    print("\nStructured Output (JSON) Result:")
    print(analyze_with_structured_output(sample_code, "zero_shot"))
    
    # Test function calling (Task 15)
    print("\nFunction Calling Result:")
    print(analyze_with_function_calling(sample_code, "zero_shot"))
    
    # Test embeddings (Task 16)
    print("\nEmbedding Result:")
    print(generate_embedding(sample_code)[:10])  # Show first 10 values
    
    # Test vector database (Task 17)
    print("\nVector Database Result:")
    snippets = [sample_code, "def add(a, b):\n    return a + b"]
    index, embeddings = create_vector_db(snippets)
    print(f"FAISS index with {index.ntotal} vectors")
    
    # Test similarity functions (Task 18)
    print("\nSimilarity Results:")
    emb1 = generate_embedding(sample_code)
    emb2 = generate_embedding("def add(a, b):\n    return a + b")
    print(f"Cosine Similarity: {cosine_similarity(emb1, emb2)}")
    print(f"L2 Distance: {l2_distance(emb1, emb2)}")
    print(f"Dot Product Similarity: {dot_product_similarity(emb1, emb2)}")
    
    # Run evaluation pipeline (Task 8)
    print("\nRunning evaluation pipeline...")
    run_evaluation()