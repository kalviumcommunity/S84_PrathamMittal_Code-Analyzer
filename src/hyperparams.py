import google.generativeai as genai
import json
from prompts import get_system_prompt, get_zero_shot_prompt, get_one_shot_prompt, get_multi_shot_prompt, get_dynamic_prompt, get_cot_prompt
from utils import log_tokens

def format_code(code):
    """Mock function to format code (e.g., using a linter)."""
    return code.replace("\n    ", "\n\t")  # Simple indentation fix

def analyze_with_temperature(code_snippet, prompt_type="zero_shot", temperature=0.7):
    """Analyze code with specified temperature."""
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
    
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={"temperature": temperature}
    )
    response = model.generate_content([system_prompt, prompt])
    
    log_tokens(response, system_prompt, prompt)
    return response.text

def analyze_with_top_p(code_snippet, prompt_type="zero_shot", top_p=0.9):
    """Analyze code with specified top_p."""
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
    
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={"top_p": top_p}
    )
    response = model.generate_content([system_prompt, prompt])
    
    log_tokens(response, system_prompt, prompt)
    return response.text

def analyze_with_top_k(code_snippet, prompt_type="zero_shot", top_k=40):
    """Simulate top_k sampling using low temperature for deterministic output."""
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
    
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={"temperature": 0.1}
    )
    response = model.generate_content([system_prompt, prompt])
    
    log_tokens(response, system_prompt, prompt)
    return response.text

def analyze_with_stop_sequence(code_snippet, prompt_type="zero_shot", stop_sequence="###"):
    """Analyze code with specified stop sequence."""
    prompt_functions = {
        "zero_shot": get_zero_shot_prompt,
        "one_shot": get_one_shot_prompt,
        "multi_shot": get_multi_shot_prompt,
        "dynamic": get_dynamic_prompt,
        "cot": get_cot_prompt
    }
    
    if prompt_type not in prompt_functions:
        raise ValueError(f"Invalid prompt type: {prompt_type}")
    
    prompt = prompt_functions[prompt_type](code_snippet) + "\n###"
    system_prompt = get_system_prompt()
    
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={"stop_sequences": [stop_sequence]}
    )
    response = model.generate_content([system_prompt, prompt])
    
    log_tokens(response, system_prompt, prompt)
    return response.text

def analyze_with_structured_output(code_snippet, prompt_type="zero_shot"):
    """Analyze code with JSON-structured output."""
    prompt_functions = {
        "zero_shot": get_zero_shot_prompt,
        "one_shot": get_one_shot_prompt,
        "multi_shot": get_multi_shot_prompt,
        "dynamic": get_dynamic_prompt,
        "cot": get_cot_prompt
    }
    
    if prompt_type not in prompt_functions:
        raise ValueError(f"Invalid prompt type: {prompt_type}")
    
    prompt = prompt_functions[prompt_type](code_snippet) + "\nReturn the response in JSON format with keys: errors, fixes, summary."
    system_prompt = get_system_prompt()
    
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={"response_mime_type": "application/json"}
    )
    response = model.generate_content([system_prompt, prompt])
    
    log_tokens(response, system_prompt, prompt)
    return json.loads(response.text)

def analyze_with_function_calling(code_snippet, prompt_type="zero_shot"):
    """Analyze code with function calling to format code."""
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
    
    formatted_code = format_code(code_snippet)
    prompt = f"Analyze this formatted code:\n{formatted_code}"
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([system_prompt, prompt])
    
    log_tokens(response, system_prompt, prompt)
    return response.text