import google.generativeai as genai
import json
from prompts import get_zero_shot_prompt
from utils import log_tokens

def get_judge_prompt(model_output, expected_output):
    """Return judge prompt to compare model output with expected results."""
    return f"""
    Compare the model output with the expected output and assign a score (0-10) based on:
    - Accuracy: Correct identification of errors and fixes.
    - Completeness: All errors and functionality covered.
    - Clarity: Clear and concise response.

    Model Output:
    {model_output}

    Expected Output:
    {expected_output}

    Provide a score and a brief explanation.
    """

def run_evaluation():
    """Run evaluation pipeline with a dataset of 5 samples."""
    dataset = [
        {
            "code": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1",
            "expected": "Errors: Missing closing parenthesis in recursive call.\nFixes: Change `factorial(n - 1` to `factorial(n - 1)`.\nSummary: Computes factorial of n using recursion."
        },
        {
            "code": "for i in range(5)\n    print(i)",
            "expected": "Errors: Missing colon after range statement.\nFixes: Add colon after `range(5):`.\nSummary: Prints numbers 0 to 4."
        },
        {
            "code": "def add(a, b):\n    return a + b",
            "expected": "Errors: None.\nFixes: None.\nSummary: Adds two numbers and returns their sum."
        },
        {
            "code": "x = [1, 2, 3]\nfor i in x\n    print(i)",
            "expected": "Errors: Missing colon after for loop condition.\nFixes: Add colon after `for i in x:`.\nSummary: Prints each element in the list x."
        },
        {
            "code": "def divide(a, b):\n    return a / b",
            "expected": "Errors: None.\nFixes: None.\nSummary: Divides a by b and returns the result."
        }
    ]

    results = []
    model = genai.GenerativeModel("gemini-1.5-flash")
    for sample in dataset:
        system_prompt = "You are an expert code analyst."
        prompt = get_zero_shot_prompt(sample['code'])
        model_output = model.generate_content([system_prompt, prompt])
        model_output_text = model_output.text
        log_tokens(model_output, system_prompt, prompt)
        
        judge_prompt = get_judge_prompt(model_output_text, sample['expected'])
        judge_response = model.generate_content([system_prompt, judge_prompt])
        judge_output = judge_response.text
        log_tokens(judge_response, system_prompt, judge_prompt)
        
        results.append({
            "code": sample['code'],
            "model_output": model_output_text,
            "expected": sample['expected'],
            "judge": judge_output
        })
    
    with open("data/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Evaluation results saved to data/eval_results.json")
    return results