def get_system_prompt():
    """Return system prompt using RTFC framework."""
    return """
    Role: You are an expert code analyst proficient in multiple programming languages.
    Task: Analyze the provided code snippet for syntax errors, logical issues, suggest fixes, and summarize its functionality.
    Format: Respond with sections: 'Errors', 'Fixes', 'Summary' in a clear, concise manner.
    Constraints: Support Python, JavaScript, Java, etc. Ensure responses are accurate and actionable.
    """

def get_zero_shot_prompt(code_snippet):
    """Return zero-shot prompt for code analysis."""
    return f"""
    Analyze the following code snippet for errors, suggest fixes, and summarize its functionality:

    ```{code_snippet}```
    """

def get_one_shot_prompt(code_snippet):
    """Return one-shot prompt with an example."""
    return f"""
    Analyze the following code snippet for errors, suggest fixes, and summarize its functionality. Follow this example:

    Example Code:
    ```
    def add(a, b)
        return a + b
    ```
    Example Response:
    Errors: Missing colon after function definition.
    Fixes: Add colon: `def add(a, b):`.
    Summary: The function adds two numbers and returns their sum.

    Now analyze this code:
    ```{code_snippet}```
    """

def get_multi_shot_prompt(code_snippet):
    """Return multi-shot prompt with two examples."""
    return f"""
    Analyze the following code snippet for errors, suggest fixes, and summarize its functionality. Follow these examples:

    Example 1:
    Code:
    ```
    def add(a, b)
        return a + b
    ```
    Response:
    Errors: Missing colon after function definition.
    Fixes: Add colon: `def add(a, b):`.
    Summary: The function adds two numbers and returns their sum.

    Example 2:
    Code:
    ```
    for i in range(10)
        print(i
    ```
    Response:
    Errors: Missing colon after range, missing closing parenthesis in print.
    Fixes: Add colon: `range(10):`, add parenthesis: `print(i)`.
    Summary: Prints numbers 0 to 9.

    Now analyze this code:
    ```{code_snippet}```
    """

def get_dynamic_prompt(code_snippet):
    """Return dynamic prompt that adapts based on code characteristics."""
    language = "Python" if "def " in code_snippet else "Unknown"
    return f"""
    Analyze the following {language} code snippet for errors, suggest fixes, and summarize its functionality. Tailor your analysis to the specific language and context:

    ```{code_snippet}```
    """

def get_cot_prompt(code_snippet):
    """Return chain-of-thought prompt for step-by-step analysis."""
    return f"""
    Analyze the following code snippet by reasoning step-by-step:
    1. Identify any syntax errors.
    2. Check for logical errors.
    3. Suggest fixes for each error.
    4. Summarize the code's functionality.

    Code:
    ```{code_snippet}```

    Provide the response with sections: 'Errors', 'Fixes', 'Summary'.
    """