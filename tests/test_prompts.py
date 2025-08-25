import pytest
from src.prompts import get_system_prompt, get_zero_shot_prompt, get_one_shot_prompt, get_multi_shot_prompt, get_dynamic_prompt, get_cot_prompt

def test_system_prompt():
    prompt = get_system_prompt()
    assert all(x in prompt for x in ["Role", "Task", "Format", "Constraints"])
    assert len(prompt) > 50

def test_zero_shot_prompt():
    code = "def test():\n    pass"
    prompt = get_zero_shot_prompt(code)
    assert code in prompt
    assert "Analyze" in prompt
    assert len(prompt.splitlines()) > 3

def test_one_shot_prompt():
    code = "def test():\n    pass"
    prompt = get_one_shot_prompt(code)
    assert code in prompt
    assert "Example Code" in prompt
    assert "Example Response" in prompt
    assert len(prompt.splitlines()) > 10

def test_multi_shot_prompt():
    code = "def test():\n    pass"
    prompt = get_multi_shot_prompt(code)
    assert code in prompt
    assert "Example 1" in prompt
    assert "Example 2" in prompt
    assert len(prompt.splitlines()) > 15

def test_dynamic_prompt():
    code = "def test():\n    pass"
    prompt = get_dynamic_prompt(code)
    assert code in prompt
    assert "Python" in prompt
    assert "Tailor" in prompt

def test_cot_prompt():
    code = "def test():\n    pass"
    prompt = get_cot_prompt(code)
    assert code in prompt
    assert "step-by-step" in prompt
    assert all(x in prompt for x in ["Errors", "Fixes", "Summary"])