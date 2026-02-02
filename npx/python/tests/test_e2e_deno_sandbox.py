"""E2E tests for Deno sandbox integration.

Tests the Deno/Pyodide execution environment used by the RLM:
1. Deno command construction for Deno 2.x compatibility
2. Variable injection into Pyodide environment
3. Code execution with serializable state
4. stdout capture from print statements
"""

import asyncio
import os
import pytest
from dspy.primitives.python_interpreter import PythonInterpreter
from cr.rlm_runner import build_deno_command


@pytest.mark.asyncio
async def test_deno_command_construction():
    """Test that Deno command is properly constructed for Deno 2.x."""
    cmd = build_deno_command()
    
    assert isinstance(cmd, list), "Command should be a list"
    assert cmd[0] == "deno", "First element should be 'deno'"
    assert "run" in cmd, "Should contain 'run' subcommand"
    
    # Check for Deno 2.x compatibility flags
    cmd_str = " ".join(cmd)
    assert "--node-modules-dir" in cmd_str, "Should have node-modules-dir flag for Deno 2.x"
    
    print(f"Deno command: {' '.join(cmd)}")


@pytest.mark.asyncio
async def test_pyodide_variable_injection():
    """Test that variables can be injected into Pyodide environment."""
    # Create interpreter with Deno 2.x command
    deno_cmd = build_deno_command()
    interpreter = PythonInterpreter(deno_command=deno_cmd)
    
    # Test variable injection
    test_vars = {
        "x": 42,
        "message": "hello world",
        "data": {"key": "value"},
        "items": [1, 2, 3],
    }
    
    code = """
x_doubled = x * 2
msg_upper = message.upper()
data_key = data['key']
items_sum = sum(items)

print(f"x_doubled={x_doubled}")
print(f"msg_upper={msg_upper}")
print(f"data_key={data_key}")
print(f"items_sum={items_sum}")
"""
    
    result = interpreter.execute(code, variables=test_vars)
    
    # Check that variables were accessible
    assert "x_doubled=84" in result or "84" in str(result), \
        f"x should be accessible and doubled. Result: {result}"
    assert "HELLO WORLD" in str(result).upper(), \
        f"message should be accessible. Result: {result}"


@pytest.mark.asyncio
async def test_print_capture():
    """Test that print statements are captured from Pyodide."""
    deno_cmd = build_deno_command()
    interpreter = PythonInterpreter(deno_command=deno_cmd)
    
    code = """
print("FETCH_FILE:test/path.py")
print("LIST_DIR:test/")
print("Normal output")
"""
    
    result = interpreter.execute(code)
    result_str = str(result) if not isinstance(result, list) else "\n".join(map(str, result))
    
    # All print statements should be captured
    assert "FETCH_FILE:test/path.py" in result_str, \
        f"FETCH_FILE command should be in output. Got: {result_str}"
    assert "LIST_DIR:test/" in result_str, \
        f"LIST_DIR command should be in output. Got: {result_str}"
    assert "Normal output" in result_str, \
        f"Normal print should be in output. Got: {result_str}"


@pytest.mark.asyncio
async def test_multi_turn_execution():
    """Test multiple code executions with state persistence."""
    deno_cmd = build_deno_command()
    interpreter = PythonInterpreter(deno_command=deno_cmd)
    
    # First execution: initialize state
    code1 = """
counter = 0
data_store = {}
print(f"Initialized: counter={counter}")
"""
    result1 = interpreter.execute(code1)
    
    # Second execution: should have access to updated variables
    updated_vars = {
        "counter": 5,
        "data_store": {"file1": "content1"},
    }
    
    code2 = """
counter += 1
data_store["file2"] = "content2"
print(f"Updated: counter={counter}")
print(f"Store keys: {list(data_store.keys())}")
"""
    result2 = interpreter.execute(code2, variables=updated_vars)
    result2_str = str(result2) if not isinstance(result2, list) else "\n".join(map(str, result2))
    
    # Check that variables persisted and updated
    assert "counter=6" in result2_str, \
        f"Counter should increment. Got: {result2_str}"
    assert "file1" in result2_str or "file2" in result2_str, \
        f"Store should have files. Got: {result2_str}"


@pytest.mark.asyncio
async def test_serialization_safety():
    """Test that only serializable data can be injected."""
    deno_cmd = build_deno_command()
    interpreter = PythonInterpreter(deno_command=deno_cmd)
    
    # These should work (serializable)
    safe_vars = {
        "string": "hello",
        "number": 42,
        "float_num": 3.14,
        "bool_val": True,
        "none_val": None,
        "list_val": [1, 2, 3],
        "dict_val": {"a": 1, "b": 2},
    }
    
    code = """
print(f"string={string}")
print(f"number={number}")
print(f"dict_val={dict_val}")
"""
    
    result = interpreter.execute(code, variables=safe_vars)
    result_str = str(result)
    
    assert "string=hello" in result_str, "String should be accessible"
    assert "number=42" in result_str, "Number should be accessible"
    assert "'a': 1" in result_str or "a:1" in result_str, "Dict should be accessible"


@pytest.mark.asyncio
async def test_error_handling_in_sandbox():
    """Test that errors in Pyodide are captured and returned."""
    deno_cmd = build_deno_command()
    interpreter = PythonInterpreter(deno_command=deno_cmd)
    
    code = """
# This will raise an error
result = undefined_variable
"""
    
    result = interpreter.execute(code)
    result_str = str(result)
    
    # Should contain error information
    assert "error" in result_str.lower() or "exception" in result_str.lower() or \
           "undefined" in result_str.lower() or "name" in result_str.lower(), \
        f"Should indicate error occurred. Got: {result_str}"


@pytest.mark.asyncio
async def test_repo_files_dict_injection():
    """Test that repo_files dict can be injected and accessed in sandbox.
    
    This simulates the actual use case in VirtualReviewRunner.
    """
    deno_cmd = build_deno_command()
    interpreter = PythonInterpreter(deno_command=deno_cmd)
    
    # Simulate fetched files
    repo_files = {
        "test/file1.py": "def hello():\n    print('hello')",
        "test/file2.py": "def world():\n    print('world')",
    }
    
    code = """
# Access fetched files
for path, content in repo_files.items():
    print(f"File: {path}")
    print(f"  Length: {len(content)}")
    
# Check specific file
if "test/file1.py" in repo_files:
    print("Found file1.py")
    print(f"Content preview: {repo_files['test/file1.py'][:20]}")
"""
    
    result = interpreter.execute(code, variables={"repo_files": repo_files})
    result_str = str(result) if not isinstance(result, list) else "\n".join(map(str, result))
    
    assert "File: test/file1.py" in result_str, \
        f"Should list file1.py. Got: {result_str}"
    assert "Found file1.py" in result_str, \
        f"Should find file1.py in dict. Got: {result_str}"
    assert "def hello" in result_str, \
        f"Should show content preview. Got: {result_str}"


if __name__ == "__main__":
    # Allow running tests directly
    print("Running Deno sandbox E2E tests...")
    pytest.main([__file__, "-v", "-s"])
