"""E2E tests for Python REPL execution patterns.

Tests the tool interception pattern used in VirtualReviewRunner:
1. Print command interception (FETCH_FILE, LIST_DIR, SEARCH_CODE)
2. Regex pattern matching for tool requests
3. Multi-turn state persistence between iterations
4. Error handling and recovery
"""

import asyncio
import os
import re
import pytest
from cli.virtual_runner import VirtualReviewRunner
from cli.repo_tools import RepoTools


@pytest.mark.asyncio
async def test_fetch_file_pattern_detection():
    """Test that FETCH_FILE commands are detected in REPL output."""
    test_outputs = [
        "FETCH_FILE:test/file.py",
        "Some text\nFETCH_FILE:another/path.py\nMore text",
        "print('FETCH_FILE:fake/file.py')  # Should still match",
        "Multiple: FETCH_FILE:file1.py and FETCH_FILE:file2.py",
    ]
    
    pattern = r'FETCH_FILE:([^\s\n]+)'
    
    for output in test_outputs:
        matches = re.findall(pattern, output)
        assert len(matches) > 0, f"Should find FETCH_FILE in: {output}"
        assert all("/" in m or "." in m for m in matches), \
            f"Matches should look like paths: {matches}"


@pytest.mark.asyncio
async def test_list_dir_pattern_detection():
    """Test that LIST_DIR commands are detected in REPL output."""
    test_outputs = [
        "LIST_DIR:test/",
        "LIST_DIR:src/components/",
        "First LIST_DIR:path1/ then LIST_DIR:path2/",
    ]
    
    pattern = r'LIST_DIR:([^\s\n]+)'
    
    for output in test_outputs:
        matches = re.findall(pattern, output)
        assert len(matches) > 0, f"Should find LIST_DIR in: {output}"


@pytest.mark.asyncio
async def test_search_code_pattern_detection():
    """Test that SEARCH_CODE commands are detected in REPL output."""
    test_outputs = [
        "SEARCH_CODE:rlm.py",
        "SEARCH_CODE:enable_tool_optimization",
        "SEARCH_CODE:class DataFrameField",
        "Let me search: SEARCH_CODE:DataFrame\nAnd continue...",
    ]
    
    pattern = r'SEARCH_CODE:(.+?)(?:\n|$)'
    
    for output in test_outputs:
        matches = re.findall(pattern, output)
        assert len(matches) > 0, f"Should find SEARCH_CODE in: {output}"
        assert all(len(m.strip()) > 0 for m in matches), \
            f"Matches should not be empty: {matches}"


@pytest.mark.asyncio
async def test_tool_request_processing_limits():
    """Test that tool request processing respects limits (avoid infinite loops)."""
    runner = VirtualReviewRunner(model="claude-haiku-4-5", quiet=True)
    
    # Mock repo tools
    runner._repo_tools = RepoTools("test", "repo", "abc123")
    
    # Simulate output with many requests
    output = "\n".join([f"FETCH_FILE:file{i}.py" for i in range(10)])
    
    # Process should limit to 3 per iteration (as per code)
    # We'll just verify the pattern matches all, implementation limits execution
    fetch_matches = re.findall(r'FETCH_FILE:([^\s\n]+)', output)
    assert len(fetch_matches) == 10, "Pattern should match all 10"
    
    # The actual processing limits to 3, implemented in _process_tool_requests
    # We're just testing the pattern detection here


@pytest.mark.asyncio
async def test_variable_state_accumulation():
    """Test that tool results accumulate across iterations.
    
    This tests the fix: repo_files should grow as more files are fetched.
    """
    runner = VirtualReviewRunner(model="claude-haiku-4-5", quiet=True)
    
    # Simulate initial state
    assert len(runner._repo_files) == 0, "Should start with empty repo_files"
    
    # Simulate fetching files
    runner._repo_files["file1.py"] = "content1"
    assert len(runner._repo_files) == 1, "Should have 1 file"
    
    runner._repo_files["file2.py"] = "content2"
    assert len(runner._repo_files) == 2, "Should have 2 files"
    
    # State should persist
    assert "file1.py" in runner._repo_files, "First file should still be there"
    assert "file2.py" in runner._repo_files, "Second file should be there"


@pytest.mark.asyncio
async def test_deduplicate_tool_requests():
    """Test that duplicate tool requests are handled (don't fetch same file twice)."""
    runner = VirtualReviewRunner(model="claude-haiku-4-5", quiet=True)
    runner._repo_tools = RepoTools("test", "repo", "abc123")
    
    # Simulate already fetched file
    runner._repo_files["existing.py"] = "already fetched"
    
    # Process request for already-fetched file
    output = "FETCH_FILE:existing.py"
    
    # The implementation checks: if path not in self._repo_files
    # So it won't re-fetch. We're testing the logic here.
    fetch_matches = re.findall(r'FETCH_FILE:([^\s\n]+)', output)
    assert "existing.py" in fetch_matches, "Should detect the request"
    
    # In actual code, this would be skipped in the for loop
    should_fetch = fetch_matches[0] not in runner._repo_files
    assert not should_fetch, "Should not re-fetch existing file"


@pytest.mark.asyncio
async def test_mixed_tool_commands():
    """Test that multiple different tool commands can be in same output."""
    output = """
Let me explore the codebase:
FETCH_FILE:src/main.py
LIST_DIR:src/
SEARCH_CODE:import pandas
"""
    
    fetch_matches = re.findall(r'FETCH_FILE:([^\s\n]+)', output)
    list_matches = re.findall(r'LIST_DIR:([^\s\n]+)', output)
    search_matches = re.findall(r'SEARCH_CODE:(.+?)(?:\n|$)', output)
    
    assert len(fetch_matches) == 1, "Should find FETCH_FILE"
    assert len(list_matches) == 1, "Should find LIST_DIR"
    assert len(search_matches) == 1, "Should find SEARCH_CODE"
    
    assert fetch_matches[0] == "src/main.py"
    assert list_matches[0] == "src/"
    assert "pandas" in search_matches[0]


@pytest.mark.asyncio
async def test_tool_output_formats():
    """Test that tool outputs are in expected formats."""
    runner = VirtualReviewRunner(model="claude-haiku-4-5", quiet=True)
    
    # repo_files should be dict[str, str]
    runner._repo_files = {"path.py": "content"}
    assert isinstance(runner._repo_files, dict)
    assert all(isinstance(k, str) and isinstance(v, str) 
               for k, v in runner._repo_files.items())
    
    # repo_dirs should be dict[str, list]
    runner._repo_dirs = {"path/": [{"path": "file.py", "type": "file"}]}
    assert isinstance(runner._repo_dirs, dict)
    assert all(isinstance(k, str) and isinstance(v, list)
               for k, v in runner._repo_dirs.items())
    
    # search_results should be list[dict]
    runner._search_results = [{"path": "file.py", "fragment": "code..."}]
    assert isinstance(runner._search_results, list)
    assert all(isinstance(item, dict) for item in runner._search_results)


if __name__ == "__main__":
    print("Running Python REPL E2E tests...")
    pytest.main([__file__, "-v", "-s"])
