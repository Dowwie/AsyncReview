"""E2E tests for VirtualReviewRunner with real Gemini API and GitHub.

These tests verify:
1. FETCH_FILE tool interception works across iterations
2. Variables are properly rebuilt with tool results
3. Deno/Pyodide sandbox integration is stable
4. Multi-turn RLM conversations handle state correctly

Requirements:
- GEMINI_API_KEY environment variable must be set
- Internet connection for GitHub and Gemini API
- Deno must be installed and in PATH
"""

import asyncio
import os
import pytest
from cli.virtual_runner import VirtualReviewRunner


# Require explicit API key for E2E tests
@pytest.fixture(scope="module")
def gemini_api_key():
    """Ensure GEMINI_API_KEY is set for E2E tests."""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY not set, skipping E2E tests")
    return key


@pytest.mark.asyncio
async def test_fetch_file_interception(gemini_api_key):
    """Test that FETCH_FILE reliably populates repo_files across iterations.
    
    This is the core test for the variable rebuild fix. It verifies that when
    the RLM requests a file via FETCH_FILE, the fetched content appears in
    the repo_files variable in subsequent iterations.
    """
    # Test PR with known file
    url = "https://github.com/stanfordnlp/dspy/pull/9240"
    question = "What is in dspy/predict/rlm.py? Please fetch and analyze the complete contents of this file."
    
    runner = VirtualReviewRunner(model="gemini-3-flash-preview", quiet=False)
    
    # Intercept to verify state propagation
    original_acall = None
    iterations_with_repo_files = []
    
    async def intercepted_acall(*args, **kwargs):
        iteration = kwargs.get('iteration', 'unknown')
        variables_info = kwargs.get('variables_info', [])
        
        # Check if repo_files is in the prompt
        variables_str = "\n".join(variables_info)
        has_repo_files = "repo_files" in variables_str
        
        print(f"\n[TEST] Iteration {iteration} - repo_files in prompt? {has_repo_files}")
        if has_repo_files:
            iterations_with_repo_files.append(iteration)
        
        return await original_acall(*args, **kwargs)
    
    runner._ensure_configured()
    original_acall = runner._rlm.generate_action.acall
    runner._rlm.generate_action.acall = intercepted_acall
    
    answer, sources, metadata = await runner.review(url, question)
    
    # Assertions
    assert answer, "Should return an answer"
    assert isinstance(answer, str), "Answer should be a string"
    
    # Check that file was actually fetched
    files_fetched = metadata.get("files_fetched", [])
    assert any("rlm.py" in f for f in files_fetched), \
        f"Should have fetched rlm.py, but got: {files_fetched}"
    
    # Critical: repo_files should appear in at least one iteration prompt
    assert len(iterations_with_repo_files) > 0, \
        "repo_files should appear in at least one iteration prompt (fix verification)"


@pytest.mark.asyncio
async def test_search_code_tool(gemini_api_key):
    """Test SEARCH_CODE tool integration."""
    url = "https://github.com/stanfordnlp/dspy/pull/9240"
    question = "Use SEARCH_CODE to find all files related to 'DataFrame'. List the paths you find."
    
    runner = VirtualReviewRunner(model="gemini-2.0-flash-exp", quiet=True)
    
    answer, sources, metadata = await runner.review(url, question)
    
    assert answer, "Should return an answer"
    # Answer should mention DataFrame or files
    assert "dataframe" in answer.lower() or "frame" in answer.lower(), \
        "Answer should mention DataFrame or related files"


@pytest.mark.asyncio
async def test_list_directory_tool(gemini_api_key):
    """Test LIST_DIR tool integration."""
    url = "https://github.com/stanfordnlp/dspy/pull/9240"
    question = "Use LIST_DIR to list the contents of the 'dspy/predict/' directory."
    
    runner = VirtualReviewRunner(model="gemini-2.0-flash-exp", quiet=True)
    
    answer, sources, metadata = await runner.review(url, question)
    
    assert answer, "Should return an answer"
    # Should mention some files from the directory
    assert "rlm.py" in answer.lower() or ".py" in answer.lower(), \
        "Answer should mention Python files in the directory"


@pytest.mark.asyncio
async def test_multi_file_fetch(gemini_api_key):
    """Test fetching multiple files in sequence."""
    url = "https://github.com/stanfordnlp/dspy/pull/9240"
    question = ("Find and fetch both dspy/predict/rlm.py and any test file related to RLM. "
                "Compare their contents briefly.")
    
    runner = VirtualReviewRunner(model="gemini-2.0-flash-exp", quiet=True)
    
    answer, sources, metadata = await runner.review(url, question)
    
    assert answer, "Should return an answer"
    files_fetched = metadata.get("files_fetched", [])
    
    # Should have fetched at least 2 files
    assert len(files_fetched) >= 2, \
        f"Should have fetched at least 2 files, got {len(files_fetched)}: {files_fetched}"


@pytest.mark.asyncio
async def test_error_handling_invalid_path(gemini_api_key):
    """Test that invalid file paths are handled gracefully."""
    url = "https://github.com/stanfordnlp/dspy/pull/9240"
    question = "Try to fetch the file 'nonexistent/fake/path.py' and report what happens."
    
    runner = VirtualReviewRunner(model="gemini-2.0-flash-exp", quiet=True)
    
    # Should not raise, even with invalid path
    answer, sources, metadata = await runner.review(url, question)
    
    assert answer, "Should return an answer even when file doesn't exist"
    # Answer should mention error or not found
    assert "error" in answer.lower() or "not found" in answer.lower() or \
           "doesn't exist" in answer.lower() or "does not exist" in answer.lower(), \
        "Answer should indicate the file was not found"


@pytest.mark.asyncio
async def test_issue_review(gemini_api_key):
    """Test reviewing a GitHub issue (not just PRs)."""
    # Use a known issue
    url = "https://github.com/stanfordnlp/dspy/issues/100"
    question = "Summarize what this issue is about."
    
    runner = VirtualReviewRunner(model="gemini-2.0-flash-exp", quiet=True)
    
    answer, sources, metadata = await runner.review(url, question)
    
    assert answer, "Should return an answer for issues"
    assert metadata.get("type") == "issue", "Should identify as issue type"


@pytest.mark.asyncio 
async def test_context_preservation(gemini_api_key):
    """Test that PR context (diff, description) is preserved alongside tool results."""
    url = "https://github.com/stanfordnlp/dspy/pull/9240"
    question = ("Based on the PR description and the actual code in dspy/predict/rlm.py, "
                "explain how the DataFrame feature is implemented.")
    
    runner = VirtualReviewRunner(model="gemini-2.0-flash-exp", quiet=True)
    
    answer, sources, metadata = await runner.review(url, question)
    
    assert answer, "Should return an answer"
    # Should reference both PR context and file content
    assert "dataframe" in answer.lower(), "Should mention DataFrame from PR context"
    
    files_fetched = metadata.get("files_fetched", [])
    assert any("rlm.py" in f for f in files_fetched), "Should have fetched rlm.py"


if __name__ == "__main__":
    # Allow running tests directly with: python test_e2e_virtual_runner.py
    import sys
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        sys.exit(1)
    
    print("Running E2E tests...")
    pytest.main([__file__, "-v", "-s"])
