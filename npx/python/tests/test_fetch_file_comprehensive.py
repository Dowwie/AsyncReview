"""Comprehensive test to verify FETCH_FILE flow works correctly.

This test suite verifies:
1. GitHub API can fetch files successfully
2. Regex pattern matching works on REPL output
3. Tool interception updates repo_files dict
4. Variables are rebuilt with new state each iteration
5. Full E2E flow with explicit FETCH_FILE commands
"""

import asyncio
import os
import re
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.virtual_runner import VirtualReviewRunner
from cli.repo_tools import RepoTools


async def test_1_github_api_direct():
    """Test 1: Direct GitHub API - Can we fetch the file?"""
    print("\n" + "="*80)
    print("TEST 1: Direct GitHub API Access")
    print("="*80)
    
    # The file exists at: https://github.com/kmad/dspy/blob/main/dspy/predict/rlm.py
    owner = "kmad"
    repo = "dspy"
    path = "dspy/predict/rlm.py"
    ref = "main"
    
    tools = RepoTools(owner, repo, ref)
    
    try:
        print(f"Fetching: {owner}/{repo}/{path} @ {ref}")
        content = await tools.fetch_file(path)
        
        if content.startswith("[ERROR") or content.startswith("[SKIPPED"):
            print(f"❌ FAILED: {content}")
            return False
        
        print(f"✓ SUCCESS: Fetched {len(content)} characters")
        print(f"✓ Content preview: {content[:200]}...")
        
        # Verify it's the RLM file
        if "class RLM" in content or "Recursive Language Model" in content:
            print("✓ Content appears to be the RLM module")
            return True
        else:
            print("❌ WARNING: Content doesn't look like RLM module")
            return False
            
    finally:
        await tools.close()


async def test_2_regex_pattern_matching():
    """Test 2: Pattern Matching - Does regex capture FETCH_FILE commands?"""
    print("\n" + "="*80)
    print("TEST 2: Regex Pattern Matching")
    print("="*80)
    
    test_outputs = [
        ("FETCH_FILE:dspy/predict/rlm.py", ["dspy/predict/rlm.py"]),
        ("print('FETCH_FILE:test/file.py')", ["test/file.py')"]),  # Regex will match to quote
        ("Some text\nFETCH_FILE:path/to/file.py\nMore text", ["path/to/file.py"]),
        ("FETCH_FILE:file1.py\nFETCH_FILE:file2.py", ["file1.py", "file2.py"]),
        ("No fetch command here", []),
    ]
    
    pattern = r'FETCH_FILE:([^\s\n]+)'
    all_passed = True
    
    for output, expected in test_outputs:
        matches = re.findall(pattern, output)
        if matches == expected:
            print(f"✓ Pattern matched: {output[:50]} → {matches}")
        else:
            print(f"❌ Pattern mismatch: {output[:50]}")
            print(f"   Expected: {expected}")
            print(f"   Got: {matches}")
            all_passed = False
    
    return all_passed


async def test_3_tool_interception():
    """Test 3: Tool Interception - Does _process_tool_requests work?"""
    print("\n" + "="*80)
    print("TEST 3: Tool Interception Logic")
    print("="*80)
    
    runner = VirtualReviewRunner(model="claude-sonnet-4-6", quiet=True)
    
    # Set up repo tools for kmad/dspy
    runner._repo_tools = RepoTools("kmad", "dspy", "main")
    runner._repo_files = {}
    
    # Simulate REPL output with FETCH_FILE command
    simulated_output = """
Looking for the file...
FETCH_FILE:dspy/predict/rlm.py
Waiting for result...
"""
    
    try:
        print("Before interception:")
        print(f"  repo_files: {list(runner._repo_files.keys())}")
        
        # Process the tool request
        executed = await runner._process_tool_requests(simulated_output)
        
        print(f"\nAfter interception:")
        print(f"  Executed: {executed}")
        print(f"  repo_files: {list(runner._repo_files.keys())}")
        
        if not executed:
            print("❌ FAILED: No tools were executed")
            return False
        
        if "dspy/predict/rlm.py" not in runner._repo_files:
            print("❌ FAILED: File not in repo_files dict")
            return False
        
        content = runner._repo_files["dspy/predict/rlm.py"]
        if content.startswith("[ERROR") or content.startswith("[SKIPPED"):
            print(f"❌ FAILED: Error fetching file: {content}")
            return False
        
        print(f"✓ SUCCESS: File fetched and stored ({len(content)} chars)")
        print(f"✓ Content preview: {content[:200]}...")
        return True
        
    finally:
        if runner._repo_tools:
            await runner._repo_tools.close()


async def test_4_variable_rebuild():
    """Test 4: Variable Rebuild - Do variables update each iteration?"""
    print("\n" + "="*80)
    print("TEST 4: Variable Rebuild Per Iteration")
    print("="*80)
    
    runner = VirtualReviewRunner(model="claude-sonnet-4-6", quiet=True)
    runner._ensure_configured()
    
    # Simulate what happens in the iteration loop
    context = "Test PR context"
    question = "Test question"
    
    # Iteration 1: Empty repo_files
    runner._repo_files = {}
    input_args_1 = {
        "context": context,
        "question": question,
        "repo_files": runner._repo_files,
        "repo_dirs": {},
        "search_results": [],
    }
    variables_1 = runner._rlm._build_variables(**input_args_1)
    vars_str_1 = str([v.format() for v in variables_1])
    
    print("Iteration 1 (empty repo_files):")
    print(f"  repo_files in variables: {'repo_files' in vars_str_1}")
    print(f"  Is empty dict: {'{}'  in vars_str_1 or 'repo_files={}' in vars_str_1}")
    
    # Simulate file being fetched
    runner._repo_files["test/file.py"] = "file content here"
    
    # Iteration 2: With file in repo_files
    input_args_2 = {
        "context": context,
        "question": question,
        "repo_files": runner._repo_files,
        "repo_dirs": {},
        "search_results": [],
    }
    variables_2 = runner._rlm._build_variables(**input_args_2)
    vars_str_2 = str([v.format() for v in variables_2])
    
    print("\nIteration 2 (with file):")
    print(f"  repo_files in variables: {'repo_files' in vars_str_2}")
    print(f"  Has test/file.py: {'test/file.py' in vars_str_2}")
    print(f"  Has file content: {'file content' in vars_str_2}")
    
    # Check that variables updated
    if 'test/file.py' in vars_str_2:
        print("✓ SUCCESS: Variables are rebuilt with new state each iteration")
        return True
    else:
        print("❌ FAILED: Variables did not update with new repo_files")
        print(f"Variables iteration 2: {vars_str_2[:500]}")
        return False


async def test_5_e2e_with_explicit_command():
    """Test 5: E2E - Full flow with explicit FETCH_FILE command in question."""
    print("\n" + "="*80)
    print("TEST 5: End-to-End with Explicit Command")
    print("="*80)
    
    # Use the PR we've been testing with - we know it exists
    url = "https://github.com/stanfordnlp/dspy/pull/9240"
    
    # Very explicit question that tells the RLM exactly what to do
    question = """
Execute this exact Python code:
print("FETCH_FILE:dspy/predict/rlm.py")

Then on the next step, check if 'dspy/predict/rlm.py' is in repo_files and print the first 500 characters.
"""
    
    runner = VirtualReviewRunner(model="claude-sonnet-4-6", quiet=False)
    
    # Intercept to see what happens
    original_acall = None
    fetch_command_seen = False
    repo_files_seen_in_prompt = False
    
    async def intercepted_acall(*args, **kwargs):
        nonlocal fetch_command_seen, repo_files_seen_in_prompt
        
        iteration = kwargs.get('iteration', 'unknown')
        variables_info = kwargs.get('variables_info', [])
        
        variables_str = "\n".join(variables_info)
        
        # Check if repo_files is in the prompt
        if 'repo_files' in variables_str:
            # Check if it has content (not just empty dict)
            if 'dspy/predict/rlm.py' in variables_str:
                repo_files_seen_in_prompt = True
                print(f"\n[TEST] ✓ Iteration {iteration}: repo_files contains rlm.py in prompt")
        
        result = await original_acall(*args, **kwargs)
        
        # Check if code contains FETCH_FILE
        if hasattr(result, 'code') and 'FETCH_FILE' in result.code:
            fetch_command_seen = True
            print(f"\n[TEST] ✓ Iteration {iteration}: Generated code contains FETCH_FILE")
        
        return result
    
    runner._ensure_configured()
    original_acall = runner._rlm.generate_action.acall
    runner._rlm.generate_action.acall = intercepted_acall
    
    try:
        print(f"Testing with URL: {url}")
        print(f"Question: {question[:100]}...")
        
        answer, sources, metadata = await runner.review(url, question)
        
        print("\n" + "-"*80)
        print("Results:")
        print(f"  Files fetched: {metadata.get('files_fetched', []  )}")
        print(f"  FETCH command was generated: {fetch_command_seen}")
        print(f"  repo_files seen in prompt: {repo_files_seen_in_prompt}")
        
        # Success criteria
        success = True
        
        if "dspy/predict/rlm.py" not in metadata.get('files_fetched', []):
            print("❌ File was not fetched")
            success = False
        else:
            print("✓ File was fetched")
        
        if not fetch_command_seen:
            print("⚠️  WARNING: FETCH command not generated (LLM didn't follow instructions)")
        else:
            print("✓ FETCH command was generated")
        
        if not repo_files_seen_in_prompt:
            print("⚠️  WARNING: repo_files not seen in prompt after fetch")
        else:
            print("✓ repo_files appeared in prompt")
        
        return success
        
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests in sequence."""
    print("\n" + "="*80)
    print("FETCH_FILE COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    tests = [
        ("GitHub API Direct", test_1_github_api_direct),
        ("Regex Pattern Matching", test_2_regex_pattern_matching),
        ("Tool Interception", test_3_tool_interception),
        ("Variable Rebuild", test_4_variable_rebuild),
        ("E2E with Explicit Command", test_5_e2e_with_explicit_command),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            result = await test_func()
            results[name] = result
        except Exception as e:
            print(f"\n❌ Test '{name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-key'")
        exit(1)
    
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
