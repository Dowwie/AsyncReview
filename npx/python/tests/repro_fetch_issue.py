"""Minimal reproduction of FETCH_FILE issue.

This script tests the fix for the variable rebuild issue in VirtualReviewRunner.
Before fix: repo_files appears empty in LLM prompt even after files are fetched.
After fix: repo_files is properly populated in each iteration's prompt.
"""

import asyncio
import os
from cli.virtual_runner import VirtualReviewRunner


async def repro():
    """Run reproduction test with detailed debug output."""
    # Set API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        return
    
    print("=" * 80)
    print("FETCH_FILE REPRODUCTION TEST")
    print("=" * 80)
    
    runner = VirtualReviewRunner(model="gemini-3-flash-preview", quiet=False)
    runner._ensure_configured()
    
    # Intercept acall to debug variable state
    original_acall = runner._rlm.generate_action.acall
    iterations_data = []
    
    async def intercepted_acall(*args, **kwargs):
        iteration = kwargs.get('iteration', 'unknown')
        variables_info = kwargs.get('variables_info', [])
        
        print(f"\n{'='*80}")
        print(f"[DEBUG] Iteration {iteration}")
        print(f"{'='*80}")
        
        # Check what variables are being passed to the LLM
        for i, v in enumerate(variables_info):
            var_preview = v[:300] if len(v) > 300 else v
            print(f"\nVariable {i} (length: {len(v)}):")
            print(var_preview)
            if len(v) > 300:
                print("... (truncated)")
        
        # Check if repo_files is mentioned in the prompt
        variables_str = "\n".join(variables_info)
        has_repo_files = "repo_files" in variables_str
        
        # Count how many files are in repo_files dict (look for dictionary representation)
        import re
        files_match = re.search(r"repo_files.*?(\{[^}]*\})", variables_str, re.DOTALL)
        files_count = 0
        if files_match:
            files_dict_str = files_match.group(1)
            # Count keys in dict representation
            files_count = files_dict_str.count(":")
        
        print(f"\n[DEBUG] repo_files in prompt? {has_repo_files}")
        print(f"[DEBUG] Number of files in repo_files: {files_count}")
        
        iterations_data.append({
            'iteration': iteration,
            'has_repo_files': has_repo_files,
            'files_count': files_count
        })
        
        return await original_acall(*args, **kwargs)
    
    runner._rlm.generate_action.acall = intercepted_acall
    
    # Test URL and question
    url = "https://github.com/stanfordnlp/dspy/pull/9240"
    question = "What is in dspy/predict/rlm.py? Please fetch and analyze the complete contents of this file."
    
    print(f"\nTesting URL: {url}")
    print(f"Question: {question}\n")
    
    try:
        answer, sources, metadata = await runner.review(url, question)
        
        print(f"\n{'='*80}")
        print("FINAL RESULT")
        print(f"{'='*80}")
        print(f"Answer preview: {answer[:500]}...")
        print(f"\nFiles fetched: {metadata.get('files_fetched', [])}")
        print(f"Model: {metadata.get('model')}")
        
        # Analyze iterations
        print(f"\n{'='*80}")
        print("ITERATION ANALYSIS")
        print(f"{'='*80}")
        for data in iterations_data:
            print(f"Iteration {data['iteration']}: "
                  f"repo_files={'✓' if data['has_repo_files'] else '✗'}, "
                  f"files={data['files_count']}")
        
        # Determine if fix worked
        print(f"\n{'='*80}")
        print("VERDICT")
        print(f"{'='*80}")
        
        if len(iterations_data) > 1:
            # After first iteration, repo_files should be populated if files were fetched
            later_iterations = iterations_data[1:]
            has_populated = any(d['files_count'] > 0 for d in later_iterations)
            
            if has_populated:
                print("✓ SUCCESS: repo_files was populated in later iterations")
                print("  The fix is working correctly!")
            else:
                print("✗ FAILURE: repo_files never got populated in later iterations")
                print("  The bug still exists!")
        else:
            print("? UNCLEAR: Only one iteration ran, cannot determine if fix works")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print("ERROR DURING EXECUTION")
        print(f"{'='*80}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(repro())
