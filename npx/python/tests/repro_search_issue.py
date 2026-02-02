"""Reproduction script to test SEARCH_CODE functionality.

This script verifies that:
1. SEARCH_CODE commands are intercepted correctly
2. Search results populate the search_results variable
3. Results are visible to the LLM in subsequent iterations
4. The llm_query_batched function can be found in dspy/predict/rlm.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.virtual_runner import VirtualReviewRunner


async def repro():
    """Run the reproduction test."""
    print("=" * 80)
    print("SEARCH_CODE REPRODUCTION TEST")
    print("=" * 80)
    
    runner = VirtualReviewRunner(model="gemini-3-flash-preview", quiet=False)
    runner._ensure_configured()
    
    # Intercept acall to debug variable state
    original_acall = runner._rlm.generate_action.acall
    iteration_count = [0]
    
    async def intercepted_acall(*args, **kwargs):
        iteration_count[0] += 1
        iteration = iteration_count[0]
        
        variables_info = kwargs.get('variables_info', [])
        variables_str = "\n".join(variables_info)
        
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration} - Variables passed to LLM:")
        print(f"{'='*80}")
        
        # Check if search_results is in the prompt
        if 'search_results' in variables_str:
            print("✓ search_results is in variables")
            
            # Check if it has content
            if 'llm_query_batched' in variables_str:
                print("✓ search_results contains 'llm_query_batched'")
                
                # Check if dspy/predict/rlm.py is mentioned
                if 'dspy/predict/rlm.py' in variables_str:
                    print("✓ search_results includes dspy/predict/rlm.py")
                else:
                    print("⚠️  dspy/predict/rlm.py not found in search results")
            else:
                # Check if it's empty
                if 'search_results: []' in variables_str or 'search_results=[]' in variables_str:
                    print("  search_results is empty (no searches performed yet)")
                else:
                    print("  search_results has content but doesn't mention llm_query_batched")
        else:
            print("❌ search_results NOT in variables")
        
        return await original_acall(*args, **kwargs)
    
    runner._rlm.generate_action.acall = intercepted_acall
    
    # Test with a PR from stanfordnlp/dspy
    url = "https://github.com/stanfordnlp/dspy/pull/9240"
    
    question = """
Use SEARCH_CODE to find all occurrences of 'llm_query_batched' in the repository.

Step 1: Execute this exact code:
print("SEARCH_CODE:llm_query_batched")

Step 2: On the next iteration, check the search_results variable and print:
- How many results were found
- The file paths that contain llm_query_batched
- Confirm if dspy/predict/rlm.py is in the results
"""
    
    print(f"\nTesting URL: {url}")
    print(f"Question: {question[:200]}...")
    print()
    
    try:
        answer, sources, metadata = await runner.review(url, question)
        
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"Answer:\n{answer}\n")
        print(f"Sources: {sources}")
        print(f"Metadata: {metadata}")
        
        # Verdict
        print("\n" + "=" * 80)
        print("VERDICT")
        print("=" * 80)
        
        if metadata.get('search_performed'):
            print("✓ Search was performed")
        else:
            print("⚠️  No search detected in metadata")
        
        if 'llm_query_batched' in answer.lower():
            print("✓ Answer mentions llm_query_batched")
        else:
            print("❌ Answer does not mention llm_query_batched")
        
        if 'dspy/predict/rlm.py' in answer:
            print("✓ Answer mentions dspy/predict/rlm.py")
        else:
            print("⚠️  Answer doesn't mention dspy/predict/rlm.py")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set")
        print("Set it with: export GEMINI_API_KEY='your-key'")
        exit(1)
    
    asyncio.run(repro())
