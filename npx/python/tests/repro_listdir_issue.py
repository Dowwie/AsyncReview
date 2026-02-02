"""Direct test of LIST_DIR against kmad/dspy repository.

Tests LIST_DIR command directly against GitHub API to verify:
- Directory listing works correctly
- Returns subdirectories (avatar/)
- Returns files (rlm.py, etc.)
- Expected count: 1 subdir + 15 files

Based on https://github.com/kmad/dspy/tree/main/dspy/predict
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.repo_tools import RepoTools


async def test_listdir_direct():
    """Test LIST_DIR directly via RepoTools."""
    print("=" * 80)
    print("LIST_DIR DIRECT TEST - kmad/dspy/dspy/predict")
    print("=" * 80)
    
    owner = "kmad"
    repo = "dspy"
    path = "dspy/predict"
    ref = "main"
    
    tools = RepoTools(owner, repo, ref)
    
    try:
        print(f"\nListing directory: {owner}/{repo}/{path} @ {ref}")
        entries = await tools.list_directory(path)
        
        if not entries:
            print("❌ FAILED: No entries returned")
            return False
        
        # Separate dirs and files
        dirs = [e for e in entries if e['type'] == 'dir']
        files = [e for e in entries if e['type'] == 'file']
        
        print(f"\n✓ SUCCESS: Found {len(entries)} total entries")
        print(f"  - {len(dirs)} subdirectories")
        print(f"  - {len(files)} files")
        
        # List subdirectories
        print(f"\nSubdirectories:")
        for d in dirs:
            name = d['path'].split('/')[-1]
            print(f"  📁 {name}")
        
        # List files (first 20)
        print(f"\nFiles (showing first 20):")
        for f in files[:20]:
            name = f['path'].split('/')[-1]
            print(f"  📄 {name}")
        
        # Verify expectations from screenshot
        print("\n" + "=" * 80)
        print("VERIFICATION")
        print("=" * 80)
        
        success = True
        
        # Check for avatar subdirectory
        avatar_found = any('avatar' in d['path'] for d in dirs)
        if avatar_found:
            print("✓ Found 'avatar' subdirectory")
        else:
            print("❌ 'avatar' subdirectory NOT found")
            success = False
        
        # Check for rlm.py file
        rlm_found = any('rlm.py' in f['path'] for f in files)
        if rlm_found:
            print("✓ Found 'rlm.py' file")
        else:
            print("❌ 'rlm.py' file NOT found")
            success = False
        
        # Check counts (approximate - may change)
        if len(dirs) >= 1:
            print(f"✓ Has at least 1 subdirectory (found {len(dirs)})")
        else:
            print(f"❌ Expected at least 1 subdirectory, found {len(dirs)}")
            success = False
        
        if len(files) >= 10:
            print(f"✓ Has at least 10 files (found {len(files)})")
        else:
            print(f"⚠️  Expected at least 10 files, found {len(files)}")
        
        return success
        
    finally:
        await tools.close()


if __name__ == "__main__":
    success = asyncio.run(test_listdir_direct())
    exit(0 if success else 1)
