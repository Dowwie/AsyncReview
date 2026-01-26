"""Build a CodebaseSnapshot from a repository path."""

import fnmatch
import hashlib
import os
import re
from pathlib import Path

from .config import (
    DEFAULT_IGNORE_PATTERNS,
    EXCLUDE_GLOBS,
    INCLUDE_GLOBS,
    MAX_FILE_BYTES,
    MAX_TOTAL_BYTES,
    PRIORITY_PATTERNS,
)
from .types import CodebaseSnapshot, FileInfo, RepoInfo, SymbolTag


# Language detection by extension
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".fish": "shell",
    ".ps1": "powershell",
    ".sql": "sql",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".md": "markdown",
    ".mdx": "markdown",
    ".rst": "restructuredtext",
    ".txt": "text",
    ".dockerfile": "dockerfile",
    ".vue": "vue",
    ".svelte": "svelte",
    ".astro": "astro",
    ".lua": "lua",
    ".r": "r",
    ".R": "r",
    ".jl": "julia",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hrl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".mli": "ocaml",
    ".clj": "clojure",
    ".cljs": "clojure",
    ".cljc": "clojure",
    ".dart": "dart",
    ".nim": "nim",
    ".zig": "zig",
    ".v": "v",
    ".sol": "solidity",
    ".proto": "protobuf",
    ".graphql": "graphql",
    ".gql": "graphql",
}


def detect_language(path: Path) -> str:
    """Detect programming language from file extension."""
    # Handle special filenames
    name = path.name.lower()
    if name == "dockerfile":
        return "dockerfile"
    if name == "makefile":
        return "makefile"
    if name in ("gemfile", "rakefile"):
        return "ruby"
    if name in (".gitignore", ".dockerignore", ".env", ".env.example"):
        return "text"

    ext = path.suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(ext, "text")


def should_ignore(path: Path, repo_root: Path) -> bool:
    """Check if a path should be ignored."""
    rel_path = str(path.relative_to(repo_root))
    parts = path.parts

    # Check default ignore patterns
    for pattern in DEFAULT_IGNORE_PATTERNS:
        # Check if any part matches
        for part in parts:
            if fnmatch.fnmatch(part, pattern):
                return True
        # Check full path
        if fnmatch.fnmatch(rel_path, pattern):
            return True

    # Check user exclude globs
    for pattern in EXCLUDE_GLOBS:
        if pattern and fnmatch.fnmatch(rel_path, pattern.strip()):
            return True

    return False


def matches_include_globs(path: Path, repo_root: Path) -> bool:
    """Check if path matches include globs (if any are specified)."""
    if not INCLUDE_GLOBS:
        return True  # No include globs = include everything not ignored

    rel_path = str(path.relative_to(repo_root))
    for pattern in INCLUDE_GLOBS:
        if pattern and fnmatch.fnmatch(rel_path, pattern.strip()):
            return True
    return False


def is_priority_file(path: Path, repo_root: Path) -> bool:
    """Check if file matches priority patterns."""
    rel_path = str(path.relative_to(repo_root))
    for pattern in PRIORITY_PATTERNS:
        if fnmatch.fnmatch(rel_path, pattern):
            return True
        if fnmatch.fnmatch(path.name, pattern):
            return True
    return False


def compute_sha1(content: bytes) -> str:
    """Compute SHA1 hash of content."""
    return hashlib.sha1(content).hexdigest()


def is_binary(content: bytes) -> bool:
    """Check if content appears to be binary."""
    # Check for null bytes in first 8KB
    return b"\x00" in content[:8192]


# Simple regex patterns for symbol extraction
SYMBOL_PATTERNS: dict[str, list[tuple[str, re.Pattern]]] = {
    "python": [
        ("function", re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(", re.MULTILINE)),
        ("class", re.compile(r"^\s*class\s+(\w+)\s*[:\(]", re.MULTILINE)),
        ("import", re.compile(r"^\s*(?:from\s+\S+\s+)?import\s+(\w+)", re.MULTILINE)),
    ],
    "javascript": [
        ("function", re.compile(r"^\s*(?:async\s+)?function\s+(\w+)\s*\(", re.MULTILINE)),
        ("function", re.compile(r"^\s*(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(", re.MULTILINE)),
        ("class", re.compile(r"^\s*class\s+(\w+)\s*(?:extends|{)", re.MULTILINE)),
        ("export", re.compile(r"^\s*export\s+(?:default\s+)?(?:const|let|var|function|class)\s+(\w+)", re.MULTILINE)),
    ],
    "typescript": [
        ("function", re.compile(r"^\s*(?:async\s+)?function\s+(\w+)\s*[<\(]", re.MULTILINE)),
        ("function", re.compile(r"^\s*(?:const|let|var)\s+(\w+)\s*(?::\s*\S+\s*)?=\s*(?:async\s+)?\(", re.MULTILINE)),
        ("class", re.compile(r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+(\w+)", re.MULTILINE)),
        ("export", re.compile(r"^\s*export\s+(?:default\s+)?(?:const|let|var|function|class|interface|type)\s+(\w+)", re.MULTILINE)),
    ],
    "rust": [
        ("function", re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)", re.MULTILINE)),
        ("class", re.compile(r"^\s*(?:pub\s+)?struct\s+(\w+)", re.MULTILINE)),
        ("class", re.compile(r"^\s*(?:pub\s+)?enum\s+(\w+)", re.MULTILINE)),
        ("class", re.compile(r"^\s*(?:pub\s+)?trait\s+(\w+)", re.MULTILINE)),
    ],
    "go": [
        ("function", re.compile(r"^\s*func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(", re.MULTILINE)),
        ("class", re.compile(r"^\s*type\s+(\w+)\s+struct", re.MULTILINE)),
        ("class", re.compile(r"^\s*type\s+(\w+)\s+interface", re.MULTILINE)),
    ],
}


def extract_symbols(content: str, language: str) -> list[SymbolTag]:
    """Extract symbol tags from file content."""
    patterns = SYMBOL_PATTERNS.get(language, [])
    if not patterns:
        return []

    symbols: list[SymbolTag] = []
    lines = content.split("\n")

    for kind, pattern in patterns:
        for match in pattern.finditer(content):
            symbol = match.group(1)
            # Find line number
            line_no = content[:match.start()].count("\n") + 1
            symbols.append(SymbolTag(symbol=symbol, kind=kind, line_no=line_no))

    # Sort by line number and dedupe
    symbols.sort(key=lambda s: (s["line_no"], s["symbol"]))
    seen = set()
    unique = []
    for s in symbols:
        key = (s["symbol"], s["line_no"])
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return unique


def build_snapshot(repo_path: str | Path) -> CodebaseSnapshot:
    """Build a CodebaseSnapshot from a repository path."""
    repo_root = Path(repo_path).resolve()
    if not repo_root.is_dir():
        raise ValueError(f"Repository path does not exist: {repo_root}")

    # Collect all files
    all_files: list[Path] = []
    for root, dirs, files in os.walk(repo_root):
        root_path = Path(root)

        # Filter directories in-place to skip ignored ones
        dirs[:] = [d for d in dirs if not should_ignore(root_path / d, repo_root)]

        for f in files:
            file_path = root_path / f
            if not should_ignore(file_path, repo_root) and matches_include_globs(file_path, repo_root):
                all_files.append(file_path)

    # Sort: priority files first, then alphabetically
    priority_files = [f for f in all_files if is_priority_file(f, repo_root)]
    other_files = [f for f in all_files if not is_priority_file(f, repo_root)]
    priority_files.sort(key=lambda p: str(p.relative_to(repo_root)))
    other_files.sort(key=lambda p: str(p.relative_to(repo_root)))
    sorted_files = priority_files + other_files

    # Build file tree and collect file info
    file_tree: list[str] = []
    files: dict[str, FileInfo] = {}
    tags: dict[str, list[SymbolTag]] = {}
    languages: dict[str, int] = {}
    total_bytes = 0

    for file_path in sorted_files:
        rel_path = str(file_path.relative_to(repo_root))
        file_tree.append(rel_path)

        # Check file size
        try:
            size = file_path.stat().st_size
        except OSError:
            continue

        if size > MAX_FILE_BYTES:
            continue  # Skip files that are too large

        if total_bytes + size > MAX_TOTAL_BYTES:
            break  # Stop if we've reached the total size cap

        # Read file content
        try:
            content_bytes = file_path.read_bytes()
        except OSError:
            continue

        if is_binary(content_bytes):
            continue  # Skip binary files

        try:
            content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            continue  # Skip files that can't be decoded

        language = detect_language(file_path)
        text_lines = content.split("\n")

        files[rel_path] = FileInfo(
            language=language,
            size_bytes=size,
            sha1=compute_sha1(content_bytes),
            text_lines=text_lines,
        )

        # Extract symbols
        file_tags = extract_symbols(content, language)
        if file_tags:
            tags[rel_path] = file_tags

        # Update stats
        languages[language] = languages.get(language, 0) + 1
        total_bytes += size

    repo_info = RepoInfo(
        root=str(repo_root),
        languages=languages,
        total_files=len(files),
        total_bytes=total_bytes,
    )

    return CodebaseSnapshot(
        repo_info=repo_info,
        file_tree=file_tree,
        files=files,
        tags=tags,
    )

