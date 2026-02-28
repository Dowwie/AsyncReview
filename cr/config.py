"""Shared configuration loaded from .env"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Provider presets: "anthropic" (default) or "openrouter"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()

_PROVIDER_PRESETS = {
    "anthropic": {
        "main": "anthropic/claude-sonnet-4-6",
        "sub": "anthropic/claude-haiku-4-5",
    },
    "openrouter": {
        # "main": "openrouter/moonshotai/kimi-k2.5",
        "main": "openrouter/z-ai/glm-5",
        # "sub": "openrouter/moonshotai/kimi-k2.5",
        "sub": "openrouter/z-ai/glm-5",
    },
}

_preset = _PROVIDER_PRESETS.get(LLM_PROVIDER, _PROVIDER_PRESETS["anthropic"])
MAIN_MODEL = os.getenv("MAIN_MODEL", _preset["main"])
SUB_MODEL = os.getenv("SUB_MODEL", _preset["sub"])
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "20"))
MAX_LLM_CALLS = int(os.getenv("MAX_LLM_CALLS", "20"))

# Retry configuration for transient API errors (overloaded, rate limit, 5xx)
RLM_MAX_RETRIES = int(os.getenv("RLM_MAX_RETRIES", "8"))
RLM_RETRY_BASE_WAIT = float(os.getenv("RLM_RETRY_BASE_WAIT", "2"))
RLM_RETRY_MAX_WAIT = float(os.getenv("RLM_RETRY_MAX_WAIT", "60"))

# Repo snapshot constraints
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", "200000"))  # 200KB per file
MAX_TOTAL_BYTES = int(os.getenv("MAX_TOTAL_BYTES", "5000000"))  # 5MB total
INCLUDE_GLOBS = os.getenv("INCLUDE_GLOBS", "").split(",") if os.getenv("INCLUDE_GLOBS") else []
EXCLUDE_GLOBS = os.getenv("EXCLUDE_GLOBS", "").split(",") if os.getenv("EXCLUDE_GLOBS") else []

# Cache/trace directory
CR_CACHE_DIR = Path(os.getenv("CR_CACHE_DIR", os.path.expanduser("~/.cr")))
TRACES_DIR = CR_CACHE_DIR / "traces"
REVIEWS_DIR = CR_CACHE_DIR / "reviews"

# GitHub API configuration (Part 2)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_API_BASE = os.getenv("GITHUB_API_BASE", "https://api.github.com")

# API Server configuration
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Default ignore patterns (always excluded)
DEFAULT_IGNORE_PATTERNS = [
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    ".svn",
    ".hg",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "target",
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.exe",
    "*.bin",
    "*.o",
    "*.a",
    "*.class",
    "*.jar",
    "*.war",
    "*.ear",
    "*.zip",
    "*.tar",
    "*.gz",
    "*.rar",
    "*.7z",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.ico",
    "*.svg",
    "*.woff",
    "*.woff2",
    "*.ttf",
    "*.eot",
    "*.mp3",
    "*.mp4",
    "*.avi",
    "*.mov",
    "*.pdf",
    "*.doc",
    "*.docx",
    "*.xls",
    "*.xlsx",
    "*.ppt",
    "*.pptx",
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "bun.lockb",
    "poetry.lock",
    "Cargo.lock",
    ".DS_Store",
    "Thumbs.db",
]

# Priority patterns for inclusion (checked first when building snapshot)
# These files are included first before hitting the MAX_TOTAL_BYTES limit
PRIORITY_PATTERNS = [
    # Documentation
    "README*",
    "readme*",
    # Package manifests
    "package.json",
    "pyproject.toml",
    "Cargo.toml",
    "go.mod",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    # Build files
    "Makefile",
    "Dockerfile",
    "docker-compose*.yml",
    ".env.example",
    # Source code at root level (important for small projects)
    "*.py",
    "*.ts",
    "*.js",
    "*.tsx",
    "*.jsx",
    "*.go",
    "*.rs",
    "*.java",
    "*.rb",
    "*.php",
    # Source directories
    "src/**",
    "lib/**",
    "app/**",
    "pkg/**",
    "cmd/**",
    # Test directories
    "tests/**",
    "test/**",
    "spec/**",
]

