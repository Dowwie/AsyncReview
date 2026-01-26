"""Type definitions for the codebase review tool."""

from dataclasses import dataclass, field
from typing import TypedDict, Literal
from datetime import datetime


class FileInfo(TypedDict):
    """Information about a single file in the snapshot."""
    language: str
    size_bytes: int
    sha1: str
    text_lines: list[str]


class SymbolTag(TypedDict):
    """A lightweight symbol tag for indexing."""
    symbol: str
    kind: Literal["function", "class", "method", "variable", "import", "export"]
    line_no: int


class RepoInfo(TypedDict):
    """Repository metadata."""
    root: str
    languages: dict[str, int]  # language -> file count
    total_files: int
    total_bytes: int


@dataclass
class CodebaseSnapshot:
    """Complete snapshot of a codebase for RLM consumption."""
    repo_info: RepoInfo
    file_tree: list[str]  # flat list of relative paths
    files: dict[str, FileInfo]  # path -> file info
    tags: dict[str, list[SymbolTag]] = field(default_factory=dict)  # path -> tags

    def to_simple_dict(self) -> dict[str, str]:
        """Convert to simple dict of path -> content for RLM.

        This is the format DSPy RLM works best with - a flat dictionary
        where keys are file paths and values are file contents as strings.
        """
        result = {}
        for path, info in self.files.items():
            content = "\n".join(info["text_lines"])
            result[path] = content
        return result

    def to_dict(self) -> dict:
        """Convert to plain dict for RLM input (full metadata)."""
        return {
            "repo_info": self.repo_info,
            "file_tree": self.file_tree,
            "files": self.files,
            "tags": self.tags,
        }


@dataclass
class TraceStep:
    """A single step in the RLM trace."""
    step: int
    reasoning: str
    code: str
    stdout: str = ""
    artifacts: dict = field(default_factory=dict)


@dataclass
class RLMTrace:
    """Complete trace of an RLM run."""
    question: str
    repo_path: str
    started_at: datetime
    steps: list[TraceStep] = field(default_factory=list)
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    ended_at: datetime | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "question": self.question,
            "repo_path": self.repo_path,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "steps": [
                {
                    "step": s.step,
                    "reasoning": s.reasoning,
                    "code": s.code,
                    "stdout": s.stdout,
                    "artifacts": s.artifacts,
                }
                for s in self.steps
            ],
            "answer": self.answer,
            "sources": self.sources,
            "error": self.error,
        }


@dataclass
class Citation:
    """A source citation in the answer."""
    path: str
    start_line: int
    end_line: int
    snippet: str = ""
    reason: str = ""

    def __str__(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"

    @classmethod
    def parse(cls, citation_str: str) -> "Citation | None":
        """Parse a citation string like 'path/to/file.py:10-20'."""
        try:
            if ":" not in citation_str:
                return None
            path, line_range = citation_str.rsplit(":", 1)
            if "-" in line_range:
                start, end = line_range.split("-", 1)
                return cls(path=path, start_line=int(start), end_line=int(end))
            else:
                line = int(line_range)
                return cls(path=path, start_line=line, end_line=line)
        except (ValueError, IndexError):
            return None

