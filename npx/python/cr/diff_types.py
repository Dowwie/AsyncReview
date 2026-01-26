"""Type definitions for Part 2: Diff Review."""

from dataclasses import dataclass, field
from typing import Literal
from datetime import datetime


@dataclass
class FileContents:
    """File contents for diff rendering."""
    name: str
    contents: str
    cache_key: str | None = None


@dataclass
class DiffFileContext:
    """Context for a single file in a diff."""
    path: str
    old_file: FileContents | None = None
    new_file: FileContents | None = None
    patch: str | None = None
    language_hint: str | None = None
    additions: int = 0
    deletions: int = 0
    status: str = "modified"  # added, removed, modified, renamed


@dataclass
class DiffSelection:
    """User selection in the diff viewer."""
    path: str
    side: Literal["additions", "deletions", "unified"]
    start_line: int
    end_line: int
    mode: Literal["range", "single-line", "hunk", "file", "changeset"]


@dataclass
class DiffCitation:
    """Citation pointing to a specific location in the diff."""
    path: str
    side: Literal["additions", "deletions", "unified"]
    start_line: int
    end_line: int
    label: str | None = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "side": self.side,
            "startLine": self.start_line,
            "endLine": self.end_line,
            "label": self.label,
            "reason": self.reason,
        }


@dataclass
class LineAnnotation:
    """AI-generated inline annotation on a diff line."""
    id: str
    path: str
    side: Literal["additions", "deletions"]
    line_number: int
    thread: list[dict]  # [{author: 'user'|'ai', bodyMarkdown, createdAt}]
    status: Literal["open", "resolved"] = "open"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "path": self.path,
            "side": self.side,
            "lineNumber": self.line_number,
            "thread": self.thread,
            "status": self.status,
        }


@dataclass
class ReviewIssue:
    """An issue found during automatic review."""
    title: str
    severity: Literal["low", "medium", "high", "critical"]
    category: Literal["investigation", "informational"]
    explanation_markdown: str
    citations: list[DiffCitation] = field(default_factory=list)
    fix_suggestions: list[dict] = field(default_factory=list)
    tests_to_add: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "severity": self.severity,
            "category": self.category,
            "explanationMarkdown": self.explanation_markdown,
            "citations": [c.to_dict() for c in self.citations],
            "fixSuggestions": self.fix_suggestions,
            "testsToAdd": self.tests_to_add,
        }


@dataclass
class PRInfo:
    """GitHub Pull Request metadata."""
    review_id: str
    owner: str
    repo: str
    number: int
    title: str
    body: str
    base_sha: str
    head_sha: str
    files: list[dict]  # [{path, status, additions, deletions}]
    created_at: datetime = field(default_factory=datetime.now)
    # Extended metadata
    user: dict | None = None
    state: str = "open"
    draft: bool = False
    head_ref: str = ""
    base_ref: str = ""
    commits: int = 0
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0
    commits_list: list[dict] = field(default_factory=list)
    comments: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "reviewId": self.review_id,
            "repo": {"owner": self.owner, "repo": self.repo},
            "baseSha": self.base_sha,
            "headSha": self.head_sha,
            "title": self.title,
            "body": self.body,
            "files": self.files,
            "user": self.user,
            "state": self.state,
            "draft": self.draft,
            "headRef": self.head_ref,
            "baseRef": self.base_ref,
            "commits": self.commits,
            "additions": self.additions,
            "deletions": self.deletions,
            "changedFiles": self.changed_files,
            "commitsList": self.commits_list,
            "comments": self.comments,
        }


@dataclass
class AnswerBlock:
    """A block in the AI response."""
    type: Literal["markdown", "code"]
    content: str
    language: str | None = None

    def to_dict(self) -> dict:
        result = {"type": self.type, "content": self.content}
        if self.language:
            result["language"] = self.language
        return result


@dataclass
class RLMIteration:
    """Represents one iteration of RLM execution."""
    iteration: int
    max_iterations: int
    reasoning: str
    code: str
    output: str | None = None

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "maxIterations": self.max_iterations,
            "reasoning": self.reasoning,
            "code": self.code,
            "output": self.output,
        }
