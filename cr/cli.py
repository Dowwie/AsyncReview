#!/usr/bin/env python3
"""CLI for Claude RLM Codebase Review Tool."""

import argparse
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from .render import (
    console,
    print_answer,
    print_error,
    print_files,
    print_help,
    print_history,
    print_info,
    print_repo_info,
    print_step,
    print_welcome,
)
from .rlm_runner import CodebaseReviewRLM
from .snapshot import build_snapshot
from .types import CodebaseSnapshot


def run_interactive(repo_path: Path):
    """Run interactive Q&A mode."""
    print_info("Building codebase snapshot...")
    try:
        snapshot = build_snapshot(repo_path)
    except Exception as e:
        print_error(f"Failed to build snapshot: {e}")
        sys.exit(1)

    print_welcome(str(repo_path), snapshot.repo_info["total_files"])

    # Initialize RLM with step callback
    rlm = CodebaseReviewRLM(on_step=print_step)
    history: list[tuple[str, str]] = []

    while True:
        try:
            console.print("[bold cyan]?[/bold cyan] ", end="")
            question = input().strip()

            if not question:
                continue

            cmd = question.lower()

            # Handle commands
            if cmd in ("quit", "exit", "q"):
                break
            if cmd == "help":
                print_help()
                continue
            if cmd == "reset":
                history.clear()
                console.clear()
                print_welcome(str(repo_path), snapshot.repo_info["total_files"])
                continue
            if cmd == "history":
                print_history(history)
                continue
            if cmd == "files":
                print_files(snapshot.file_tree)
                continue
            if cmd == "info":
                print_repo_info(snapshot.repo_info)
                continue

            # Run question
            console.print()
            if history:
                console.print(f"[dim]Context: {len(history)} previous turn(s)[/dim]")

            # Truncate long questions for display
            display_q = question[:60] + "..." if len(question) > 60 else question
            console.rule(f"[bold]{display_q}[/bold]")

            try:
                answer, sources, trace = rlm.run(
                    repo_path=repo_path,
                    question=question,
                    history=history,
                    save_trace_file=True,
                )
                print_answer(answer, sources)
                history.append((question, answer))

                # Show trace file location
                if trace.ended_at:
                    print_info(f"Trace saved to ~/.cr/traces/")

            except Exception as e:
                print_error(str(e))

            console.print()

        except (KeyboardInterrupt, EOFError):
            break

    console.print("\n[dim]Goodbye![/dim]")


def parse_runbook(runbook_path: Path) -> Iterator[tuple[int, int, str]]:
    """Yield (index, total, question) from a runbook file.

    Skips blank lines and comments. Parses the file lazily but
    requires a first pass to count total questions for progress display.
    """
    lines = [
        line.strip()
        for line in runbook_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    for i, question in enumerate(lines, 1):
        yield i, len(lines), question


def write_report(
    runbook_name: str,
    repo_path: Path,
    history: list[tuple[str, str]],
    total: int,
) -> Path:
    """Write the runbook Q&A results to a report file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path.cwd() / f"{runbook_name}-report-{timestamp}.md"

    lines = [
        f"# Runbook Report: {runbook_name}",
        f"Repository: {repo_path}",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Questions answered: {len(history)}/{total}",
        "",
    ]

    for i, (question, answer) in enumerate(history, 1):
        lines.extend([
            f"## Q{i}: {question}",
            "",
            answer,
            "",
        ])

    report_path.write_text("\n".join(lines))
    return report_path


def run_runbook(repo_path: Path, runbook_path: Path):
    """Run a sequence of questions from a runbook file."""
    if not runbook_path.exists():
        print_error(f"Runbook not found: {runbook_path}")
        sys.exit(1)

    questions = parse_runbook(runbook_path)
    first = next(questions, None)
    if first is None:
        print_error("No questions found in runbook")
        sys.exit(1)

    print_info("Building codebase snapshot...")
    try:
        snapshot = build_snapshot(repo_path)
    except Exception as e:
        print_error(f"Failed to build snapshot: {e}")
        sys.exit(1)

    print_welcome(str(repo_path), snapshot.repo_info["total_files"])
    console.print(f"[bold]Running {first[1]} questions from runbook[/bold]\n")

    rlm = CodebaseReviewRLM(on_step=print_step, cache=False)
    history: list[tuple[str, str]] = []
    answered = 0
    total = first[1]
    interrupted = False

    def execute_question(i: int, total: int, question: str):
        nonlocal answered
        console.print()
        if history:
            console.print(f"[dim]Context: {len(history)} previous turn(s)[/dim]")

        display_q = question[:60] + "..." if len(question) > 60 else question
        console.rule(f"[bold][{i}/{total}] {display_q}[/bold]")

        try:
            answer, sources, trace = rlm.run(
                repo_path=repo_path,
                question=question,
                history=history,
                save_trace_file=True,
                snapshot=snapshot,
            )
            print_answer(answer, sources)
            history.append((question, answer))
            answered += 1
        except Exception as e:
            print_error(f"Question {i} failed: {e}")

        console.print()

    try:
        execute_question(*first)
        for step in questions:
            execute_question(*step)
    except KeyboardInterrupt:
        interrupted = True
        console.print("\n[bold yellow]Interrupted.[/bold yellow] Saving partial report...")

    runbook_name = runbook_path.stem
    report_path = write_report(runbook_name, repo_path, history, total)

    if interrupted:
        console.print(f"\n[bold yellow]Runbook interrupted.[/bold yellow] {answered}/{total} questions answered.")
    else:
        console.print(f"\n[bold green]Runbook complete.[/bold green] {answered}/{total} questions answered.")
    print_info(f"Report saved to {report_path}")
    return report_path


REFINE_QUESTION = """\
The codebase you have access to contains three files:
- report.md: The output of a runbook analysis, including a gap analysis that compared \
the runbook's synthesized guide against a real implementation.
- runbook.txt: The current runbook — a sequence of interview questions used to investigate \
a codebase.
- memory.md: Lessons from prior refinement iterations.

Analyze the gap findings in the report. For each gap, determine whether it could have \
been caught by a better, more general question, or whether it is inherently specific to \
that particular provider and no generalized question could anticipate it.

For gaps that a better question could catch: improve existing questions or add new ones. \
Questions must remain broad and general. They arc toward the goal by investigating \
domains and areas of concern, not by directing toward specific findings or \
implementation details. The question should work equally well on a codebase you've \
never seen. A question like "How does the system resolve a provider name to a live \
instance at runtime?" is good — it investigates a domain. A question like "Does the \
factory use string paths with load_class?" is bad — it presupposes the answer and \
would be meaningless on a different codebase.

Respect the memory file. Do not re-introduce questions that were previously removed. \
Do not reclassify gaps already marked irreducible. Do not undo prior improvements.

The validation and revision phases must remain as the final steps of the runbook.

Your answer must contain two sections separated by a line containing only "---":

SECTION 1: The complete improved runbook. Same format as the input: comment lines \
starting with # for phase headers, one question per line, blank lines between phases.

SECTION 2: Memory entries. One per line:
- IRREDUCIBLE: <gap that no general question can address>
- LEARNED: <lesson about what worked or what to avoid>
- KEPT: <change from a prior iteration that should be preserved>

If no memory entries are needed, write "NO NEW ENTRIES" after the separator.
"""


def run_refine(report_path: Path, runbook_path: Path, memory_path: Path | None = None) -> bool:
    """Refine a runbook using the RLM agent loop.

    Returns True if the runbook was changed.
    """
    report = report_path.read_text()
    runbook = runbook_path.read_text()
    memory = ""
    if memory_path and memory_path.exists():
        memory = memory_path.read_text()

    context = {
        "report.md": report,
        "runbook.txt": runbook,
        "memory.md": memory or "No prior iterations.",
    }

    print_info("Refining runbook via RLM agent...")

    rlm = CodebaseReviewRLM(on_step=print_step, cache=False)

    answer, sources, trace = rlm.run_with_context(
        context=context,
        question=REFINE_QUESTION,
        save_trace_file=True,
    )

    print_answer(answer, sources)

    if "---" in answer:
        parts = answer.split("---", 1)
        new_runbook = parts[0].strip()
        memory_entries = parts[1].strip()
    else:
        new_runbook = answer.strip()
        memory_entries = ""

    if new_runbook == runbook.strip():
        print_info("No changes needed — runbook is stable.")
        return False

    runbook_path.write_text(new_runbook + "\n")
    print_info(f"Runbook updated: {runbook_path}")

    if memory_path and memory_entries and memory_entries != "NO NEW ENTRIES":
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(memory_path, "a") as f:
            f.write(f"\n## Iteration — {timestamp}\n")
            f.write(memory_entries + "\n")
        print_info(f"Memory updated: {memory_path}")

    return True


def run_one_shot(repo_path: Path, question: str):
    """Run a single question and exit."""
    print_info("Building codebase snapshot...")
    try:
        snapshot = build_snapshot(repo_path)
    except Exception as e:
        print_error(f"Failed to build snapshot: {e}")
        sys.exit(1)

    print_info(f"Indexed {snapshot.repo_info['total_files']} files")

    # Initialize RLM with step callback
    rlm = CodebaseReviewRLM(on_step=print_step)

    console.print()
    console.rule(f"[bold]{question[:60]}{'...' if len(question) > 60 else ''}[/bold]")

    try:
        answer, sources, trace = rlm.run_one_shot(
            repo_path=repo_path,
            question=question,
            save_trace_file=True,
        )
        print_answer(answer, sources)
        print_info(f"Trace saved to ~/.cr/traces/")

    except Exception as e:
        print_error(str(e))
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Claude RLM Codebase Review Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # review command (one-shot)
    review_parser = subparsers.add_parser("review", help="Ask a single question about the codebase")
    review_parser.add_argument("--repo", "-r", type=str, default=".", help="Path to repository (default: .)")
    review_parser.add_argument("--question", "-q", type=str, required=True, help="Question to ask")

    # ask command (interactive or runbook)
    ask_parser = subparsers.add_parser("ask", help="Interactive Q&A mode")
    ask_parser.add_argument("--repo", "-r", type=str, default=".", help="Path to repository (default: .)")
    ask_parser.add_argument("--runbook", "-b", type=str, default=None, help="Path to runbook file (runs questions in sequence)")

    # refine command
    refine_parser = subparsers.add_parser("refine", help="Refine a runbook based on gap findings in a report")
    refine_parser.add_argument("--report", type=str, required=True, help="Path to the report file from a runbook run")
    refine_parser.add_argument("--runbook", "-b", type=str, required=True, help="Path to the runbook file to refine")
    refine_parser.add_argument("--memory", "-m", type=str, default=None, help="Path to memory file for cross-iteration context")

    # serve command (Part 2 API server)
    serve_parser = subparsers.add_parser("serve", help="Start the API server for web UI")
    serve_parser.add_argument("--host", type=str, default=None, help="Host to bind (default: from .env or 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=None, help="Port to bind (default: from .env or 8000)")

    args = parser.parse_args()

    if args.command == "review":
        repo_path = Path(args.repo).resolve()
        run_one_shot(repo_path, args.question)
    elif args.command == "ask":
        repo_path = Path(args.repo).resolve()
        if args.runbook:
            run_runbook(repo_path, Path(args.runbook).resolve())
        else:
            run_interactive(repo_path)
    elif args.command == "refine":
        memory_path = Path(args.memory).resolve() if args.memory else None
        changed = run_refine(
            Path(args.report).resolve(),
            Path(args.runbook).resolve(),
            memory_path,
        )
        sys.exit(0 if changed else 1)
    elif args.command == "serve":
        from .config import API_HOST, API_PORT
        from .server import app
        import uvicorn

        host = args.host or API_HOST
        port = args.port or API_PORT
        print_info(f"Starting API server at http://{host}:{port}")
        uvicorn.run(app, host=host, port=port)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

