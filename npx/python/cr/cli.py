#!/usr/bin/env python3
"""CLI for Claude RLM Codebase Review Tool."""

import argparse
import sys
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

    # ask command (interactive)
    ask_parser = subparsers.add_parser("ask", help="Interactive Q&A mode")
    ask_parser.add_argument("--repo", "-r", type=str, default=".", help="Path to repository (default: .)")

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
        run_interactive(repo_path)
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

