#!/usr/bin/env bash
set -euo pipefail

REPO="${1:?Usage: loop.sh <repo_path> <runbook_path> [max_iterations]}"
RUNBOOK="${2:?Usage: loop.sh <repo_path> <runbook_path> [max_iterations]}"
MAX_ITER="${3:-5}"

RUNBOOK_DIR="$(dirname "$RUNBOOK")"
RUNBOOK_NAME="$(basename "$RUNBOOK" .runbook)"
MEMORY_FILE="${RUNBOOK_DIR}/${RUNBOOK_NAME}.memory"

# Initialize memory file if it doesn't exist
if [ ! -f "$MEMORY_FILE" ]; then
    echo "# Refinement memory for ${RUNBOOK_NAME}" > "$MEMORY_FILE"
    echo "# Tracks irreducible gaps and lessons learned across iterations" >> "$MEMORY_FILE"
    echo "" >> "$MEMORY_FILE"
fi

echo "=== Iterative Runbook Refinement ==="
echo "Repository: $REPO"
echo "Runbook:    $RUNBOOK"
echo "Memory:     $MEMORY_FILE"
echo "Max iterations: $MAX_ITER"
echo ""

for i in $(seq 1 "$MAX_ITER"); do
    echo "========================================"
    echo "=== Iteration $i of $MAX_ITER ==="
    echo "========================================"
    echo ""

    # Snapshot the runbook before this iteration
    cp "$RUNBOOK" "${RUNBOOK}.prev"

    # Run the runbook
    REPORT_FILE=$(uv run cr ask --repo "$REPO" --runbook "$RUNBOOK" 2>&1 | \
        grep "Report saved to" | sed 's/.*Report saved to //')

    if [ -z "$REPORT_FILE" ]; then
        echo "ERROR: No report file produced. Check cr ask output."
        exit 1
    fi

    echo ""
    echo "Report: $REPORT_FILE"
    echo ""

    # Refine the runbook based on the report + accumulated memory
    # Exit code 0 = changed, 1 = no change (convergence)
    if uv run cr refine --report "$REPORT_FILE" --runbook "$RUNBOOK" --memory "$MEMORY_FILE"; then
        echo ""
        echo "Runbook was refined. Diff:"
        diff "${RUNBOOK}.prev" "$RUNBOOK" || true
        echo ""
    else
        echo ""
        echo "=== Converged after $i iteration(s) ==="
        echo "No addressable gaps remain."
        rm -f "${RUNBOOK}.prev"
        break
    fi

    rm -f "${RUNBOOK}.prev"

    if [ "$i" -eq "$MAX_ITER" ]; then
        echo ""
        echo "=== Reached max iterations ($MAX_ITER) ==="
        echo "Review the latest report for remaining gaps."
    fi
done

echo ""
echo "Final runbook: $RUNBOOK"
echo "Memory file:   $MEMORY_FILE"
echo "Reports in:    $(pwd)"
