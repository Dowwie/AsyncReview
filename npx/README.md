# asyncreview

AI-powered GitHub PR/Issue reviews from the command line using Recursive Language Models (RLM).

## Quick Start

```bash
# Review a PR
npx asyncreview review --url https://github.com/org/repo/pull/123 -q "Any security concerns?"

# With JSON output for scripting
npx asyncreview review --url https://github.com/org/repo/pull/123 -q "What does this PR do?" --quiet --output json
```

## Requirements

- **Node.js 18+**
- **Python 3.11+** (auto-detected)
- **Anthropic API Key** (prompted if not set)

## API Key

The Anthropic API key can be provided in three ways (in order of priority):

1. `--api <key>` flag
2. `ANTHROPIC_API_KEY` environment variable
3. Interactive prompt (if neither above is set)

```bash
# Using --api flag
npx asyncreview review --url <url> -q "Review this" --api YOUR_API_KEY

# Using environment variable
export ANTHROPIC_API_KEY=your_key
npx asyncreview review --url <url> -q "Review this"
```

## GitHub Token (For Private Repos)

For private repositories, you'll need to provide a GitHub token. The token can be provided in three ways (in order of priority):

1. `--github-token <token>` flag
2. `GITHUB_TOKEN` environment variable
3. Interactive prompt (optional, if neither above is set)

```bash
# Using --github-token flag
npx asyncreview review --url <url> -q "Review this" --github-token YOUR_GITHUB_TOKEN

# Using environment variable
export GITHUB_TOKEN=your_token
npx asyncreview review --url <url> -q "Review this"
```

## Options

| Option | Description |
|--------|-------------|
| `-u, --url <url>` | GitHub PR or Issue URL (required) |
| `-q, --question <question>` | Question to ask (required) |
| `-o, --output <format>` | Output format: `text`, `markdown`, `json` (default: text) |
| `--quiet` | Suppress progress output |
| `-m, --model <model>` | Model to use (default: claude-sonnet-4-6) |
| `--api <key>` | Anthropic API key |
| `--github-token <token>` | GitHub token for private repos |

## Examples

```bash
# Quick PR summary
npx asyncreview review -u https://github.com/vercel/next.js/pull/1234 -q "Summarize the changes"

# Security review
npx asyncreview review -u https://github.com/org/repo/pull/123 -q "Any security vulnerabilities?"

# Markdown output for docs
npx asyncreview review -u https://github.com/org/repo/pull/123 -q "Document these changes" -o markdown

# Scripting with JSON
npx asyncreview review -u https://github.com/org/repo/pull/123 -q "Review" --quiet -o json | jq .answer

# Private repository review
npx asyncreview review -u https://github.com/org/private-repo/pull/456 -q "Review this" --github-token ghp_xxxxxxxxxxxx
```

## License

MIT
