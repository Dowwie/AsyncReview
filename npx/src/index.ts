#!/usr/bin/env node
/**
 * AsyncReview CLI - AI-powered GitHub PR/Issue reviews
 * 
 * Usage:
 *   npx asyncreview review --url https://github.com/org/repo/pull/123 -q "Any risks?"
 */

import { program } from 'commander';
import { runReview } from './cli.js';

program
    .name('asyncreview')
    .description('AI-powered GitHub PR/Issue reviews from the command line')
    .version('0.1.0');

program
    .command('review')
    .description('Review a GitHub PR or Issue')
    .requiredOption('-u, --url <url>', 'GitHub PR or Issue URL')
    .option('-q, --question <question>', 'Question to ask about the PR/Issue (optional with --expert)')
    .option('--expert', 'Run expert code review (SOLID, Security, Performance, Code Quality)')
    .option('-o, --output <format>', 'Output format: text, markdown, json', 'text')
    .option('--quiet', 'Suppress progress output')
    .option('-m, --model <model>', 'Model to use (e.g. claude-sonnet-4-6)')
    .option('--api <key>', 'Anthropic API key (defaults to ANTHROPIC_API_KEY env var)')
    .option('--github-token <token>', 'GitHub token for private repos (defaults to GITHUB_TOKEN env var)')
    .action(async (options) => {
        await runReview(options);
    });

program.parse();
