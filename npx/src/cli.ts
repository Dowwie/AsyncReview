/**
 * Main CLI handler for the review command
 */

import chalk from 'chalk';
import ora from 'ora';
import { getApiKey, getGitHubToken } from './api-key.js';
import {
    checkPython,
    checkDeno,
    installDeno,
    checkAsyncReviewInstalled,
    installAsyncReview,
    runPythonReview
} from './python-runner.js';

export interface ReviewOptions {
    url: string;
    question: string;
    output: string;
    quiet?: boolean;
    model?: string;
    api?: string;
    githubToken?: string;
}

export async function runReview(options: ReviewOptions): Promise<void> {
    const { url, question, output, quiet = false, model, api, githubToken } = options;

    try {

        // 4. Get API key
        const apiKey = await getApiKey(api);

        // 5. Get GitHub token (required for code search API)
        const ghToken = await getGitHubToken(githubToken, true);

        // 6. Run the review
        if (!quiet) {
            console.log(chalk.cyan(`\n 🔍 Reviewing: ${url}`));
            console.log(chalk.dim(`   Question: ${question}\n`));
        }

        const result = await runPythonReview({
            url,
            question,
            output,
            quiet,
            model,
            apiKey,
            githubToken: ghToken,
        });

        // In quiet mode, we suppressed stdout during execution
        // So we must print the final result now
        if (quiet && result.trim()) {
            console.log(result);
        }

    } catch (error) {
        if (error instanceof Error) {
            console.error(chalk.red(`\nError: ${error.message}`));
        } else {
            console.error(chalk.red('\nAn unexpected error occurred'));
        }
        process.exit(1);
    }
}
