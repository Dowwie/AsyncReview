/**
 * Main CLI handler for the review command
 */

import chalk from 'chalk';
import ora from 'ora';
import { getApiKey, getGitHubToken } from './api-key.js';
import {
    checkPython,
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
        // 1. Check Python availability
        // 1. Check Python availability
        const spinner = !quiet ? ora('Checking Python environment...').start() : null;
        const pythonCheck = checkPython();

        if (!pythonCheck.available) {
            if (spinner) spinner.fail('Python 3.11+ is required');
            if (!quiet) {
                console.log(chalk.red('\nPlease install Python 3.11 or later:'));
                console.log(chalk.dim('  • macOS: brew install python'));
                console.log(chalk.dim('  • Ubuntu: sudo apt install python3.11'));
                console.log(chalk.dim('  • Windows: https://python.org/downloads'));
            }
            process.exit(1);
        }
        if (spinner) spinner.succeed(`Python ${pythonCheck.version?.replace('Python ', '')} found`);

        // 2. Check if asyncreview is installed
        if (!checkAsyncReviewInstalled(pythonCheck.pythonCmd)) {
            if (!quiet) {
                console.log(chalk.yellow('\nAsyncReview environment not ready.'));
                console.log(chalk.dim('Setting up isolated environment...\n'));
            }

            // Pass quiet flag to installation
            const installed = await installAsyncReview(pythonCheck.pythonCmd, quiet);
            if (!installed) {
                if (!quiet) {
                    console.log(chalk.red('\nFailed to install AsyncReview.'));
                    console.log(chalk.dim('Try running manually: pip install -e <path-to-asyncreview>'));
                }
                process.exit(1);
            }
        }

        // 3. Get API key
        const apiKey = await getApiKey(api);

        // 4. Get GitHub token (optional for public repos)
        const ghToken = await getGitHubToken(githubToken, false);

        // 5. Run the review
        if (!quiet) {
            console.log(chalk.cyan(`\n🔍 Reviewing: ${url}`));
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
