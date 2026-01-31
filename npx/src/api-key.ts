/**
 * API key and GitHub token management - handles env vars, CLI flags, and interactive prompts
 */

import inquirer from 'inquirer';
import chalk from 'chalk';

export async function getApiKey(cliApiKey?: string): Promise<string> {
    // 1. Check --api flag first (highest priority)
    if (cliApiKey) {
        return cliApiKey;
    }

    // 2. Check environment variable
    const envKey = process.env.GEMINI_API_KEY;
    if (envKey) {
        return envKey;
    }

    // 3. No API key found - prompt user
    console.log(chalk.yellow('\n⚠️  No Gemini API key found.\n'));
    console.log(chalk.dim('You can set it via:'));
    console.log(chalk.dim('  • --api <key> flag'));
    console.log(chalk.dim('  • GEMINI_API_KEY environment variable\n'));

    const answers = await inquirer.prompt([
        {
            type: 'password',
            name: 'apiKey',
            message: 'Enter your Gemini API key:',
            mask: '•',
            validate: (input: string) => {
                if (!input || input.trim().length === 0) {
                    return 'API key is required';
                }
                return true;
            },
        },
    ]);

    return answers.apiKey;
}

export async function getGitHubToken(cliToken?: string, requireToken: boolean = false): Promise<string> {
    // 1. Check --github-token flag first (highest priority)
    if (cliToken) {
        return cliToken;
    }

    // 2. Check environment variable
    const envToken = process.env.GITHUB_TOKEN;
    if (envToken) {
        return envToken;
    }

    // 3. If not required, return empty string (for public repos)
    if (!requireToken) {
        return '';
    }

    // 4. No token found but required - prompt user
    console.log(chalk.yellow('\n⚠️  No GitHub token found.\n'));
    console.log(chalk.dim('A GitHub token is required for private repositories.'));
    console.log(chalk.dim('You can set it via:'));
    console.log(chalk.dim('  • --github-token <token> flag'));
    console.log(chalk.dim('  • GITHUB_TOKEN environment variable\n'));

    const answers = await inquirer.prompt([
        {
            type: 'password',
            name: 'githubToken',
            message: 'Enter your GitHub token (or press Enter to skip):',
            mask: '•',
        },
    ]);

    return answers.githubToken || '';
}
