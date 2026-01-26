/**
 * API key management - handles env vars, CLI flags, and interactive prompts
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
