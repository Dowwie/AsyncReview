/**
 * Python environment setup and runner
 */

import { spawn, spawnSync } from 'child_process';
import chalk from 'chalk';
import ora from 'ora';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
// From dist/ we need to go up to npx/
// Check if we are running from the bundled version (npx/python exists)
const NPX_ROOT = path.resolve(__dirname, '..');
const BUNDLED_PYTHON_ROOT = path.resolve(NPX_ROOT, 'python');

// If 'python' directory exists in npx root, use it. Otherwise assume we are in the repo structure.
import fs from 'fs';
const IS_BUNDLED = fs.existsSync(BUNDLED_PYTHON_ROOT);
const ASYNCREVIEW_ROOT = IS_BUNDLED ? BUNDLED_PYTHON_ROOT : path.resolve(NPX_ROOT, '..');

interface PythonCheckResult {
    available: boolean;
    version?: string;
    pythonCmd: string;
}

/**
 * Check if Python 3.11+ is available
 * Prioritizes virtual environment Python if it exists
 */
export function checkPython(): PythonCheckResult {
    // Check for local venv Python first (in npx/.venv)
    const npxDir = path.resolve(ASYNCREVIEW_ROOT, 'npx');
    const localVenvPython = path.join(npxDir, '.venv', 'bin', 'python');

    // Check if the local venv python exists and is executable
    try {
        if (spawnSync('test', ['-x', localVenvPython]).status === 0) {
            const result = spawnSync(localVenvPython, ['--version'], { encoding: 'utf-8' });
            if (result.status === 0) {
                return { available: true, version: result.stdout.trim(), pythonCmd: localVenvPython };
            }
        }
    } catch { }

    // Fallback to system python to bootstrap
    for (const cmd of ['python3', 'python']) {
        try {
            const result = spawnSync(cmd, ['--version'], { encoding: 'utf-8' });
            if (result.status === 0) {
                const version = result.stdout.trim() || result.stderr.trim();
                const match = version.match(/Python (\d+)\.(\d+)/);
                if (match) {
                    const major = parseInt(match[1]);
                    const minor = parseInt(match[2]);
                    if (major >= 3 && minor >= 11) {
                        return { available: true, version, pythonCmd: cmd };
                    }
                }
            }
        } catch {
            // Continue to next command
        }
    }
    return { available: false, pythonCmd: 'python3' };
}

/**
 * Check if asyncreview Python package is installed
 */
export function checkAsyncReviewInstalled(pythonCmd: string): boolean {
    // If we are using the local venv, check if it works
    if (pythonCmd.includes('.venv')) {
        try {
            // Check for both cli module and rich (as a proxy for dependencies)
            const result = spawnSync(pythonCmd, ['-c', 'import cli.main; import rich'], {
                encoding: 'utf-8',
                cwd: ASYNCREVIEW_ROOT,
            });
            return result.status === 0;
        } catch {
            return false;
        }
    }
    return false;
}

/**
 * Setup virtual environment and install dependencies
 */
/**
 * Setup virtual environment and install dependencies
 */
export async function installAsyncReview(systemPython: string, quiet: boolean = false): Promise<boolean> {
    const spinner = !quiet ? ora('Setting up isolated Python environment...').start() : null;
    const npxDir = path.resolve(ASYNCREVIEW_ROOT, 'npx');
    const venvDir = path.join(npxDir, '.venv');

    try {
        // 1. Create venv if it doesn't exist
        if (spinner) spinner.text = 'Creating virtual environment...';
        spawnSync(systemPython, ['-m', 'venv', venvDir]);

        const venvPip = path.join(venvDir, 'bin', 'pip');

        // 2. Install dependencies via pip
        if (spinner) spinner.text = 'Installing dependencies (this may take a minute)...';

        return new Promise((resolve) => {
            // If bundled, we install from '.', if dev, we install editable '-e .'
            // Actually, for bundled, we just want 'pip install .' to install dependencies and the package itself
            const installArgs = ['install', ASYNCREVIEW_ROOT];
            if (!IS_BUNDLED) {
                // In dev mode, use editable install
                installArgs.push('-e');
            }

            // We need to be careful with the path passed to pip install
            // If we pass ASYNCREVIEW_ROOT directly, it should work for both '.' and absolute paths

            // Refined args:
            const args = ['install'];
            if (!IS_BUNDLED) args.push('-e');
            args.push('.');

            const proc = spawn(venvPip, args, {
                cwd: ASYNCREVIEW_ROOT,
                stdio: ['ignore', 'pipe', 'pipe']
            });

            proc.on('close', (code) => {
                if (code === 0) {
                    if (spinner) spinner.succeed('AsyncReview environment ready');
                    resolve(true);
                } else {
                    if (spinner) spinner.fail('Failed to install dependencies');
                    resolve(false);
                }
            });

            proc.on('error', () => {
                if (spinner) spinner.fail('Failed to start installation');
                resolve(false);
            });
        });

    } catch (e) {
        if (spinner) spinner.fail('Failed to setup environment');
        return false;
    }
}

export interface RunOptions {
    url: string;
    question: string;
    output: string;
    quiet: boolean;
    model?: string;
    apiKey: string;
}

/**
 * Run the Python asyncreview CLI using the isolated environment
 */
export async function runPythonReview(options: RunOptions): Promise<string> {

    // 1. Ensure we have a valid environment
    let pythonCheck = checkPython();

    // If we don't have a venv python, or we have one but dependencies aren't there
    // We need to trigger installation using the system python we found
    if (!pythonCheck.available) throw new Error("Python 3.11+ required to set up environment");

    const npxDir = path.resolve(ASYNCREVIEW_ROOT, 'npx');
    const venvPython = path.join(npxDir, '.venv', 'bin', 'python');

    // Force usage of venv python for execution
    const pythonCmd = checkAsyncReviewInstalled(venvPython) ? venvPython : pythonCheck.pythonCmd;

    return new Promise((resolve, reject) => {
        const args = [
            '-m', 'cli.main',
            'review',
            '--url', options.url,
            '-q', options.question,
            '--output', options.output,
        ];

        if (options.quiet) {
            args.push('--quiet');
        }

        if (options.model) {
            args.push('--model', options.model);
        }

        // Add PYTHONPATH so Python can find the cli module
        const pythonPath = process.env.PYTHONPATH
            ? `${ASYNCREVIEW_ROOT}:${process.env.PYTHONPATH}`
            : ASYNCREVIEW_ROOT;

        const proc = spawn(pythonCmd, args, {
            cwd: ASYNCREVIEW_ROOT,
            env: {
                ...process.env,
                GEMINI_API_KEY: options.apiKey,
                PYTHONPATH: pythonPath,
                // Ensure we don't inherit conflicting python env vars
                VIRTUAL_ENV: path.dirname(path.dirname(pythonCmd))
            },
            stdio: ['pipe', 'pipe', 'pipe'],
        });

        let stdout = '';
        let stderr = '';

        proc.stdout.on('data', (data) => {
            stdout += data.toString();
            if (!options.quiet) {
                process.stdout.write(data);
            }
        });

        proc.stderr.on('data', (data) => {
            stderr += data.toString();
            if (!options.quiet) {
                process.stderr.write(data);
            }
        });

        proc.on('close', (code) => {
            if (code === 0) {
                resolve(stdout);
            } else {
                reject(new Error(stderr || `Process exited with code ${code}`));
            }
        });

        proc.on('error', (err) => {
            reject(err);
        });
    });
}
