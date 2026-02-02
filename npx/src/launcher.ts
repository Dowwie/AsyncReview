#!/usr/bin/env node
/**
 * AsyncReview Runtime Launcher
 * 
 * This is the "thin wrapper" that:
 * 1. Detects the user's platform
 * 2. Downloads the runtime from GitHub Releases if not cached
 * 3. Executes the cached runtime
 */

import { spawn } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import * as https from 'https';
import { createGunzip } from 'zlib';
import { extract } from 'tar';
import { fileURLToPath } from 'url';

// ES module __dirname equivalent
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Read version from package.json
const PACKAGE_JSON = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'package.json'), 'utf-8'));
const VERSION = PACKAGE_JSON.version;

// GitHub repo for releases
const GITHUB_REPO = 'AsyncFuncAI/AsyncReview';

// Platform detection
function getPlatform(): string {
    const platform = os.platform();
    const arch = os.arch();

    let platformStr: string;
    if (platform === 'darwin') {
        platformStr = 'darwin';
    } else if (platform === 'linux') {
        platformStr = 'linux';
    } else if (platform === 'win32') {
        platformStr = 'windows';
    } else {
        throw new Error(`Unsupported platform: ${platform}`);
    }

    let archStr: string;
    if (arch === 'arm64' || arch === 'aarch64') {
        archStr = 'arm64';
    } else if (arch === 'x64' || arch === 'x86_64') {
        archStr = 'x64';
    } else {
        throw new Error(`Unsupported architecture: ${arch}`);
    }

    return `${platformStr}-${archStr}`;
}

// Get cache directory
function getCacheDir(): string {
    const home = os.homedir();
    if (os.platform() === 'darwin') {
        return path.join(home, 'Library', 'Caches', 'asyncreview', 'runtimes');
    } else if (os.platform() === 'win32') {
        return path.join(home, 'AppData', 'Local', 'asyncreview', 'runtimes');
    } else {
        // Linux and others use XDG
        const xdgCache = process.env.XDG_CACHE_HOME || path.join(home, '.cache');
        return path.join(xdgCache, 'asyncreview', 'runtimes');
    }
}

// Get runtime path
function getRuntimePath(version: string, platform: string): string {
    return path.join(getCacheDir(), version, platform);
}

// Get runtime entrypoint
function getRuntimeEntrypoint(runtimePath: string): string {
    return path.join(runtimePath, 'bin', 'asyncreview');
}

// Check if runtime is installed
function isRuntimeInstalled(runtimePath: string): boolean {
    const entrypoint = getRuntimeEntrypoint(runtimePath);
    return fs.existsSync(entrypoint);
}

// Download file with redirect following
function downloadFile(url: string, destPath: string): Promise<void> {
    return new Promise((resolve, reject) => {
        const file = fs.createWriteStream(destPath);

        const request = (url: string) => {
            https.get(url, (response) => {
                // Handle redirects
                if (response.statusCode === 301 || response.statusCode === 302) {
                    const redirectUrl = response.headers.location;
                    if (redirectUrl) {
                        request(redirectUrl);
                        return;
                    }
                }

                if (response.statusCode !== 200) {
                    reject(new Error(`Download failed: HTTP ${response.statusCode}`));
                    return;
                }

                response.pipe(file);
                file.on('finish', () => {
                    file.close();
                    resolve();
                });
            }).on('error', (err) => {
                fs.unlink(destPath, () => { });
                reject(err);
            });
        };

        request(url);
    });
}

// Extract tarball
async function extractTarball(tarPath: string, destDir: string): Promise<void> {
    fs.mkdirSync(destDir, { recursive: true });

    return new Promise((resolve, reject) => {
        fs.createReadStream(tarPath)
            .pipe(createGunzip())
            .pipe(extract({ cwd: destDir }))
            .on('finish', resolve)
            .on('error', reject);
    });
}

// Download and install runtime
async function installRuntime(version: string, platform: string): Promise<string> {
    const runtimePath = getRuntimePath(version, platform);
    const artifactName = `asyncreview-runtime-v${version}-${platform}.tar.gz`;
    const downloadUrl = `https://github.com/${GITHUB_REPO}/releases/download/v${version}/${artifactName}`;

    console.log(`⬇️  Downloading AsyncReview v${version} for ${platform}...`);
    console.log(`   From: ${downloadUrl}`);

    // Create temp directory
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'asyncreview-'));
    const tempTarPath = path.join(tempDir, artifactName);

    try {
        // Download
        await downloadFile(downloadUrl, tempTarPath);

        // Extract
        console.log('📦 Extracting...');
        await extractTarball(tempTarPath, runtimePath);

        // Make entrypoint executable
        const entrypoint = getRuntimeEntrypoint(runtimePath);
        fs.chmodSync(entrypoint, 0o755);

        console.log(`✅ Installed to: ${runtimePath}\n`);

        return runtimePath;
    } finally {
        // Clean up temp files
        fs.rmSync(tempDir, { recursive: true, force: true });
    }
}

// Main entry point
async function main(): Promise<void> {
    const platform = getPlatform();
    const runtimePath = getRuntimePath(VERSION, platform);

    // Install runtime if not present
    if (!isRuntimeInstalled(runtimePath)) {
        try {
            await installRuntime(VERSION, platform);
        } catch (error: any) {
            console.error(`\n❌ Failed to download AsyncReview runtime.`);
            console.error(`   ${error.message}`);
            console.error(`\nPlease check:`);
            console.error(`   • Your internet connection`);
            console.error(`   • The release exists: https://github.com/${GITHUB_REPO}/releases/tag/v${VERSION}`);
            console.error(`   • Your platform (${platform}) is supported`);
            process.exit(1);
        }
    }

    // Run the runtime with all arguments
    const entrypoint = getRuntimeEntrypoint(runtimePath);
    const args = process.argv.slice(2);

    const child = spawn(entrypoint, args, {
        stdio: 'inherit',
        env: {
            ...process.env,
            FORCE_COLOR: '1',  // Enable colors
        },
    });

    child.on('close', (code) => {
        process.exit(code ?? 0);
    });

    child.on('error', (err) => {
        console.error(`Failed to start runtime: ${err.message}`);
        process.exit(1);
    });
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
