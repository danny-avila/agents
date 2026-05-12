import { readFileSync, writeFileSync } from 'node:fs';

const packageJsonPath = new URL('../package.json', import.meta.url);
const packageJson = JSON.parse(readFileSync(packageJsonPath, 'utf8'));

delete packageJson.scripts;

writeFileSync(packageJsonPath, `${JSON.stringify(packageJson, null, 2)}\n`);

console.log('Removed package scripts from publish manifest.');
