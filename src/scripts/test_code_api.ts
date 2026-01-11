// src/scripts/test_code_api.ts
/**
 * Direct test of the Code API to verify session file persistence.
 * This bypasses the LLM and tests the API directly.
 *
 * Run with: npx ts-node -r dotenv/config src/scripts/test_code_api.ts
 */
import { config } from 'dotenv';
config();

import fetch, { RequestInit } from 'node-fetch';
import { HttpsProxyAgent } from 'https-proxy-agent';

const API_KEY = process.env.LIBRECHAT_CODE_API_KEY ?? '';
const BASE_URL =
  process.env.LIBRECHAT_CODE_BASEURL ?? 'https://api.librechat.ai/v1';
const PROXY = process.env.PROXY;

if (!API_KEY) {
  console.error('LIBRECHAT_CODE_API_KEY not set');
  process.exit(1);
}

interface FileRef {
  id: string;
  name: string;
  session_id?: string; // New: self-contained file reference
}

interface ExecResult {
  session_id: string;
  stdout: string;
  stderr: string;
  files?: FileRef[];
}

interface FileInfo {
  name: string;
  metadata: Record<string, string>;
}

async function makeRequest(
  endpoint: string,
  body?: Record<string, unknown>
): Promise<unknown> {
  const fetchOptions: RequestInit = {
    method: body ? 'POST' : 'GET',
    headers: {
      'Content-Type': 'application/json',
      'User-Agent': 'LibreChat/1.0',
      'X-API-Key': API_KEY,
    },
  };

  if (body) {
    fetchOptions.body = JSON.stringify(body);
  }

  if (PROXY) {
    fetchOptions.agent = new HttpsProxyAgent(PROXY);
  }

  console.log(`\n>>> ${body ? 'POST' : 'GET'} ${endpoint}`);
  if (body) {
    console.log('Body:', JSON.stringify(body, null, 2));
  }

  const response = await fetch(endpoint, fetchOptions);
  const result = await response.json();

  console.log(`<<< Response (${response.status}):`);
  console.log(JSON.stringify(result, null, 2));

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${JSON.stringify(result)}`);
  }

  return result;
}

async function testCodeAPI(): Promise<void> {
  console.log('='.repeat(60));
  console.log('TEST 1: Create a file');
  console.log('='.repeat(60));

  const createCode = `
import json

config = {
    "app_name": "TestApp",
    "version": "1.0.0",
    "debug": True
}

with open("/mnt/data/test_config.json", "w") as f:
    json.dump(config, f, indent=2)

with open("/mnt/data/test_config.json", "r") as f:
    print(f.read())
`;

  const result1 = (await makeRequest(`${BASE_URL}/exec`, {
    lang: 'py',
    code: createCode,
  })) as ExecResult;

  const sessionId = result1.session_id;
  const files = result1.files ?? [];

  console.log('\n--- Result Summary ---');
  console.log('session_id:', sessionId);
  console.log('files:', files);
  console.log('stdout:', result1.stdout);
  console.log('stderr:', result1.stderr);

  if (!sessionId || files.length === 0) {
    console.error('\n❌ No session_id or files returned!');
    return;
  }

  // Check if files now include session_id (new API feature)
  const hasSessionIdInFiles = files.some((f) => f.session_id != null);
  console.log('\n✅ Files include session_id:', hasSessionIdInFiles);

  console.log('\n' + '='.repeat(60));
  console.log(
    'TEST 2: Fetch files IMMEDIATELY (no delay - testing race condition fix)'
  );
  console.log('='.repeat(60));

  const filesResult = (await makeRequest(
    `${BASE_URL}/files/${sessionId}?detail=full`
  )) as FileInfo[];

  console.log('\n--- Files in session (detail=full) ---');
  for (const file of filesResult) {
    console.log('File:', file.name);
    console.log('  metadata:', file.metadata);
  }

  if (filesResult.length === 0) {
    console.log(
      '\n⚠️  Files endpoint returned empty - race condition may still exist'
    );
  } else {
    console.log('\n✅ Files available immediately!');
  }

  // Test new normalized detail level
  console.log('\n' + '='.repeat(60));
  console.log('TEST 2b: Fetch files with detail=normalized');
  console.log('='.repeat(60));

  const normalizedResult = (await makeRequest(
    `${BASE_URL}/files/${sessionId}?detail=normalized`
  )) as FileRef[];

  console.log('\n--- Files in session (detail=normalized) ---');
  console.log(JSON.stringify(normalizedResult, null, 2));

  console.log('\n' + '='.repeat(60));
  console.log(
    'TEST 3: Read file IMMEDIATELY using files from original response'
  );
  console.log('='.repeat(60));

  // Use files directly - if API returns session_id, use that; otherwise add it
  const fileReferences: FileRef[] = files.map((file) => ({
    session_id: file.session_id ?? sessionId,
    id: file.id,
    name: file.name,
  }));

  console.log(
    '\nFile references we will send:',
    JSON.stringify(fileReferences, null, 2)
  );

  const readCode = `
import json

with open("/mnt/data/test_config.json", "r") as f:
    config = json.load(f)
    print("Read config:")
    print(json.dumps(config, indent=2))
    print("Version:", config.get("version"))
`;

  const result2 = (await makeRequest(`${BASE_URL}/exec`, {
    lang: 'py',
    code: readCode,
    files: fileReferences,
  })) as ExecResult;

  console.log('\n--- Result Summary ---');
  console.log('stdout:', result2.stdout);
  console.log('stderr:', result2.stderr);

  if (result2.stderr && result2.stderr.includes('FileNotFoundError')) {
    console.log(
      '\n❌ File not found! The file reference format might be wrong.'
    );

    // Try alternative format - just session_id
    console.log('\n' + '='.repeat(60));
    console.log('TEST 4: Try with just session_id in request');
    console.log('='.repeat(60));

    const result3 = (await makeRequest(`${BASE_URL}/exec`, {
      lang: 'py',
      code: readCode,
      session_id: sessionId,
    })) as ExecResult;

    console.log('\n--- Result Summary ---');
    console.log('stdout:', result3.stdout);
    console.log('stderr:', result3.stderr);
  } else {
    console.log('\n✅ File read successfully!');
  }
}

testCodeAPI().catch((err) => {
  console.error('Error:', err);
  process.exit(1);
});
