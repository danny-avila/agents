// src/tools/ReadFile.ts
import { Constants } from '@/common';

export const ReadFileToolName = Constants.READ_FILE;

export const ReadFileToolDescription = `Read the contents of a file. Returns text content with line numbers for easy reference.

For skill files, use the path format: {skillName}/{filePath} (e.g. "pdf-analyzer/src/utils.py", "code-review/SKILL.md").

BEHAVIOR:
- Text files: returned with numbered lines.
- Images (png, jpeg, gif, webp): returned as visual content the model can see.
- PDFs: returned as document content.
- Other binary files: metadata returned with a note to use bash for processing.
- Large files (>256KB text, >10MB binary): metadata only.
- SKILL.md: returns the skill's instructions directly.

CONSTRAINTS:
- Only files from invoked skills or code execution output are accessible.
- Do not guess file paths. Use paths from the skill documentation or tool output.`;

export const ReadFileToolSchema = {
  type: 'object',
  properties: {
    file_path: {
      type: 'string',
      description:
        'Path to the file. For skill files: "{skillName}/{path}" (e.g. "pdf-analyzer/src/utils.py"). For code execution output: the path as returned by the execution tool.',
    },
  },
  required: ['file_path'],
} as const;

export const ReadFileToolDefinition = {
  name: ReadFileToolName,
  description: ReadFileToolDescription,
  parameters: ReadFileToolSchema,
  responseFormat: 'content_and_artifact' as const,
} as const;
