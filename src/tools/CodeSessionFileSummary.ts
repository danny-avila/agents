import type * as t from '@/types';

const IMAGE_FILE_EXTENSIONS = new Set([
  '.avif',
  '.bmp',
  '.gif',
  '.ico',
  '.jpeg',
  '.jpg',
  '.png',
  '.tif',
  '.tiff',
  '.webp',
]);

const CODE_SESSION_FILE_SUMMARY_PATTERN =
  /^Generated files:\nSession files: \d+ persisted file\(s\) are available in \/mnt\/data, including \d+ image\(s\)\. Use known \/mnt\/data paths directly in later code-tool calls\. The app displays files\/images automatically; do not invent download links or wrap generated images in Markdown\.$/;

function getFileExtension(name: string): string {
  const lastSlash = name.lastIndexOf('/');
  const basename = lastSlash >= 0 ? name.slice(lastSlash + 1) : name;
  const lastDot = basename.lastIndexOf('.');
  return lastDot >= 0 ? basename.slice(lastDot).toLowerCase() : '';
}

function isImageFile(file: Partial<t.FileRef> | null | undefined): boolean {
  const name = file?.name;
  return (
    typeof name === 'string' &&
    IMAGE_FILE_EXTENSIONS.has(getFileExtension(name))
  );
}

function buildCodeSessionFileSummary(
  fileCount: number,
  imageCount: number
): string {
  return (
    'Generated files:\n' +
    `Session files: ${fileCount} persisted file(s) are available in /mnt/data, including ${imageCount} image(s). ` +
    'Use known /mnt/data paths directly in later code-tool calls. ' +
    'The app displays files/images automatically; do not invent download links or wrap generated images in Markdown.'
  );
}

function isGeneratedFile(file: Partial<t.FileRef> | null | undefined): boolean {
  return file?.inherited !== true;
}

export function stripCodeSessionFileSummary(output: string): string {
  const summaryStart = output.lastIndexOf('Generated files:');
  if (summaryStart < 0) return output;
  const beforeSummary = output.slice(0, summaryStart);
  if (beforeSummary !== '' && !beforeSummary.endsWith('\n\n')) return output;
  const maybeSummary = output.slice(summaryStart);
  if (!CODE_SESSION_FILE_SUMMARY_PATTERN.test(maybeSummary)) return output;
  return beforeSummary.trimEnd();
}

export function appendCodeSessionFileSummary(
  output: string,
  files: t.FileRefs | undefined
): string {
  if (files == null || files.length === 0) {
    return output.trim();
  }

  const generatedFiles = files.filter(isGeneratedFile);
  if (generatedFiles.length === 0) {
    return output.trim();
  }

  const imageCount = generatedFiles.filter(isImageFile).length;
  const summary = buildCodeSessionFileSummary(
    generatedFiles.length,
    imageCount
  );

  return `${output.trimEnd()}\n\n${summary}`.trim();
}
