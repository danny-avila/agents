import type * as t from './types';
import { getDomainName, fileExtRegex } from './utils';

/** Default per-search budget for model-facing highlight content (chars). Hosts
 *  that know the context window (e.g. LibreChat) pass a window-relative value;
 *  this fixed fallback keeps standalone consumers bounded instead of dumping the
 *  full reranked content of every source into the prompt. */
const DEFAULT_MAX_LLM_OUTPUT_CHARS = 50000;

/** Minimum room (chars) worth filling with a truncated boundary highlight; below
 *  this we drop it whole rather than emit a useless sliver. */
const MIN_PARTIAL_HIGHLIGHT_CHARS = 200;

/** Resolves the per-search highlight budget from config, the
 *  `SEARCH_MAX_LLM_OUTPUT_CHARS` env var, or the default (50,000 chars). */
export function resolveMaxLLMOutputChars(maxOutputChars?: number): number {
  if (maxOutputChars != null && maxOutputChars > 0) {
    return maxOutputChars;
  }
  const envValue = Number(process.env.SEARCH_MAX_LLM_OUTPUT_CHARS);
  if (Number.isFinite(envValue) && envValue > 0) {
    return envValue;
  }
  return DEFAULT_MAX_LLM_OUTPUT_CHARS;
}

/** Bounds the highlight chunks — the dominant, unbounded part of search output —
 *  to `maxChars`, walking sources in relevance order (organic first, then news;
 *  highlights in their reranked order). Whole highlights are kept until the
 *  budget is hit, the boundary one is truncated if meaningful room remains, and
 *  the rest are dropped. Snippets/titles/URLs are left untouched (small,
 *  high-signal) and the full content stays in the `WEB_SEARCH` artifact for
 *  citations. Mutates `results` in place; returns how many highlights were
 *  dropped or truncated (0 when everything fit). */
function trimHighlightsToBudget(results: t.SearchResultData, maxChars: number): number {
  let used = 0;
  let trimmed = 0;
  const sections: (t.ValidSource[] | undefined)[] = [results.organic, results.topStories];
  for (const sources of sections) {
    if (sources == null) {
      continue;
    }
    for (const source of sources) {
      const highlights = source.highlights;
      if (highlights == null || highlights.length === 0) {
        continue;
      }
      const kept: t.Highlight[] = [];
      for (const highlight of highlights) {
        const length = highlight.text.length;
        if (used + length <= maxChars) {
          kept.push(highlight);
          used += length;
          continue;
        }
        const remaining = maxChars - used;
        if (remaining >= MIN_PARTIAL_HIGHLIGHT_CHARS) {
          kept.push({ ...highlight, text: `${highlight.text.slice(0, remaining)}\n…[truncated]` });
          used = maxChars;
        }
        trimmed++;
      }
      source.highlights = kept;
    }
  }
  return trimmed;
}

function addHighlightSection(): string[] {
  return ['\n## Highlights', ''];
}

// Helper function to format a source (organic or top story)
function formatSource(
  source: t.ValidSource,
  index: number,
  turn: number,
  sourceType: 'search' | 'news',
  references: t.ResultReference[]
): string {
  /** Array of all lines to include in the output */
  const outputLines: string[] = [];

  // Add the title
  outputLines.push(
    `# ${sourceType.charAt(0).toUpperCase() + sourceType.slice(1)} ${index}: ${source.title != null && source.title ? `"${source.title}"` : '(no title)'}`
  );
  outputLines.push(`\nAnchor: \\ue202turn${turn}${sourceType}${index}`);
  outputLines.push(`URL: ${source.link}`);

  // Add optional fields
  if ('snippet' in source && source.snippet != null) {
    outputLines.push(`Summary: ${source.snippet}`);
  }

  if (source.date != null) {
    outputLines.push(`Date: ${source.date}`);
  }

  if (source.attribution != null) {
    outputLines.push(`Source: ${source.attribution}`);
  }

  // Add highlight section or empty line
  if ((source.highlights?.length ?? 0) > 0) {
    outputLines.push(...addHighlightSection());
  } else {
    outputLines.push('');
  }

  // Process highlights if they exist
  (source.highlights ?? [])
    .filter((h) => h.text.trim().length > 0)
    .forEach((h, hIndex) => {
      outputLines.push(
        `### Highlight ${hIndex + 1} [Relevance: ${h.score.toFixed(2)}]`
      );
      outputLines.push('');
      outputLines.push('```text');
      outputLines.push(h.text.trim());
      outputLines.push('```');
      outputLines.push('');

      if (h.references != null && h.references.length) {
        let hasHeader = false;
        const refLines: string[] = [];

        for (let j = 0; j < h.references.length; j++) {
          const ref = h.references[j];
          if (ref.reference.originalUrl.includes('mailto:')) {
            continue;
          }
          if (ref.type !== 'link') {
            continue;
          }
          if (fileExtRegex.test(ref.reference.originalUrl)) {
            continue;
          }
          references.push({
            type: ref.type,
            link: ref.reference.originalUrl,
            attribution: getDomainName(ref.reference.originalUrl),
            title: (
              ((ref.reference.title ?? '') || ref.reference.text) ??
              ''
            ).split('\n')[0],
          });

          if (!hasHeader) {
            refLines.push('Core References:');
            hasHeader = true;
          }

          refLines.push(
            `- ${ref.type}#${ref.originalIndex + 1}: ${ref.reference.originalUrl}`
          );
          refLines.push(
            `\t- Anchor: \\ue202turn${turn}ref${references.length - 1}`
          );
        }

        if (hasHeader) {
          outputLines.push(...refLines);
          outputLines.push('');
        }
      }

      if (hIndex < (source.highlights?.length ?? 0) - 1) {
        outputLines.push('---');
        outputLines.push('');
      }
    });

  outputLines.push('');
  return outputLines.join('\n');
}

export function formatResultsForLLM(
  turn: number,
  results: t.SearchResultData,
  maxOutputChars?: number
): { output: string; references: t.ResultReference[] } {
  /** Bound highlight content to the per-search budget before formatting */
  const trimmedHighlights = trimHighlightsToBudget(
    results,
    resolveMaxLLMOutputChars(maxOutputChars)
  );

  /** Array to collect all output lines */
  const outputLines: string[] = [];

  const addSection = (title: string): void => {
    outputLines.push('');
    outputLines.push(`=== ${title} ===`);
    outputLines.push('');
  };

  const references: t.ResultReference[] = [];

  // Organic (web) results
  if (results.organic?.length != null && results.organic.length > 0) {
    addSection(`Web Results, Turn ${turn}`);
    for (let i = 0; i < results.organic.length; i++) {
      const r = results.organic[i];
      outputLines.push(formatSource(r, i, turn, 'search', references));
      delete results.organic[i].highlights;
    }
  }

  // Top stories (news)
  const topStories = results.topStories ?? [];
  if (topStories.length) {
    addSection('News Results');
    for (let i = 0; i < topStories.length; i++) {
      const r = topStories[i];
      outputLines.push(formatSource(r, i, turn, 'news', references));
      if (results.topStories?.[i]?.highlights) {
        delete results.topStories[i].highlights;
      }
    }
  }

  // // Images
  // const images = results.images ?? [];
  // if (images.length) {
  //   addSection('Image Results');
  //   const imageLines = images.map((img, i) => [
  //     `Anchor: \ue202turn0image${i}`,
  //     `Title: ${img.title ?? '(no title)'}`,
  //     `Image URL: ${img.imageUrl}`,
  //     ''
  //   ].join('\n'));
  //   outputLines.push(imageLines.join('\n'));
  // }

  // Knowledge Graph
  if (results.knowledgeGraph != null) {
    addSection('Knowledge Graph');
    const kgLines = [
      `**Title:** ${results.knowledgeGraph.title ?? '(no title)'}`,
      results.knowledgeGraph.type != null
        ? `**Type:** ${results.knowledgeGraph.type}`
        : '',
      results.knowledgeGraph.description != null
        ? `**Description:** ${results.knowledgeGraph.description}`
        : '',
      results.knowledgeGraph.descriptionSource != null
        ? `**Description Source:** ${results.knowledgeGraph.descriptionSource}`
        : '',
      results.knowledgeGraph.descriptionLink != null
        ? `**Description Link:** ${results.knowledgeGraph.descriptionLink}`
        : '',
      results.knowledgeGraph.imageUrl != null
        ? `**Image URL:** ${results.knowledgeGraph.imageUrl}`
        : '',
      results.knowledgeGraph.website != null
        ? `**Website:** ${results.knowledgeGraph.website}`
        : '',
      results.knowledgeGraph.attributes != null
        ? `**Attributes:**\n\`\`\`json\n${JSON.stringify(
          results.knowledgeGraph.attributes,
          null,
          2
        )}\n\`\`\``
        : '',
      '',
    ].filter(Boolean);

    outputLines.push(kgLines.join('\n\n'));
  }

  // Answer Box
  if (results.answerBox != null) {
    addSection('Answer Box');
    const abLines = [
      results.answerBox.title != null
        ? `**Title:** ${results.answerBox.title}`
        : '',
      results.answerBox.snippet != null
        ? `**Snippet:** ${results.answerBox.snippet}`
        : '',
      results.answerBox.snippetHighlighted != null
        ? `**Snippet Highlighted:** ${results.answerBox.snippetHighlighted
          .map((s) => `\`${s}\``)
          .join(' ')}`
        : '',
      results.answerBox.link != null
        ? `**Link:** ${results.answerBox.link}`
        : '',
      '',
    ].filter(Boolean);

    outputLines.push(abLines.join('\n\n'));
  }

  // People also ask
  const peopleAlsoAsk = results.peopleAlsoAsk ?? [];
  if (peopleAlsoAsk.length) {
    addSection('People Also Ask');

    const paaLines: string[] = [];
    peopleAlsoAsk.forEach((p, i) => {
      const questionLines = [
        `### Question ${i + 1}:`,
        `"${p.question}"`,
        `${p.snippet != null && p.snippet ? `Snippet: ${p.snippet}` : ''}`,
        `${p.title != null && p.title ? `Title: ${p.title}` : ''}`,
        `${p.link != null && p.link ? `Link: ${p.link}` : ''}`,
        '',
      ].filter(Boolean);

      paaLines.push(questionLines.join('\n\n'));
    });

    outputLines.push(paaLines.join(''));
  }

  let output = outputLines.join('\n').trim();
  if (trimmedHighlights > 0) {
    output += `\n\n_[${trimmedHighlights} additional highlight${
      trimmedHighlights === 1 ? '' : 's'
    } omitted to fit the context budget; the cited sources contain the full content.]_`;
  }
  return { output, references };
}
