/* eslint-disable no-console */

import type * as t from './types';

/**
 * Singleton instance of the default logger
 */
let defaultLoggerInstance: t.Logger | null = null;

/**
 * Creates a default logger that maps to console methods
 * Uses a singleton pattern to avoid creating multiple instances
 * @returns A default logger that implements the Logger interface
 */
export const createDefaultLogger = (): t.Logger => {
  if (!defaultLoggerInstance) {
    defaultLoggerInstance = {
      error: console.error,
      warn: console.warn,
      info: console.info,
      debug: console.debug,
    } as t.Logger;
  }
  return defaultLoggerInstance;
};

export const fileExtRegex =
  /\.(pdf|jpe?g|png|gif|svg|webp|bmp|ico|tiff?|avif|heic|doc[xm]?|xls[xm]?|ppt[xm]?|zip|rar|mp[34]|mov|avi|wav)(?:\?.*)?$/i;

export const getDomainName = (
  link: string,
  metadata?: t.ScrapeMetadata | t.GenericScrapeMetadata,
  logger?: t.Logger
): string | undefined => {
  try {
    const sourceUrl = metadata?.sourceURL;
    const metadataUrl = metadata?.url;
    const url =
      typeof sourceUrl === 'string'
        ? sourceUrl
        : typeof metadataUrl === 'string'
          ? metadataUrl
          : link || '';
    const domain = new URL(url).hostname.replace(/^www\./, '');
    if (domain) {
      return domain;
    }
  } catch (e) {
    // URL parsing failed
    if (logger) {
      logger.error('Error parsing URL:', e);
    } else {
      console.error('Error parsing URL:', e);
    }
  }

  return;
};

export function getAttribution(
  link: string,
  metadata?: t.ScrapeMetadata | t.GenericScrapeMetadata,
  logger?: t.Logger
): string | undefined {
  if (!metadata) return getDomainName(link, metadata, logger);

  const twitterSite = metadata['twitter:site'];
  const twitterSiteFormatted =
    typeof twitterSite === 'string' ? twitterSite.replace(/^@/, '') : undefined;
  const title = metadata.title;

  const possibleAttributions = [
    metadata.ogSiteName,
    metadata['og:site_name'],
    typeof title === 'string' ? title.split('|').pop()?.trim() : undefined,
    twitterSiteFormatted,
  ];

  const attribution = possibleAttributions.find(
    (attr): attr is string => typeof attr === 'string' && attr.trim() !== ''
  );
  if (attribution != null) {
    return attribution;
  }

  return getDomainName(link, metadata, logger);
}
