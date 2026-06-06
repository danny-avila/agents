/* eslint-disable no-console */

import { isAxiosError } from 'axios';

import type { AxiosError } from 'axios';
import type * as t from './types';

const LOG_VALUE_MAX_LENGTH = 2048;

export interface SafeErrorLog {
  message: string;
  name?: string;
  code?: string;
  status?: number;
  method?: string;
  url?: string;
  responseData?: string;
  value?: string;
}

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

const truncateLogValue = (value: string): string =>
  value.length <= LOG_VALUE_MAX_LENGTH
    ? value
    : `${value.slice(0, LOG_VALUE_MAX_LENGTH)}... [truncated]`;

const stringifyLogValue = (value: unknown): string => {
  if (typeof value === 'string') {
    return truncateLogValue(value);
  }
  if (
    typeof value === 'undefined' ||
    typeof value === 'function' ||
    typeof value === 'symbol'
  ) {
    return truncateLogValue(String(value));
  }

  try {
    return truncateLogValue(JSON.stringify(value));
  } catch {
    return truncateLogValue(String(value));
  }
};

const sanitizeUrlForLog = (url: string): string => {
  try {
    const parsed = new URL(url);
    return truncateLogValue(`${parsed.origin}${parsed.pathname}`);
  } catch {
    return truncateLogValue(url.split('?')[0].split('#')[0]);
  }
};

const formatAxiosErrorForLog = (
  error: AxiosError<unknown, unknown>
): SafeErrorLog => {
  const log: SafeErrorLog = {
    message: error.message,
    name: error.name,
  };
  const { code, config, response, status } = error;
  const responseStatus = response?.status ?? status;
  const method = config?.method;
  const url = config?.url;

  if (typeof code === 'string' && code !== '') {
    log.code = code;
  }
  if (responseStatus != null) {
    log.status = responseStatus;
  }
  if (typeof method === 'string' && method !== '') {
    log.method = method.toUpperCase();
  }
  if (typeof url === 'string' && url !== '') {
    log.url = sanitizeUrlForLog(url);
  }
  if (typeof response?.data !== 'undefined') {
    log.responseData = stringifyLogValue(response.data);
  }

  return log;
};

export const formatErrorForLog = (error: unknown): SafeErrorLog => {
  if (isAxiosError<unknown, unknown>(error)) {
    return formatAxiosErrorForLog(error);
  }
  if (error instanceof Error) {
    return {
      message: error.message,
      name: error.name,
    };
  }
  return {
    message: String(error),
    value: stringifyLogValue(error),
  };
};

export const fileExtRegex =
  /\.(pdf|jpe?g|png|gif|svg|webp|bmp|ico|tiff?|avif|heic|doc[xm]?|xls[xm]?|ppt[xm]?|zip|rar|mp[34]|mov|avi|wav)(?:\?.*)?$/i;

export const getDomainName = (
  link: string,
  metadata?: t.ScrapeMetadata | t.GenericScrapeMetadata,
  logger?: t.Logger
): string | undefined => {
  try {
    const sourceUrl =
      typeof metadata?.sourceURL === 'string' ? metadata.sourceURL : undefined;
    const metadataUrl =
      typeof metadata?.url === 'string' ? metadata.url : undefined;
    const url = sourceUrl ?? metadataUrl ?? link;
    const domain = new URL(url).hostname.replace(/^www\./, '');
    if (domain !== '') {
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
