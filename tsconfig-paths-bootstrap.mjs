import { pathToFileURL } from 'url';
// @ts-ignore
import { resolve as resolveTs } from 'ts-node/esm';
import * as tsConfigPaths from 'tsconfig-paths';

// @ts-ignore
const configResult = tsConfigPaths.loadConfig();
// `baseUrl` was removed from tsconfig for TypeScript 7 compatibility, so
// fall back to the project root with the relative `@/*` mapping when
// tsconfig-paths cannot derive a baseUrl on its own.
const absoluteBaseUrl =
  configResult.resultType === 'success'
    ? configResult.absoluteBaseUrl
    : process.cwd();
const paths =
  configResult.resultType === 'success'
    ? configResult.paths
    : { '@/*': ['./src/*'] };
const matchPath = tsConfigPaths.createMatchPath(absoluteBaseUrl, paths);

export function resolve(specifier, context, defaultResolve) {
  const match = matchPath(specifier);
  if (match) {
    return resolveTs(pathToFileURL(match).href, context, defaultResolve);
  }
  return resolveTs(specifier, context, defaultResolve);
}

// @ts-ignore
export { load, getFormat, transformSource } from 'ts-node/esm';
