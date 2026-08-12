import type { StandardGraph } from './Graph';
import type * as t from '@/types';

export type GraphFactoryRequest =
  | { kind: 'standard'; input: t.StandardGraphInput }
  | { kind: 'multi-agent'; input: t.MultiAgentGraphInput };

export type GraphFactory = (request: GraphFactoryRequest) => StandardGraph;

export type GraphFactoryDependencies = {
  graphFactory: GraphFactory;
};
