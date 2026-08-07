import type {
  GraphFactory,
  GraphFactoryDependencies,
  GraphFactoryRequest,
} from '@/graphs/graphFactory';
import type { MultiAgentGraphInput, StandardGraphInput } from '@/types';
import { MultiAgentGraph } from '@/graphs/MultiAgentGraph';
import { StandardGraph } from '@/graphs/Graph';

const createGraphByKind: GraphFactory = (request) => {
  if (request.kind === 'multi-agent') {
    return new MultiAgentGraph(request.input, dependencies);
  }
  return new StandardGraph(request.input, dependencies);
};

const dependencies: GraphFactoryDependencies = {
  graphFactory: createGraphByKind,
};

export function createGraph(request: {
  kind: 'standard';
  input: StandardGraphInput;
}): StandardGraph;
export function createGraph(request: {
  kind: 'multi-agent';
  input: MultiAgentGraphInput;
}): MultiAgentGraph;
export function createGraph(request: GraphFactoryRequest): StandardGraph;
export function createGraph(request: GraphFactoryRequest): StandardGraph {
  return createGraphByKind(request);
}
