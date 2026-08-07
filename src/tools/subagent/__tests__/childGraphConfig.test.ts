import type {
  AgentInputs,
  GraphEdge,
  GraphSubagentConfig,
  SubagentConfig,
  SubagentConfigEntry,
} from '@/types';
import {
  buildGraphChildInputs,
  resolveSubagentConfigEntries,
  resolveSubagentConfigs,
  validateGraphSubagentConfig,
} from '@/tools/subagent/childGraphConfig';
import { AgentContext } from '@/agents/AgentContext';
import { Providers } from '@/common';

const makeAgent = (agentId: string): AgentInputs => ({
  agentId,
  provider: Providers.OPENAI,
  instructions: `You are ${agentId}.`,
});

const makeGraphConfig = (): GraphSubagentConfig => ({
  kind: 'graph',
  type: 'research-team',
  name: 'Research Team',
  description: 'Coordinates parallel research and synthesis.',
  entryAgentId: 'coordinator',
  resultAgentId: 'synthesizer',
  agents: [
    makeAgent('coordinator'),
    makeAgent('left'),
    makeAgent('right'),
    makeAgent('synthesizer'),
  ],
  edges: [
    {
      from: 'coordinator',
      to: ['left', 'right'],
      edgeType: 'direct',
    },
    {
      from: ['left', 'right'],
      to: 'synthesizer',
      edgeType: 'direct',
    },
  ],
});

describe('graph subagent config', () => {
  it('accepts a single-entry, single-result convergent direct DAG', () => {
    const config = makeGraphConfig();
    const parentContext = AgentContext.fromConfig(makeAgent('parent'));

    expect(() => validateGraphSubagentConfig(config)).not.toThrow();
    expect(resolveSubagentConfigEntries([config], parentContext)).toEqual([
      config,
    ]);
  });

  it('accepts prompts in a chain and on a converged result transition', () => {
    const chain = makeGraphConfig();
    chain.agents.splice(2, 1);
    chain.edges = [
      {
        from: 'coordinator',
        to: 'left',
        edgeType: 'direct',
        prompt: 'Continue the chain.',
      },
      {
        from: 'left',
        to: 'synthesizer',
        edgeType: 'direct',
        prompt: 'Produce the result.',
        excludeResults: true,
      },
    ];
    const converged = makeGraphConfig();
    converged.edges[1].prompt = 'Synthesize the branch results.';

    expect(() => validateGraphSubagentConfig(chain)).not.toThrow();
    expect(() => validateGraphSubagentConfig(converged)).not.toThrow();
  });

  it('preserves legacy configs with host-defined metadata fields', () => {
    interface HostSubagentConfig extends SubagentConfig {
      kind: 'host-defined';
      agents: string[];
    }
    const config: HostSubagentConfig = {
      kind: 'host-defined',
      type: 'legacy-host',
      name: 'Legacy Host',
      description: 'Uses metadata owned by the host.',
      agents: ['host-metadata'],
      agentInputs: makeAgent('legacy-child'),
    };
    const parentContext = AgentContext.fromConfig(makeAgent('parent'));

    expect(resolveSubagentConfigs([config], parentContext)[0].agentInputs).toBe(
      config.agentInputs
    );
  });

  it('rejects graph topology fields on the explicit agent variant', () => {
    const config = {
      kind: 'agent',
      type: 'invalid-agent',
      name: 'Invalid Agent',
      description: 'Mixes variants.',
      agentInputs: makeAgent('child'),
      agents: [makeAgent('graph-member')],
    } as SubagentConfigEntry;
    const parentContext = AgentContext.fromConfig(makeAgent('parent'));

    expect(() => resolveSubagentConfigEntries([config], parentContext)).toThrow(
      /cannot define graph topology fields/
    );
  });

  it.each([
    [
      'implicit edge type',
      (config: GraphSubagentConfig) => {
        (config.edges[0] as GraphEdge).edgeType = undefined;
      },
      /edgeType to "direct"/,
    ],
    [
      'handoff edge',
      (config: GraphSubagentConfig) => {
        (config.edges[0] as GraphEdge).edgeType = 'handoff';
      },
      /edgeType to "direct"/,
    ],
    [
      'conditional edge',
      (config: GraphSubagentConfig) => {
        (config.edges[0] as GraphEdge).condition = () => true;
      },
      /cannot define a condition/,
    ],
    [
      'custom prompt key',
      (config: GraphSubagentConfig) => {
        (config.edges[0] as GraphEdge).promptKey = 'custom';
      },
      /cannot define promptKey/,
    ],
    [
      'invalid prompt',
      (config: GraphSubagentConfig) => {
        Reflect.set(config.edges[0], 'prompt', 123);
      },
      /prompt must be a string or function/,
    ],
    [
      'invalid excludeResults',
      (config: GraphSubagentConfig) => {
        Reflect.set(config.edges[0], 'excludeResults', 'yes');
      },
      /excludeResults must be a boolean/,
    ],
    [
      'excludeResults without a prompt',
      (config: GraphSubagentConfig) => {
        config.edges[0].excludeResults = true;
      },
      /cannot define excludeResults without a prompt/,
    ],
    [
      'a prompted fan-out edge',
      (config: GraphSubagentConfig) => {
        config.edges[0].prompt = 'Branch prompt';
      },
      /prompted edge.*exactly one destination/,
    ],
    [
      'a prompted branch from a fan-out source',
      (config: GraphSubagentConfig) => {
        config.edges.splice(
          0,
          1,
          {
            from: 'coordinator',
            to: 'left',
            edgeType: 'direct',
            prompt: 'Left branch prompt',
          },
          {
            from: 'coordinator',
            to: 'right',
            edgeType: 'direct',
          }
        );
      },
      /prompted edge.*must target resultAgentId "synthesizer"/,
    ],
    [
      'parallel intermediate prompts after a fan-out',
      (config: GraphSubagentConfig) => {
        config.agents.splice(
          3,
          0,
          makeAgent('left-result'),
          makeAgent('right-result')
        );
        config.edges.splice(
          1,
          1,
          {
            from: 'left',
            to: 'left-result',
            edgeType: 'direct',
            prompt: 'Left result prompt',
          },
          {
            from: 'right',
            to: 'right-result',
            edgeType: 'direct',
            prompt: 'Right result prompt',
          },
          {
            from: ['left-result', 'right-result'],
            to: 'synthesizer',
            edgeType: 'direct',
          }
        );
      },
      /prompted edge.*must target resultAgentId "synthesizer"/,
    ],
    [
      'unknown endpoint',
      (config: GraphSubagentConfig) => {
        config.edges[0].to = ['left', 'missing'];
      },
      /unknown agent "missing"/,
    ],
    [
      'duplicate member ID',
      (config: GraphSubagentConfig) => {
        config.agents.push(makeAgent('left'));
      },
      /duplicate agent ID "left"/,
    ],
    [
      'nested execution',
      (config: GraphSubagentConfig) => {
        (config as { allowNested?: boolean }).allowNested = true;
      },
      /cannot enable allowNested/,
    ],
    [
      'more than 32 members',
      (config: GraphSubagentConfig) => {
        config.agents = Array.from({ length: 33 }, (_, index) =>
          makeAgent(`member-${index}`)
        );
      },
      /cannot exceed 32 agents/,
    ],
    [
      'ambiguous fan-in',
      (config: GraphSubagentConfig) => {
        config.edges[1] = {
          from: 'left',
          to: 'synthesizer',
          edgeType: 'direct',
        };
        config.edges.push({
          from: 'right',
          to: 'synthesizer',
          edgeType: 'direct',
        });
      },
      /must use one edge with an array-valued from field/,
    ],
    [
      'second root',
      (config: GraphSubagentConfig) => {
        config.edges[0].to = 'left';
        config.edges[1].from = 'left';
      },
      /only root/,
    ],
    [
      'second sink',
      (config: GraphSubagentConfig) => {
        config.edges[1].from = 'left';
      },
      /only sink/,
    ],
  ])('rejects %s', (_name, mutate, expected) => {
    const config = makeGraphConfig();
    mutate(config);
    expect(() => validateGraphSubagentConfig(config)).toThrow(expected);
  });

  it.each([
    '__start__',
    '__end__',
    'messages',
    'agentMessages',
    'subagentResult',
    'member|branch',
    'member:branch',
  ])('rejects graph-runtime member ID %s', (agentId) => {
    const config = makeGraphConfig();
    config.agents[0].agentId = agentId;

    expect(() => validateGraphSubagentConfig(config)).toThrow(
      /is reserved by the graph runtime/
    );
  });

  it.each([
    [null, /must be an object/],
    [{ ...makeGraphConfig(), agents: undefined }, /agents must be an array/],
    [{ ...makeGraphConfig(), edges: [null] }, /edges must contain objects/],
    [{ ...makeGraphConfig(), maxTurns: 0 }, /positive safe integer/],
    [
      { ...makeGraphConfig(), maxTurns: Number.MAX_SAFE_INTEGER },
      /exceeds the safe graph recursion budget/,
    ],
    [
      { ...makeGraphConfig(), agentInputs: makeAgent('legacy') },
      /cannot define agentInputs or self/,
    ],
  ])('rejects malformed runtime graph config %#', (config, expected) => {
    expect(() => validateGraphSubagentConfig(config)).toThrow(expected);
  });

  it('rejects cycles even when entry and result remain unique', () => {
    const config: GraphSubagentConfig = {
      ...makeGraphConfig(),
      agents: [
        makeAgent('entry'),
        makeAgent('a'),
        makeAgent('b'),
        makeAgent('result'),
      ],
      entryAgentId: 'entry',
      resultAgentId: 'result',
      edges: [
        { from: ['entry', 'b'], to: 'a', edgeType: 'direct' },
        { from: 'a', to: 'b', edgeType: 'direct' },
        { from: 'b', to: 'result', edgeType: 'direct' },
      ],
    };

    expect(() => validateGraphSubagentConfig(config)).toThrow(/acyclic/);
  });

  it('prepares every member once without mutating host inputs', () => {
    const graphTool = { name: 'direct_tool' } as NonNullable<
      AgentInputs['graphTools']
    >[number];
    const nestedConfig = {
      type: 'nested',
      name: 'Nested',
      description: 'Nested worker',
      agentInputs: makeAgent('nested'),
    };
    const sourceAgent: AgentInputs = {
      ...makeAgent('coordinator'),
      toolDefinitions: [{ name: 'event_tool' }],
      graphTools: [graphTool],
      initialSummary: { text: 'old summary', tokenCount: 2 },
      discoveredTools: ['event_tool'],
      subagentConfigs: [nestedConfig],
      maxSubagentDepth: 9,
    };
    const config: GraphSubagentConfig = {
      ...makeGraphConfig(),
      agents: [sourceAgent],
      edges: [],
      entryAgentId: 'coordinator',
      resultAgentId: 'coordinator',
    };

    const prepared = buildGraphChildInputs(config, 3, true);

    expect(prepared).toHaveLength(1);
    expect(prepared[0]).toMatchObject({
      agentId: 'coordinator',
      toolDefinitions: [{ name: 'event_tool' }],
      graphTools: [graphTool],
    });
    expect(prepared[0].subagentConfigs).toBeUndefined();
    expect(prepared[0].maxSubagentDepth).toBeUndefined();
    expect(prepared[0].initialSummary).toBeUndefined();
    expect(prepared[0].discoveredTools).toBeUndefined();
    expect(sourceAgent.initialSummary).toEqual({
      text: 'old summary',
      tokenCount: 2,
    });
    expect(sourceAgent.maxSubagentDepth).toBe(9);
  });
});
