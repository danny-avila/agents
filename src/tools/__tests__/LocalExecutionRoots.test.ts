import { getReadRoots, getWriteRoots } from '../local/LocalExecutionEngine';

describe('local execution workspace roots', () => {
  it('uses the current working directory boundary when config is omitted', () => {
    expect(getWriteRoots()).toEqual([process.cwd()]);
    expect(getReadRoots()).toEqual([process.cwd()]);
  });
});
