import { CallbackManagerForChainRun } from '@langchain/core/callbacks/manager';
import {
  mergeConfigs,
  patchConfig,
  Runnable,
  RunnableConfig,
} from '@langchain/core/runnables';
import { AsyncLocalStorageProviderSingleton } from '@langchain/core/singletons';

/**
 * Delays the execution for a specified number of milliseconds.
 *
 * @param {number} ms - The number of milliseconds to delay.
 * @return {Promise<void>} A promise that resolves after the specified delay.
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export interface RunnableCallableArgs extends Partial<any> {
  name?: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  func: (...args: any[]) => any;
  tags?: string[];
  trace?: boolean;
  recurse?: boolean;
}

export class RunnableCallable<I = unknown, O = unknown> extends Runnable<I, O> {
  lc_namespace: string[] = ['langgraph'];

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  func: (...args: any[]) => any;

  tags?: string[];

  config?: RunnableConfig;

  trace: boolean = true;

  recurse: boolean = true;

  constructor(fields: RunnableCallableArgs) {
    super();
    this.name = fields.name ?? fields.func.name;
    this.func = fields.func;
    this.config = fields.tags ? { tags: fields.tags } : undefined;
    this.trace = fields.trace ?? this.trace;
    this.recurse = fields.recurse ?? this.recurse;
  }

  protected async _tracedInvoke(
    input: I,
    config?: Partial<RunnableConfig>,
    runManager?: CallbackManagerForChainRun
  ): Promise<O> {
    return new Promise<O>((resolve, reject) => {
      let childConfig: Partial<RunnableConfig> | null = patchConfig(config, {
        callbacks: runManager?.getChild(),
      });
      void AsyncLocalStorageProviderSingleton.runWithConfig(
        childConfig,
        async () => {
          try {
            const output = await this.func(input, childConfig);
            childConfig = null;
            resolve(output);
          } catch (e) {
            childConfig = null;
            reject(e);
          }
        }
      );
    });
  }

  async invoke(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    input: any,
    options?: Partial<RunnableConfig> | undefined
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ): Promise<any> {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let returnValue: any;

    if (this.trace) {
      returnValue = await this._callWithConfig(
        this._tracedInvoke,
        input,
        mergeConfigs(this.config, options)
      );
    } else {
      returnValue = await this.func(input, mergeConfigs(this.config, options));
    }

    if (Runnable.isRunnable(returnValue) && this.recurse) {
      return await returnValue.invoke(input, options);
    }

    return returnValue;
  }
}
