declare module '../../../../src/utils/processor' {
    export class Processor {
      static create(config: any): Promise<Processor>;
      processStream(input: any, config: any): Promise<any>;
    }
  }
  
  declare module '../../../../src/utils/llmConfig' {
    export function getLLMConfig(provider: string): any;
  }