import { Tool } from '@langchain/core/tools';
import * as math from 'mathjs';

export class Calculator extends Tool {
  static lc_name(): string {
    return 'Calculator';
  }

  get lc_namespace(): string[] {
    return [...super.lc_namespace, 'calculator'];
  }

  name = 'calculator';

  async _call(input: string): Promise<string> {
    try {
      return math.evaluate(input).toString();
    } catch {
      return 'I don\'t know how to do that.';
    }
  }

  description =
    'Useful for getting the result of a math expression. The input to this tool should be a valid mathematical expression that could be executed by a simple calculator.';
}
