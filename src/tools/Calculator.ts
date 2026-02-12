import { Tool } from '@langchain/core/tools';
import * as math from 'mathjs';

export const CalculatorToolName = 'calculator';

export const CalculatorToolDescription =
  'Useful for getting the result of a math expression. The input to this tool should be a valid mathematical expression that could be executed by a simple calculator.';

export const CalculatorSchema = {
  type: 'object',
  properties: {
    input: {
      type: 'string',
      description: 'A valid mathematical expression to evaluate',
    },
  },
  required: ['input'],
} as const;

export const CalculatorToolDefinition = {
  name: CalculatorToolName,
  description: CalculatorToolDescription,
  schema: CalculatorSchema,
} as const;

export class Calculator extends Tool {
  static lc_name(): string {
    return 'Calculator';
  }

  get lc_namespace(): string[] {
    return [...super.lc_namespace, 'calculator'];
  }

  name = CalculatorToolName;

  async _call(input: string): Promise<string> {
    try {
      return math.evaluate(input).toString();
    } catch {
      return 'I don\'t know how to do that.';
    }
  }

  description = CalculatorToolDescription;
}
