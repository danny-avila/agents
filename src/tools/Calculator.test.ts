import { Calculator } from './Calculator';

describe('Calculator', () => {
  let calculator: Calculator;

  beforeEach(() => {
    calculator = new Calculator();
  });

  describe('basic properties', () => {
    it('should have correct name', () => {
      expect(calculator.name).toBe('calculator');
    });

    it('should have correct description', () => {
      expect(calculator.description).toBe(
        'Useful for getting the result of a math expression. The input to this tool should be a valid mathematical expression that could be executed by a simple calculator.'
      );
    });

    it('should have correct lc_name', () => {
      expect(Calculator.lc_name()).toBe('Calculator');
    });

    it('should have correct lc_namespace', () => {
      const namespace = calculator.lc_namespace;
      expect(namespace).toContain('calculator');
    });
  });

  describe('basic arithmetic operations', () => {
    it('should add two numbers', async () => {
      const result = await calculator._call('2 + 3');
      expect(result).toBe('5');
    });

    it('should subtract two numbers', async () => {
      const result = await calculator._call('10 - 4');
      expect(result).toBe('6');
    });

    it('should multiply two numbers', async () => {
      const result = await calculator._call('6 * 7');
      expect(result).toBe('42');
    });

    it('should divide two numbers', async () => {
      const result = await calculator._call('15 / 3');
      expect(result).toBe('5');
    });

    it('should handle modulo operation', async () => {
      const result = await calculator._call('17 % 5');
      expect(result).toBe('2');
    });

    it('should handle exponentiation', async () => {
      const result = await calculator._call('2 ^ 3');
      expect(result).toBe('8');
    });
  });

  describe('complex expressions', () => {
    it('should handle order of operations', async () => {
      const result = await calculator._call('2 + 3 * 4');
      expect(result).toBe('14');
    });

    it('should handle parentheses', async () => {
      const result = await calculator._call('(2 + 3) * 4');
      expect(result).toBe('20');
    });

    it('should handle nested parentheses', async () => {
      const result = await calculator._call('((2 + 3) * 4) / 5');
      expect(result).toBe('4');
    });

    it('should handle multiple operations', async () => {
      const result = await calculator._call('(10 + 5) * 2 - 8 / 4');
      expect(result).toBe('28');
    });
  });

  describe('decimal numbers', () => {
    it('should handle decimal addition', async () => {
      const result = await calculator._call('2.5 + 3.7');
      expect(result).toBe('6.2');
    });

    it('should handle decimal multiplication', async () => {
      const result = await calculator._call('2.5 * 4');
      expect(result).toBe('10');
    });

    it('should handle decimal division', async () => {
      const result = await calculator._call('7.5 / 2.5');
      expect(result).toBe('3');
    });
  });

  describe('negative numbers', () => {
    it('should handle negative numbers in addition', async () => {
      const result = await calculator._call('-5 + 3');
      expect(result).toBe('-2');
    });

    it('should handle negative numbers in subtraction', async () => {
      const result = await calculator._call('10 - (-5)');
      expect(result).toBe('15');
    });

    it('should handle negative numbers in multiplication', async () => {
      const result = await calculator._call('-4 * -3');
      expect(result).toBe('12');
    });

    it('should handle negative numbers in division', async () => {
      const result = await calculator._call('-12 / 4');
      expect(result).toBe('-3');
    });
  });

  describe('mathematical functions', () => {
    it('should handle square root', async () => {
      const result = await calculator._call('sqrt(16)');
      expect(result).toBe('4');
    });

    it('should handle absolute value', async () => {
      const result = await calculator._call('abs(-42)');
      expect(result).toBe('42');
    });

    it('should handle sine function', async () => {
      const result = await calculator._call('sin(0)');
      expect(result).toBe('0');
    });

    it('should handle cosine function', async () => {
      const result = await calculator._call('cos(0)');
      expect(result).toBe('1');
    });

    it('should handle logarithm', async () => {
      const result = await calculator._call('log(10)');
      expect(parseFloat(result)).toBeCloseTo(2.302585, 5);
    });

    it('should handle exponential', async () => {
      const result = await calculator._call('exp(0)');
      expect(result).toBe('1');
    });

    it('should handle floor function', async () => {
      const result = await calculator._call('floor(4.7)');
      expect(result).toBe('4');
    });

    it('should handle ceil function', async () => {
      const result = await calculator._call('ceil(4.3)');
      expect(result).toBe('5');
    });

    it('should handle round function', async () => {
      const result = await calculator._call('round(4.5)');
      expect(result).toBe('5');
    });
  });

  describe('constants', () => {
    it('should handle pi constant', async () => {
      const result = await calculator._call('pi');
      expect(parseFloat(result)).toBeCloseTo(3.14159, 5);
    });

    it('should handle e constant', async () => {
      const result = await calculator._call('e');
      expect(parseFloat(result)).toBeCloseTo(2.71828, 5);
    });

    it('should use pi in calculations', async () => {
      const result = await calculator._call('2 * pi');
      expect(parseFloat(result)).toBeCloseTo(6.28318, 4);
    });
  });

  describe('error handling', () => {
    it('should return error message for invalid expression', async () => {
      const result = await calculator._call('invalid expression');
      expect(result).toBe('I don\'t know how to do that.');
    });

    it('should handle division by zero', async () => {
      const result = await calculator._call('5 / 0');
      expect(result).toBe('Infinity');
    });

    it('should return error message for incomplete expression', async () => {
      const result = await calculator._call('5 +');
      expect(result).toBe('I don\'t know how to do that.');
    });

    it('should return error message for mismatched parentheses', async () => {
      const result = await calculator._call('(5 + 3');
      expect(result).toBe('I don\'t know how to do that.');
    });

    it('should return error message for empty input', async () => {
      const result = await calculator._call('');
      expect(result).toBe('I don\'t know how to do that.');
    });

    it('should return error message for only whitespace', async () => {
      const result = await calculator._call('   ');
      expect(result).toBe('I don\'t know how to do that.');
    });

    it('should return error message for undefined variable', async () => {
      const result = await calculator._call('x + 5');
      expect(result).toBe('I don\'t know how to do that.');
    });
  });

  describe('edge cases', () => {
    it('should handle very large numbers', async () => {
      const result = await calculator._call('999999999 * 999999999');
      expect(parseFloat(result)).toBeCloseTo(999999998000000000, -10);
    });

    it('should handle very small decimal numbers', async () => {
      const result = await calculator._call('0.0001 + 0.0002');
      expect(parseFloat(result)).toBeCloseTo(0.0003, 10);
    });

    it('should handle expressions with spaces', async () => {
      const result = await calculator._call('  5   +   3  ');
      expect(result).toBe('8');
    });

    it('should handle zero in calculations', async () => {
      const result = await calculator._call('0 + 0');
      expect(result).toBe('0');
    });

    it('should handle power of zero', async () => {
      const result = await calculator._call('5 ^ 0');
      expect(result).toBe('1');
    });

    it('should handle zero to the power', async () => {
      const result = await calculator._call('0 ^ 5');
      expect(result).toBe('0');
    });
  });

  describe('complex real-world calculations', () => {
    it('should calculate area of circle', async () => {
      const result = await calculator._call('pi * 5^2');
      expect(parseFloat(result)).toBeCloseTo(78.53981, 4);
    });

    it('should calculate compound interest formula component', async () => {
      const result = await calculator._call('(1 + 0.05/12)^(12*5)');
      expect(parseFloat(result)).toBeCloseTo(1.28336, 4);
    });

    it('should calculate percentage', async () => {
      const result = await calculator._call('(25 / 200) * 100');
      expect(result).toBe('12.5');
    });

    it('should calculate average', async () => {
      const result = await calculator._call('(10 + 20 + 30 + 40 + 50) / 5');
      expect(result).toBe('30');
    });
  });
});
