import { defineConfig, globalIgnores } from "eslint/config";
import typescriptEslint from "@typescript-eslint/eslint-plugin";
import importX from "eslint-plugin-import-x";
import globals from "globals";
import tsParser from "@typescript-eslint/parser";
import js from "@eslint/js";

export default defineConfig([globalIgnores([
    "dist/**/*",
    "config/**/*",
    "routes/**/*",
    "**/*.js",
    "**/*.mjs",
    "src/proto/",
    "src/scripts/",
    "types/**/*",
    "./script_docs.ts",
    "**/*.spec.ts",
]), {
    extends: [
        js.configs.recommended,
        ...typescriptEslint.configs["flat/recommended"],
        importX.flatConfigs.errors,
        importX.flatConfigs.warnings,
        importX.flatConfigs.typescript,
    ],

    languageOptions: {
        globals: {
            ...globals.node,
        },

        parser: tsParser,
        ecmaVersion: 2021,
        sourceType: "module",

        parserOptions: {
            project: ["./tsconfig.json", "./scripts/tsconfig.json"],
        },
    },

    settings: {
        "import-x/resolver": {
            typescript: {
                alwaysTryTypes: true,
                project: "./tsconfig.json",
            },
        },
    },

    rules: {
        "no-trailing-spaces": "error",
        indent: ["error", 2],
        "linebreak-style": ["error", "unix"],
        quotes: ["error", "single"],
        semi: ["error", "always"],

        "no-multiple-empty-lines": ["error", {
            max: 1,
            maxEOF: 0,
        }],

        "no-console": "warn",
        "prefer-const": "error",

        "@typescript-eslint/no-unused-vars": ["error", {
            argsIgnorePattern: "^_",
            varsIgnorePattern: "^_",
            caughtErrorsIgnorePattern: "^_",
            destructuredArrayIgnorePattern: "^_"
        }],

        "@typescript-eslint/consistent-type-assertions": "error",
        "@typescript-eslint/explicit-function-return-type": "error",
        "@typescript-eslint/no-explicit-any": "error",
        "no-nested-ternary": "error",
        "@typescript-eslint/no-unnecessary-condition": "warn",
        "@typescript-eslint/strict-boolean-expressions": "warn",
        "no-useless-assignment": "warn",
        "preserve-caught-error": "warn",
    },
}, {
    files: ["src/stream.ts", "src/utils/logging.ts", "scripts/**/*.ts"],

    rules: {
        "no-console": "off",
    },
}, {
    files: ["**/*.test.ts", "**/*.spec.ts", "**/__tests__/**/*.ts"],

    rules: {
        "@typescript-eslint/no-require-imports": "off",
        "@typescript-eslint/explicit-function-return-type": "off",
    },
}]);
