#!/bin/bash

# Check if a file path is provided as an argument
if [ $# -eq 1 ]; then
    target_file=$1
    echo "Target file provided: $target_file"
else
    # Run ESLint and capture its output
    eslint_output=$(npm run lint 2>&1)

    # Extract the last mentioned file path from the ESLint output
    target_file=$(echo "$eslint_output" | grep -oP '/home/danny/agents/\K.*\.ts' | tail -n 1)

    if [ -z "$target_file" ]; then
        echo "No TypeScript files with issues found in the ESLint output."
        exit 1
    fi
    
    echo "Using last file from ESLint output: $target_file"
fi

# Check if the target file exists
if [ ! -f "$target_file" ]; then
    echo "File not found: $target_file"
    exit 1
fi

echo "Analyzing file: $target_file"

# Run the ESM error checker on the target file
js_output=$(node esm-error-checker.mjs "$target_file")

# Print the output
echo -e "$js_output"

# Save the output to a file
echo -e "$js_output" > lint_output.txt
echo "Output saved to lint_output.txt"

# Try to copy to clipboard, if xclip is available
if command -v xclip > /dev/null; then
    echo -e "$js_output" | xclip -selection clipboard
    echo "Output copied to clipboard."
fi
