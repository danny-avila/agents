#!/bin/bash

# Script to convert all CRLF line endings to LF in the project

echo "Fixing line endings from CRLF to LF..."
echo

# Function to convert files
convert_files() {
    local pattern=$1
    local description=$2
    
    echo "Converting $description files..."
    find . -type f -name "$pattern" \
        -not -path "./node_modules/*" \
        -not -path "./dist/*" \
        -not -path "./.git/*" \
        -not -path "./coverage/*" \
        -not -path "./types/*" \
        -print0 | while IFS= read -r -d '' file; do
        # Check if file has CRLF endings
        if file "$file" | grep -q "CRLF"; then
            echo "  Converting: $file"
            # Convert CRLF to LF
            sed -i 's/\r$//' "$file"
        fi
    done
}

# Convert TypeScript and JavaScript files
convert_files "*.ts" "TypeScript"
convert_files "*.tsx" "TypeScript JSX"
convert_files "*.js" "JavaScript"
convert_files "*.jsx" "JavaScript JSX"
convert_files "*.mjs" "JavaScript module"

# Convert configuration files
convert_files "*.json" "JSON"
convert_files "*.yml" "YAML"
convert_files "*.yaml" "YAML"
convert_files ".eslintrc*" "ESLint config"
convert_files ".prettierrc*" "Prettier config"
convert_files "*.md" "Markdown"

# Convert other common files
convert_files "*.sh" "Shell script"
convert_files "Dockerfile*" "Docker"
convert_files ".env*" "Environment"
convert_files ".gitignore" "Git ignore"
convert_files "LICENSE" "License"

echo
echo "Line ending conversion complete!"

# Show statistics
echo
echo "Checking for any remaining CRLF files..."
remaining=$(find . -type f \( -name "*.ts" -o -name "*.tsx" -o -name "*.js" -o -name "*.jsx" -o -name "*.json" \) \
    -not -path "./node_modules/*" \
    -not -path "./dist/*" \
    -not -path "./.git/*" \
    -exec file {} \; | grep -c "CRLF")

if [ "$remaining" -eq 0 ]; then
    echo "✅ All files now use LF line endings!"
else
    echo "⚠️  Found $remaining files still with CRLF endings"
fi
