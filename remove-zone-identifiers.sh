#!/bin/bash

# Script to remove all Zone.Identifier files

echo "Finding all Zone.Identifier files..."
echo

# Count the files
count=$(find . -type f -name "*:Zone.Identifier" 2>/dev/null | wc -l)

if [ $count -eq 0 ]; then
    echo "No Zone.Identifier files found."
    exit 0
fi

echo "Found $count Zone.Identifier files."
echo
echo "Files to be removed:"
echo "==================="

# List all files
find . -type f -name "*:Zone.Identifier" 2>/dev/null | sort

echo
echo "==================="
echo
read -p "Do you want to remove all these files? (y/N): " confirm

if [[ $confirm =~ ^[Yy]$ ]]; then
    echo
    echo "Removing Zone.Identifier files..."
    find . -type f -name "*:Zone.Identifier" -delete 2>/dev/null
    echo "Done! All Zone.Identifier files have been removed."
else
    echo "Operation cancelled."
fi
