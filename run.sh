#!/bin/bash

# Run python scripts
for file in notebooks/[0-9][0-9]*.py; do
    echo ""
    echo "Running $file"
    python "$file"
done