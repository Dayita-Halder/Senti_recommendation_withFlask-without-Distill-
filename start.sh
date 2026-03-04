#!/bin/bash
# Railway startup script - ensures pickle files exist

echo "=== Checking for pickle files ==="
if [ ! -d "pickle" ]; then
    echo "Creating pickle directory..."
    mkdir -p pickle
fi

ls -lh pickle/ || echo "Pickle directory is empty"

# List all files in current directory
echo "=== Current directory contents ==="
ls -la

# Check git LFS
if command -v git-lfs &> /dev/null; then
    echo "=== Git LFS status ==="
    git lfs ls-files
fi

# Start the app
echo "=== Starting application ==="
exec gunicorn -w 4 -b 0.0.0.0:$PORT app:app
