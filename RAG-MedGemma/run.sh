#!/bin/bash

set -e

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Sync dependencies
echo "Syncing dependencies with uv..."
uv sync

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Run the application
echo "Running application..."
python src/main.py
