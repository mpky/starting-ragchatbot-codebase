#!/bin/bash

# Format Python code with Black and isort
echo "🎨 Formatting Python code..."

echo "Running isort..."
uv run isort backend/ main.py

echo "Running Black..."
uv run black backend/ main.py

echo "✅ Code formatting complete!"