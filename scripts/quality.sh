#!/bin/bash

# Run all quality checks
echo "🚀 Running all code quality checks..."

echo "1. Formatting with isort and Black..."
./scripts/format.sh

echo
echo "2. Running linting with flake8..."
./scripts/lint.sh

echo
echo "3. Running tests..."
cd backend && uv run pytest tests/ -v

echo
echo "✅ All quality checks complete!"