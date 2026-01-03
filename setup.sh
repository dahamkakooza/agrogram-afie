#!/bin/bash

# ============================================
# AFIE SERVICE - COMPLETE BASH SETUP
# ============================================

set -e

echo "==========================================="
echo "AFIE SERVICE - PRODUCTION SETUP"
echo "==========================================="

# Check if running in Git Bash/MINGW64
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    echo "🔍 Running on Windows (Git Bash/MINGW64)"
    WINDOWS=true
    VENV_ACTIVATE="source venv/Scripts/activate"
else
    echo "🔍 Running on Linux/Mac"
    WINDOWS=false
    VENV_ACTIVATE="source venv/bin/activate"
fi

# Check Python version
echo ""
echo "🔍 Checking Python version..."
python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.10"

if [ -z "$python_version" ]; then
    echo "❌ Python not found. Please install Python 3.10 or higher."
    exit 1
fi

# Compare versions
if printf '%s\n%s\n' "$required_version" "$python_version" | sort -V -C 2>/dev/null; then
    echo "✅ Python $python_version (>= $required_version)"
else
    echo "❌ Python 3.10 or higher required. Found: $python_version"
    exit 1
fi

# Create and activate virtual environment
echo ""
echo "📦 Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv 2>/dev/null || python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "🚀 Activating virtual environment..."
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
    echo "✅ Virtual environment activated (Windows)"
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated (Linux/Mac)"
else
    echo "❌ Could not find activate script"
    exit 1
fi

# Upgrade pip and setuptools
echo ""
echo "⬆️  Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# Fix requirements-dev.txt file
echo ""
echo "🔧 Fixing requirements file..."
if [ -f "requirements-dev.txt" ]; then
    # Fix alembic-verify version
    sed -i 's/alembic-verify==0\.4\.0/alembic-verify==0.1.4/g' requirements-dev.txt
    # Fix types-redis version
    sed -i 's/types-redis==4\.6\.0\.20240316/types-redis==4.6.0.20240409/g' requirements-dev.txt
    echo "✅ Fixed requirements-dev.txt"
fi

# Install production dependencies
echo ""
echo "📥 Installing production dependencies..."
pip install -r requirements.txt

# Install development dependencies
echo ""
echo "🔧 Installing development dependencies..."
pip install -r requirements-dev.txt

# Setup pre-commit if available
if command -v pre-commit &> /dev/null; then
    pre-commit install
    echo "✅ Pre-commit hooks installed"
fi

# Setup environment file
echo ""
echo "⚙️  Setting up environment file..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ Created .env file from template"
        echo "⚠️  Please update .env file with your configuration"
    else
        echo "📝 Creating minimal .env file..."
        cat > .env << 'EOF'
# AFIE Service Configuration
ENVIRONMENT=development
DEBUG=True

# Database
DATABASE_URL=postgresql://afie_user:afie_password@localhost:5432/afie_db

# Redis
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-secret-key-change-in-production-keep-it-secure

# Feature Flags
FEATURE_FARM_PROFILER=True
FEATURE_RECOMMENDATIONS=False
FEATURE_LEARNING_SYSTEM=False
FEATURE_PREDICTIONS=False
