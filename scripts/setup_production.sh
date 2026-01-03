#!/bin/bash
# COMPLETE PRODUCTION SETUP SCRIPT

set -e

echo "==========================================="
echo "AFIE SERVICE - PRODUCTION SETUP"
echo "==========================================="

# Check Python version
echo "🔍 Checking Python version..."
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
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
echo "📦 Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv 2>/dev/null || python -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

source venv/bin/activate

# Upgrade pip and setuptools
echo "⬆️  Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "📥 Installing production dependencies..."
pip install -r requirements.txt

# Install development dependencies (if in development)
if [ "$1" == "--dev" ]; then
    echo "🔧 Installing development dependencies..."
    pip install -r requirements-dev.txt
    pre-commit install
    echo "✅ Development dependencies installed"
fi

# Setup environment
echo "⚙️  Setting up environment..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "⚠️  Created .env file from template. Please update with your configuration."
    else
        echo "⚠️  No .env.example found. Creating minimal .env file..."
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
EOF
    fi
else
    echo "✅ .env file already exists"
fi

# Setup PostgreSQL (if available)
echo "🗄️  Setting up PostgreSQL..."
if command -v psql &> /dev/null; then
    # Check if PostgreSQL is running
    if pg_isready -h localhost -p 5432 &> /dev/null; then
        echo "✅ PostgreSQL is running"
        
        # Create database if it doesn't exist
        echo "🔧 Creating database..."
        psql -h localhost -p 5432 -U postgres -tc "SELECT 1 FROM pg_database WHERE datname = 'afie_db'" | grep -q 1 || \
        psql -h localhost -p 5432 -U postgres -c "CREATE DATABASE afie_db"
        
        # Create user if it doesn't exist
        echo "👤 Creating database user..."
        psql -h localhost -p 5432 -U postgres -d afie_db -c "
        DO \$\$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'afie_user') THEN
                CREATE USER afie_user WITH PASSWORD 'afie_password';
            END IF;
        END
        \$\$;"
        
        # Grant privileges
        echo "🔑 Granting privileges..."
        psql -h localhost -p 5432 -U postgres -d afie_db -c "
        GRANT ALL PRIVILEGES ON DATABASE afie_db TO afie_user;
        GRANT CREATE ON DATABASE afie_db TO afie_user;
        "
        
        # Install extensions
        echo "🔌 Installing PostgreSQL extensions..."
        psql -h localhost -p 5432 -U postgres -d afie_db -c "
        CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";
        CREATE EXTENSION IF NOT EXISTS \"pg_trgm\";
        CREATE EXTENSION IF NOT EXISTS \"btree_gin\";
        "
        
        echo "✅ PostgreSQL setup complete"
    else
        echo "⚠️  PostgreSQL is not running. Skipping database setup."
        echo "   To start PostgreSQL:"
        echo "   - Ubuntu: sudo service postgresql start"
        echo "   - macOS: brew services start postgresql"
        echo "   - Windows: Start PostgreSQL service from Services"
    fi
else
    echo "⚠️  PostgreSQL client not found. Skipping database setup."
    echo "   Install PostgreSQL client tools for your system."
fi

# Run database migrations or create tables
echo "🚀 Setting up database tables..."
if [ -f "alembic.ini" ] && command -v alembic &> /dev/null; then
    alembic upgrade head
    echo "✅ Database migrations applied"
else
    echo "🔧 Creating tables directly..."
    python -c "
import asyncio
import sys
import os
sys.path.insert(0, os.getcwd())

async def setup_tables():
    try:
        from src.data.database.base import create_tables
        await create_tables()
        print('✅ Database tables created')
    except Exception as e:
        print(f'❌ Failed to create tables: {e}')
        import traceback
        traceback.print_exc()

asyncio.run(setup_tables())
"
fi

# Run tests
echo "🧪 Running tests..."
if [ "$1" == "--dev" ]; then
    if command -v pytest &> /dev/null; then
        pytest tests/unit/ -v --tb=short || echo "⚠️  Some tests failed"
    else
        echo "⚠️  pytest not found. Skipping tests."
    fi
fi

echo ""
echo "==========================================="
echo "🎉 SETUP COMPLETE!"
echo "==========================================="
echo ""
echo "To start the AFIE service:"
echo ""
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Start the service:"
echo "   uvicorn src.main:app --reload"
echo ""
echo "3. Or run with production settings:"
echo "   export ENVIRONMENT=production"
echo "   uvicorn src.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "📡 API URL: http://localhost:8000"
echo "📚 Docs URL: http://localhost:8000/docs"
echo "🏥 Health URL: http://localhost:8000/health"
echo ""
echo "For Docker setup:"
echo "   docker-compose up -d"
echo ""
echo "==========================================="