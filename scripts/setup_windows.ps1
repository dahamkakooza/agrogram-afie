# ============================================
# AFIE SERVICE - WINDOWS PRODUCTION SETUP
# ============================================

Write-Host "===========================================" -ForegroundColor Green
Write-Host "AFIE SERVICE - PRODUCTION SETUP (WINDOWS)" -ForegroundColor Green
Write-Host "===========================================" -ForegroundColor Green

# Check Python version
Write-Host "`n🔍 Checking Python version..." -ForegroundColor Cyan
try {
    $pythonVersion = python --version
    if ($pythonVersion -match "Python (\d+\.\d+)") {
        $version = $matches[1]
        if ([version]$version -ge [version]"3.10") {
            Write-Host "✅ Python $version (>= 3.10)" -ForegroundColor Green
        } else {
            Write-Host "❌ Python 3.10 or higher required. Found: $version" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "❌ Could not parse Python version: $pythonVersion" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ Python not found. Please install Python 3.10 or higher." -ForegroundColor Red
    exit 1
}

# Create and activate virtual environment
Write-Host "`n📦 Setting up virtual environment..." -ForegroundColor Cyan
if (!(Test-Path "venv")) {
    python -m venv venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✅ Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "`n🚀 Activating virtual environment..." -ForegroundColor Cyan
if (Test-Path "venv\Scripts\Activate.ps1") {
    .\venv\Scripts\Activate.ps1
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "❌ Could not activate virtual environment. Activating manually..." -ForegroundColor Yellow
    $env:VIRTUAL_ENV = "$PWD\venv"
    $env:PATH = "$PWD\venv\Scripts;$env:PATH"
}

# Upgrade pip and setuptools
Write-Host "`n⬆️  Upgrading pip and setuptools..." -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
Write-Host "`n📥 Installing production dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt

# Install development dependencies (if specified)
if ($args -contains "--dev") {
    Write-Host "`n🔧 Installing development dependencies..." -ForegroundColor Cyan
    pip install -r requirements-dev.txt
    if (Get-Command pre-commit -ErrorAction SilentlyContinue) {
        pre-commit install
        Write-Host "✅ Pre-commit hooks installed" -ForegroundColor Green
    }
    Write-Host "✅ Development dependencies installed" -ForegroundColor Green
}

# Setup environment
Write-Host "`n⚙️  Setting up environment..." -ForegroundColor Cyan
if (!(Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item .env.example .env
        Write-Host "⚠️  Created .env file from template. Please update with your configuration." -ForegroundColor Yellow
    } else {
        Write-Host "⚠️  No .env.example found. Creating minimal .env file..." -ForegroundColor Yellow
        @"
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
"@ | Out-File -FilePath .env -Encoding UTF8
        Write-Host "✅ Created .env file" -ForegroundColor Green
    }
} else {
    Write-Host "✅ .env file already exists" -ForegroundColor Green
}

# Check PostgreSQL
Write-Host "`n🗄️  Checking PostgreSQL..." -ForegroundColor Cyan
try {
    # Check if PostgreSQL service exists
    $pgService = Get-Service -Name postgresql* -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($pgService -and ($pgService.Status -eq "Running")) {
        Write-Host "✅ PostgreSQL is running" -ForegroundColor Green
        
        Write-Host "`n📝 Note: Manual database setup required:" -ForegroundColor Yellow
        Write-Host "   1. Open pgAdmin or PostgreSQL shell" -ForegroundColor Yellow
        Write-Host "   2. Create database: CREATE DATABASE afie_db;" -ForegroundColor Yellow
        Write-Host "   3. Create user: CREATE USER afie_user WITH PASSWORD 'afie_password';" -ForegroundColor Yellow
        Write-Host "   4. Grant privileges: GRANT ALL PRIVILEGES ON DATABASE afie_db TO afie_user;" -ForegroundColor Yellow
        Write-Host "   5. Install extensions: CREATE EXTENSION IF NOT EXISTS 'uuid-ossp';" -ForegroundColor Yellow
    } else {
        Write-Host "⚠️  PostgreSQL is not running or not installed" -ForegroundColor Yellow
        Write-Host "   Download PostgreSQL from: https://www.postgresql.org/download/windows/" -ForegroundColor Yellow
        Write-Host "   Or use Docker: docker run --name afie-postgres -e POSTGRES_PASSWORD=afie_password -p 5432:5432 -d postgres" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Could not check PostgreSQL status: $_" -ForegroundColor Yellow
}

# Create database tables
Write-Host "`n🚀 Creating database tables..." -ForegroundColor Cyan
try {
    $setupScript = @"
import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

async def setup_database():
    try:
        print("🔧 Setting up database...")
        from src.data.database.base import create_tables
        await create_tables()
        print("✅ Database tables created successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Failed to create tables: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(setup_database())
    sys.exit(0 if success else 1)
"@
    
    # Save script to temporary file
    $tempScript = "setup_database_temp.py"
    $setupScript | Out-File -FilePath $tempScript -Encoding UTF8
    
    # Run the script
    python $tempScript
    
    # Clean up
    Remove-Item $tempScript -ErrorAction SilentlyContinue
    
} catch {
    Write-Host "❌ Failed to setup database: $_" -ForegroundColor Red
}

# Run tests if in dev mode
if ($args -contains "--dev") {
    Write-Host "`n🧪 Running tests..." -ForegroundColor Cyan
    try {
        if (Get-Command pytest -ErrorAction SilentlyContinue) {
            pytest tests/unit/ -v --tb=short
            Write-Host "✅ Tests completed" -ForegroundColor Green
        } else {
            Write-Host "⚠️  pytest not found. Skipping tests." -ForegroundColor Yellow
        }
    } catch {
        Write-Host "⚠️  Tests failed or encountered error: $_" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "===========================================" -ForegroundColor Green
Write-Host "🎉 SETUP COMPLETE!" -ForegroundColor Green
Write-Host "===========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To start the AFIE service:" -ForegroundColor White
Write-Host ""
Write-Host "1. Activate virtual environment (if not already):" -ForegroundColor Gray
Write-Host "   .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Start the service:" -ForegroundColor Gray
Write-Host "   uvicorn src.main:app --reload" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Or run with production settings:" -ForegroundColor Gray
Write-Host "   `$env:ENVIRONMENT='production'" -ForegroundColor Gray
Write-Host "   uvicorn src.main:app --host 0.0.0.0 --port 8000" -ForegroundColor Gray
Write-Host ""
Write-Host "📡 API URL: http://localhost:8000" -ForegroundColor Cyan
Write-Host "📚 Docs URL: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "🏥 Health URL: http://localhost:8000/health" -ForegroundColor Cyan
Write-Host ""
Write-Host "For troubleshooting:" -ForegroundColor Yellow
Write-Host "   - Check .env file configuration" -ForegroundColor Yellow
Write-Host "   - Ensure PostgreSQL is running" -ForegroundColor Yellow
Write-Host "   - Check logs in 'logs/' directory" -ForegroundColor Yellow
Write-Host ""
Write-Host "===========================================" -ForegroundColor Green