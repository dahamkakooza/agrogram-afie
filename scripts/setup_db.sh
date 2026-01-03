#!/bin/bash

# Database setup script for AFIE service

set -e

echo "Setting up AFIE database..."

# Check if PostgreSQL is running
if ! pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
    echo "PostgreSQL is not running. Starting PostgreSQL..."
    # Start PostgreSQL (adjust based on your system)
    sudo service postgresql start || brew services start postgresql || echo "Please start PostgreSQL manually"
    sleep 5
fi

# Create database if it doesn't exist
echo "Creating database if it doesn't exist..."
psql -h localhost -p 5432 -U postgres -tc "SELECT 1 FROM pg_database WHERE datname = 'afie_db'" | grep -q 1 || \
psql -h localhost -p 5432 -U postgres -c "CREATE DATABASE afie_db"

# Create user if it doesn't exist
echo "Creating database user..."
psql -h localhost -p 5432 -U postgres -d afie_db -c "
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'afie_user') THEN
        CREATE USER afie_user WITH PASSWORD 'afie_password';
    END IF;
END
\$\$;"

# Grant privileges
echo "Granting privileges..."
psql -h localhost -p 5432 -U postgres -d afie_db -c "
GRANT ALL PRIVILEGES ON DATABASE afie_db TO afie_user;
GRANT CREATE ON DATABASE afie_db TO afie_user;
"

# Install required extensions
echo "Installing PostgreSQL extensions..."
psql -h localhost -p 5432 -U postgres -d afie_db -c "
CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";
CREATE EXTENSION IF NOT EXISTS \"pg_trgm\";
CREATE EXTENSION IF NOT EXISTS \"btree_gin\";
"

# Run migrations
echo "Running database migrations..."
cd "$(dirname "$0")/.."
alembic upgrade head

echo "Database setup complete!"