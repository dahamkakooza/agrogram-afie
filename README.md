# Adaptive Farm Intelligence Engine (AFIE)

The brain of AgroGram - an AI system that learns each farm's unique "personality" and provides hyper-personalized recommendations that evolve with the farmer.

## Project Structure

This service follows a modular architecture with clear separation of concerns:

- `src/api/` - HTTP/WebSocket interfaces
- `src/core/` - Business logic and intelligence
- `src/data/` - Data processing and storage
- `src/ml/` - Machine learning models
- `src/models/` - Data schemas and structures
- `src/services/` - External integrations
- `src/utils/` - Shared utilities
- `src/config/` - Configuration management

## Quick Start

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables: `cp .env.example .env`
4. Run database migrations: `alembic upgrade head`
5. Start the service: `uvicorn src.main:app --reload`

## Development

See `docs/` for detailed architecture and API documentation.
