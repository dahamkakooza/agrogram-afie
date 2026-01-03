"""
Minimal working version of AFIE service.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

app = FastAPI(
    title="Adaptive Farm Intelligence Engine (AFIE) - Minimal",
    description="AI system for farm intelligence",
    version="1.0.0",
)

# Basic CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to add process time header."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "AFIE",
        "version": "1.0.0",
        "status": "operational",
        "message": "Minimal version running"
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/api/v1/farms/test")
async def test_farms():
    """Test farms endpoint."""
    return {
        "farms": [
            {"id": "test-1", "name": "Test Farm 1", "type": "smallholder"},
            {"id": "test-2", "name": "Test Farm 2", "type": "commercial"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)