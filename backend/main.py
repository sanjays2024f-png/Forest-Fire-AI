from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.database.db import init_db
from backend.utils.config import APP_NAME, APP_DEBUG, BACKEND_HOST, BACKEND_PORT, CORS_ORIGINS

app = FastAPI(
    title=APP_NAME,
    description="AI-Based Real-Time Forest Fire and Smoke Monitoring System using hybrid deep learning",
    version="1.0.0",
    debug=APP_DEBUG,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def startup_event() -> None:
    init_db()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=APP_DEBUG)
