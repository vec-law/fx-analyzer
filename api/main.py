from fastapi import FastAPI
from contextlib import asynccontextmanager
from api.routers import auth
from api.dependencies import db_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    db_manager.ensure_admin()
    yield

app = FastAPI(lifespan=lifespan)

app.include_router(auth.router)
