from fastapi import FastAPI
from contextlib import asynccontextmanager
from api.dependencies import db_manager
from api.routers import auth
from api.routers import users
from api.routers import trainings

@asynccontextmanager
async def lifespan(app: FastAPI):
    db_manager.ensure_admin()
    yield

app = FastAPI(lifespan=lifespan)

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(trainings.router)
