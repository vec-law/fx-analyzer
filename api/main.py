from fastapi import FastAPI
from api.routers import auth
import os

app = FastAPI()

app.include_router(auth.router)
