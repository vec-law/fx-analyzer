from pydantic import BaseModel
from fastapi import APIRouter

class LoginRequest(BaseModel):
    user_name: str
    password: str

router = APIRouter()

@router.post("/auth/login")
def login(request: LoginRequest):
    pass
