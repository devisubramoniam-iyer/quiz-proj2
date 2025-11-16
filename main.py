from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

app = FastAPI()

# Read secret from environment variable
SECRET = os.getenv("SECRET_KEY")

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str


@app.post("/")
def handle_quiz(data: QuizRequest):

    # If environment var missing → throw error
    if SECRET is None:
        raise HTTPException(
            status_code=500,
            detail="Server misconfiguration: SECRET_KEY not set."
        )

    # Check secret
    if data.secret != SECRET:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid secret")

    return {
        "message": "Request received successfully",
        "email": data.email,
        "url": data.url
    }
