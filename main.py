from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Define the structure of the request body
class Item(BaseModel):
    message: str

@app.post("/echo")
def echo_data(item: Item):
    return {"you_sent": item}


