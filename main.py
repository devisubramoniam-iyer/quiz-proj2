from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Define the structure of the request body
class Item(BaseModel):
    {
  "email": "your email",
  "secret": "your secret",
  "url": "https://tds-llm-analysis.s-anand.net/demo"
}

@app.post("/echo")
def echo_data(item: Item):
    return {"you_sent": item}



