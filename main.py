import os
from fastapi import FastAPI
from pydantic import BaseModel
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

app = FastAPI()

class ReviewRequest(BaseModel):
    text: str

def classify_review(review_text):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "user", "content": f"Classify this review as positive or negative. Reply with only one word, lowercase.\n\nReview: {review_text}"}
        ],
        "temperature": 0
    }
    response = requests.post(url, headers=headers, json=payload, timeout=15)
    result = response.json()
    return result["choices"][0]["message"]["content"].strip().lower()

@app.post("/classify")
def classify(request: ReviewRequest):
    sentiment = classify_review(request.text)
    return {"review": request.text, "sentiment": sentiment}