import os
from fastapi import FastAPI
from pydantic import BaseModel, field_validator
import requests
from dotenv import load_dotenv
from typing import List

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

app = FastAPI()

from pydantic import BaseModel, field_validator

class ReviewRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("text can't be empty")
        if len(value) > 5000:
            raise ValueError("text is too long, keep it under 5000 characters")
        return value

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

from typing import List

class BatchReviewRequest(BaseModel):
    texts: List[str]

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, value):
        if not value:
            raise ValueError("texts list can't be empty")
        if len(value) > 50:
            raise ValueError("max 50 reviews per batch request")
        return value


@app.post("/classify-batch")
def classify_batch(request: BatchReviewRequest):
    results = []
    for text in request.texts:
        sentiment = classify_review(text)
        results.append({"review": text, "sentiment": sentiment})
    return {"results": results}