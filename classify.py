import os
import pandas as pd
import requests
from dotenv import load_dotenv

print("Script started")

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
print("API key loaded:", "YES" if api_key else "NO - MISSING")

df = pd.read_csv("IMDB Dataset.csv")
sample = df.sample(20, random_state=42)
print("Sample size:", len(sample))

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
    print("Status code:", response.status_code)
    result = response.json()

    if "choices" not in result:
        print("ERROR RESPONSE:", result)
        return "error"

    return result["choices"][0]["message"]["content"].strip().lower()

correct = 0
total = 0

print("Starting loop...")
for index, row in sample.iterrows():
    predicted = classify_review(row["review"])
    actual = row["sentiment"]
    is_correct = predicted == actual
    correct += is_correct
    total += 1
    print(f"Predicted: {predicted:10} | Actual: {actual:10} | {'CORRECT' if is_correct else 'WRONG'}")

accuracy = (correct / total) * 100
print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")