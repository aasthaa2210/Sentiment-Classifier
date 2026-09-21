import os
import time
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

df = pd.read_csv("IMDB Dataset.csv")
sample = df.sample(20, random_state=42)


def classify_review(review_text, max_retries=3):
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

    # groq sometimes times out or errors randomly, so retrying a few times
    # before giving up (learned this the hard way lol)
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            result = response.json()

            if "choices" in result:
                return result["choices"][0]["message"]["content"].strip().lower()

            print("no choices in response, something went wrong:", result)

        except requests.exceptions.Timeout:
            print("timed out, attempt", attempt + 1)
        except requests.exceptions.ConnectionError:
            print("connection error, attempt", attempt + 1)
        except Exception as e:
            # catching anything else just in case, will figure out what later
            print("something broke:", e)

        if attempt < max_retries - 1:
            time.sleep(2)  # small wait before trying again

    return "error"  # giving up after max_retries, mark it as error for now


correct = 0
total = 0

for index, row in sample.iterrows():
    predicted = classify_review(row["review"])
    actual = row["sentiment"]
    is_correct = predicted == actual
    correct += is_correct
    total += 1
    print(f"Predicted: {predicted:10} | Actual: {actual:10} | {'CORRECT' if is_correct else 'WRONG'}")

accuracy = (correct / total) * 100
print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")