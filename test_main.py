from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app

client = TestClient(app)


def test_classify_rejects_empty_text():
    response = client.post("/classify", json={"text": ""})
    assert response.status_code == 422


def test_classify_rejects_too_long_text():
    long_text = "word " * 2000  # way over 5000 chars
    response = client.post("/classify", json={"text": long_text})
    assert response.status_code == 422


# this one fakes the groq api call, so we don't burn real credits every time we run tests
@patch("main.classify_review")
def test_classify_returns_sentiment(mock_classify):
    mock_classify.return_value = "positive"

    response = client.post("/classify", json={"text": "I loved this movie!"})

    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "positive"
    assert data["review"] == "I loved this movie!"


def test_classify_missing_text_field():
    response = client.post("/classify", json={})
    assert response.status_code == 422