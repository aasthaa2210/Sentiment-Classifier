# SentiIQ — LLM-Powered Sentiment Classification API

A lightweight REST API that classifies text (e.g. movie or product reviews) as **positive** or **negative** sentiment, using a large language model instead of a traditionally trained ML classifier. Built to demonstrate practical LLM integration, prompt design, and model evaluation — with results measured against real labeled data, not assumed.

**Accuracy:** 95% (19/20) on a random sample of the IMDB 50K Movie Reviews dataset.

## Tech Stack

- **Python**
- **FastAPI** — REST API framework
- **Groq API** (`openai/gpt-oss-20b`) — LLM inference
- **pandas** — data loading and sampling
- **python-dotenv** — environment variable management

## How It Works

1. Loads labeled review data (IMDB Dataset of 50K Movie Reviews) as ground truth.
2. Sends each review to an LLM via the Groq API with a designed prompt instructing it to classify sentiment.
3. Compares the model's predictions against real human-labeled answers to calculate accuracy.
4. Exposes the classification logic as a live API endpoint (`POST /classify`) that any application can call with new, unseen text.

## Project Structure

```
api/
├── load_data.py      # Loads and previews the dataset
├── classify.py        # Runs classification + accuracy evaluation on a sample
├── main.py             # FastAPI app exposing the /classify endpoint
├── requirements.txt
├── .env                 # Groq API key (not committed)
└── .gitignore
```

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/sentiment-classifier-api.git
   cd sentiment-classifier-api/api
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\Activate.ps1
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your Groq API key**
   Create a `.env` file in the `api` folder:
   ```
   GROQ_API_KEY=your_key_here
   ```
   Get a free key at [console.groq.com](https://console.groq.com).

5. **Download the dataset**
   Get the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) from Kaggle and place `IMDB Dataset.csv` inside the `api` folder.

## Usage

**Run the accuracy evaluation script:**
```bash
python classify.py
```
Prints predictions vs. actual labels for a sample of reviews, plus overall accuracy.

**Run the API server:**
```bash
uvicorn main:app --reload
```
Then open `http://127.0.0.1:8000/docs` for the interactive Swagger UI, or send a request directly:

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely terrible, I wanted my money back."}'
```

**Response:**
```json
{
  "review": "This movie was absolutely terrible, I wanted my money back.",
  "sentiment": "negative"
}
```

## What This Demonstrates

- **LLM Integration** — practical API use and prompt engineering, not just a wrapper call.
- **Model Evaluation** — accuracy measured against ground truth, not assumed.
- **Backend Design** — clean REST API architecture with FastAPI.
- **End-to-End Thinking** — data → model → measurable output → usable service.

## Use Cases

This pattern (unstructured text → LLM classification → structured output → API) is used in production for:
- Customer feedback analysis (auto-flagging negative reviews)
- Social media / brand sentiment monitoring
- Support ticket triage and routing
- Survey response analysis
- Market/news sentiment signals

## Author

Aastha Rathod
[GitHub](https://github.com/aasthaa2210) · [LinkedIn](https://linkedin.com/in/aastha-rathod21)