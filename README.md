# Twitter Sentiment Analysis Web App

A full-stack machine learning web application that analyzes the sentiment of tweets and text in real time. Built with a Linear SVM model trained on 1.6 million tweets, served via a Flask REST API, and deployed with a responsive HTML/CSS/JS frontend.

🔗 **Live Demo:** [sentiment-app-gules.vercel.app](https://sentiment-app-gules.vercel.app)  
⚙️ **API:** [sentiment-api-9dsj.onrender.com](https://sentiment-api-9dsj.onrender.com)

---

## Features

- Real-time sentiment prediction (Positive / Negative)
- Confidence score displayed with animated progress bar
- Example tweets to try instantly
- Responsive UI — works on mobile and desktop
- REST API with `/predict` and `/health` endpoints

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Linear SVM (scikit-learn) |
| Text Processing | TF-IDF Vectorizer, NLTK, Porter Stemmer |
| Backend | Python, Flask, Flask-CORS |
| Frontend | HTML, CSS, JavaScript (Fetch API) |
| Backend Deploy | Render |
| Frontend Deploy | Vercel |
| Dataset | Sentiment140 (1.6M tweets) |

---

## Project Structure

```
sentiment_app/
├── app.py                  # Flask REST API
├── index.html              # Frontend UI
├── requirements.txt        # Python dependencies
├── trained_model.sav       # Trained LinearSVC model
├── vectorizer.sav          # Fitted TF-IDF vectorizer
└── twitter_sentiment_clean.ipynb  # Training notebook
```

---

## How It Works

1. **Data** — Sentiment140 dataset with 1.6M labelled tweets (0 = negative, 4 = positive)
2. **Preprocessing** — Remove special characters, lowercase, remove stopwords, apply Porter Stemming
3. **Vectorization** — TF-IDF converts text to numerical feature vectors
4. **Model** — LinearSVC wrapped in CalibratedClassifierCV for probability scores
5. **API** — Flask exposes a `/predict` POST endpoint that accepts text and returns sentiment + confidence
6. **Frontend** — Fetch API sends user input to Flask, displays result dynamically

---

## API Usage

**Endpoint:** `POST /predict`

**Request:**
```json
{
  "text": "I love this amazing day!"
}
```

**Response:**
```json
{
  "sentiment": "Positive",
  "confidence": 98.2,
  "label": 1
}
```

---

## Local Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/sentiment-app.git
cd sentiment-app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the API
python app.py

# Open index.html with Live Server in VS Code
```

---

## Model Performance

| Metric | Score |
|---|---|
| Training Accuracy | ~85% |
| Test Accuracy | ~79% |
| Dataset Size | 1.6M tweets |
| Algorithm | LinearSVC + Calibration |

---

## Dataset

[Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) — 1.6 million tweets automatically labelled using emoticons as proxies for sentiment.
