# Fake Review Detection Application

A full-stack fake review detection project with:

- Frontend: Next.js 13 app in `client/`
- Backend: FastAPI server in `server/`
- ML models:
  - BERT-based text classifier (`server/controllers/bert_model.py`)
  - Classical ML + graph-based ensemble (`server/controllers/classical_ml.py`, `server/utils/graph.py`)

## Features

- Analyze a single review or a review set from Amazon ASIN
- Combined prediction from BERT and heuristic graph model
- `ml_score` endpoint with case-based `Fake | Valid` classification
- `bert_score` endpoint using text embedding prediction
- Local development mode with hot reload (Uvicorn, Next.js)

## Repository Structure

- `client/` : Next.js frontend, UI components, forms, results
- `server/` : FastAPI backend, controllers, routes, ML pipelines
- `server/models/` : saved model artifacts (`bert.pt`, `svm_pipeline.pkl`, `graph_model.pkl`)
- `server/data/reviews.csv` : review dataset for graph model training
- `server/utils/algorithms/` : model training utilities

## Prerequisites

- Python 3.11+ (or 3.10/3.12 compatible)
- Node.js 18+
- Git

## Getting Started

Follow these steps to set up and run the Fake Review Detection Application locally.

### 1. Clone the Repository

```bash
git clone https://github.com/sreyas-b-anand/s6-mini-project.git
cd s6-mini-project
```

### 2. Install Backend Libraries

Navigate to the server directory and set up the Python environment:

```powershell
cd server
python -m venv venv
venv\Scripts\activate 
pip install -r requirements.txt
```

This installs all required Python packages including FastAPI, scikit-learn, torch, and NLTK data.

### 3. Run the Backend

From the project root directory:

```powershell
cd c:\minipro\s6-mini-project
venv\Scripts\activate
uvicorn server.main:app --reload --host 127.0.0.1 --port 8000
```

The backend will start on http://127.0.0.1:8000. You should see logs indicating the models are loading.

### 4. Install Frontend Libraries

In a new terminal, navigate to the client directory:

```bash
cd client
npm i
```

This installs all Node.js dependencies including Next.js, React, and Tailwind CSS.

### 5. Run the Frontend

From the client directory:

```bash
npm run dev
```

Open http://localhost:3000 in your browser to access the application.

## Backend Setup

```powershell
cd server
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m nltk.downloader stopwords punkt
```

### Start backend

```powershell
cd c:\minipro\s6-mini-project
venv\Scripts\activate
uvicorn server.main:app --reload --host 127.0.0.1 --port 8000
```

**Note:** `server/main.py` now includes a sys.path adjustment so module imports (`server.*`) and unpickling from `joblib` are robust in Uvicorn mode.

## Frontend Setup

```bash
cd client
npm install
npm run dev
```

Open `http://localhost:3000`.

## API Endpoints

- `POST /model/bert_score` -> body: `type: "single"` or `"link"`, `review`, `rating`, `url` as needed
- `POST /model/ml_score` -> body: `{ text, rating, category }`
- `GET /model/train_graph` -> rebuild graph model from `server/data/reviews.csv`

## Common Troubleshooting

- If you see `ModuleNotFoundError: No module named 'server'`, ensure server root is in PYTHONPATH and run from repo root:
  - `uvicorn server.main:app --reload`
- If weights load warnings appear for BERT (`UNEXPECTED` keys), they are expected when using a model checkpoint for a different target head.
- If `graph_model.pkl` is missing, run `/model/train_graph` once or use the training script in `server/utils/graph.py`.

## Development Notes

- All graph building and prediction logic in `server/utils/graph.py`.
- Classical ML pipeline in `server/controllers/classical_ml.py` loads `svm_pipeline.pkl` + `graph_model.pkl` on startup for fast requests.
- FastAPI router logic in `server/routes/model_routes.py`.

## License

MIT
