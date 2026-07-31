import sys
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make the ML side importable from here.
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(repo_root / "ml" / "src"))

from predict import MoodPredictor
from app.pdf_parser import extract_paragraphs

app = FastAPI(title="Book-Rythm API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load once at startup, not per-request.
predictor = MoodPredictor(
    repo_root / "ml" / "models" / "emotion_model_weighted.pt",
    repo_root / "ml" / "models" / "vocab.json",
)


class PredictRequest(BaseModel):
    text: str


class BatchRequest(BaseModel):
    paragraphs: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    return predictor.predict(request.text)


@app.post("/predict-batch")
def predict_batch(request: BatchRequest):
    return {"results": [predictor.predict(p) for p in request.paragraphs]}


@app.post("/upload")
async def upload(file: UploadFile):
    pdf_bytes = await file.read()
    paragraphs = extract_paragraphs(pdf_bytes)

    results = []
    for paragraph in paragraphs:
        prediction = predictor.predict(paragraph)
        results.append({
            "text": paragraph,
            "mood": prediction["mood"],
            "confidence": prediction["confidence"],
        })

    return {"paragraphs": results}
