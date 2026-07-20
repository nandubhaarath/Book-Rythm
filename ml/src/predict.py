"""
Loads the trained model and exposes a single function: text -> mood.
This is the boundary between the ML side and the app side.
"""

import torch

from tokeniser import Tokeniser
from model import EmotionClassifier
from moods import MOODS

MAX_LENGTH = 32
PAD_ID = 0


class MoodPredictor:
    def __init__(self, model_path, vocab_path):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Load the exact vocabulary the model was trained on.
        self.tokeniser = Tokeniser()
        self.tokeniser.load(vocab_path)

        # Rebuild the architecture, then pour the trained weights in.
        self.model = EmotionClassifier(
            vocab_size=self.tokeniser.vocab_size()).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device))
        self.model.eval()   # inference mode, not training

    def predict(self, text):
        # Tokenise and pad exactly as we did in training.
        # Any mismatch here silently corrupts predictions.
        ids = self.tokeniser.encode(text)[:MAX_LENGTH]
        ids = ids + [PAD_ID] * (MAX_LENGTH - len(ids))

        # Add a batch dimension: the model expects (batch, seq_len),
        # so a single sentence becomes a batch of 1.
        tensor = torch.tensor([ids], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            # Softmax turns raw scores into probabilities that sum to 1.
            probabilities = torch.softmax(logits, dim=1)[0]

        mood_index = probabilities.argmax().item()

        return {
            "mood": MOODS[mood_index],
            "confidence": probabilities[mood_index].item(),
            "all_scores": {
                mood: probabilities[i].item()
                for i, mood in enumerate(MOODS)
            },
        }


if __name__ == "__main__":
    predictor = MoodPredictor(
        "models/emotion_model_weighted.pt",
        "models/vocab.json",
    )

    # Test on sentences that SHOULD be obvious. If it fails these,
    # something is wrong with the loading.
    tests = [
        "I am so happy, this is the best day of my life",
        "He crept through the darkness, heart pounding with terror",
        "She wept quietly, mourning everything she had lost",
        "I love you more than words can say",
        "The door was made of oak and stood six feet tall",
    ]

    for text in tests:
        result = predictor.predict(text)
        print(f"\n{text}")
        print(f"  -> {result['mood']} ({result['confidence']:.1%})")