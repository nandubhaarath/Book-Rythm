"""
Honest evaluation: per-mood performance, not just overall accuracy.
Reveals whether the model actually learned all 6 moods, or just
the common ones.
"""

import torch

from dataset import load_goemotions
from prepare_data import prepare_split
from tokeniser import Tokeniser
from data_loader import make_loader
from model import EmotionClassifier
from moods import MOODS


def collect_predictions(model, loader, device):
    # Run the whole validation set through and record every
    # prediction alongside its true answer.
    model.eval()
    all_predictions = []
    all_truths = []

    with torch.no_grad():
        for token_ids, true_moods in loader:
            token_ids = token_ids.to(device)
            logits = model(token_ids)
            predictions = logits.argmax(dim=1).cpu()

            all_predictions.extend(predictions.tolist())
            all_truths.extend(true_moods.tolist())

    return all_predictions, all_truths


def confusion_matrix(predictions, truths, num_classes=6):
    # matrix[true][predicted] = how many times that mistake happened.
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for true, pred in zip(truths, predictions):
        matrix[true][pred] += 1
    return matrix


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_goemotions()
    train_texts, train_moods = prepare_split(data["train"])
    val_texts, val_moods = prepare_split(data["validation"])

    # Rebuild the SAME vocabulary the model was trained with.
    tok = Tokeniser(min_frequency=2)
    tok.build_vocab(train_texts)

    val_loader = make_loader(val_texts, val_moods, tok, shuffle=False)

    # Load the trained weights back into a fresh model.
    model = EmotionClassifier(vocab_size=tok.vocab_size()).to(device)
    model.load_state_dict(torch.load("models/emotion_model_weighted.pt"))

    predictions, truths = collect_predictions(model, val_loader, device)

    # --- The confusion matrix ---
    matrix = confusion_matrix(predictions, truths)

    print("CONFUSION MATRIX")
    print("Rows = true mood, Columns = what the model predicted\n")
    print(f"{'':10s}", end="")
    for mood in MOODS:
        print(f"{mood:>9s}", end="")
    print()

    for i, mood in enumerate(MOODS):
        print(f"{mood:10s}", end="")
        for j in range(len(MOODS)):
            print(f"{matrix[i][j]:>9d}", end="")
        print()

    # --- Per-mood precision and recall ---
    print("\n\nPER-MOOD PERFORMANCE\n")
    print(f"{'mood':10s}{'precision':>12s}{'recall':>10s}{'support':>10s}")

    for i, mood in enumerate(MOODS):
        # True positives: correctly predicted as this mood.
        tp = matrix[i][i]
        # All examples the model PREDICTED as this mood (column sum).
        predicted_as_this = sum(matrix[row][i] for row in range(len(MOODS)))
        # All examples that TRULY are this mood (row sum).
        truly_this = sum(matrix[i])

        precision = tp / predicted_as_this if predicted_as_this > 0 else 0.0
        recall = tp / truly_this if truly_this > 0 else 0.0

        print(f"{mood:10s}{precision:>11.1%}{recall:>10.1%}{truly_this:>10d}")