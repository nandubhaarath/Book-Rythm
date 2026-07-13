"""
The training loop — where the model actually learns.
Runs batches through the model, measures error, and nudges weights.
"""

import torch
import torch.nn as nn

from dataset import load_goemotions
from prepare_data import prepare_split
from tokeniser import Tokeniser
from data_loader import make_loader
from model import EmotionClassifier
from moods import MOODS

# Settings — the knobs you'll tune later.
EPOCHS = 5              # how many full passes over the training data
LEARNING_RATE = 0.001   # how big a step the optimiser takes
BATCH_SIZE = 64


def train_one_epoch(model, loader, loss_fn, optimiser, device):
    model.train()   # tells the model it's in training mode
    total_loss = 0
    correct = 0
    total = 0

    for token_ids, true_moods in loader:
        # Move this batch onto the GPU.
        token_ids = token_ids.to(device)
        true_moods = true_moods.to(device)

        # 1. FORWARD — predict.
        logits = model(token_ids)

        # 2. LOSS — how wrong were we?
        loss = loss_fn(logits, true_moods)

        # 3. BACKWARD — work out which way each weight should move.
        optimiser.zero_grad()   # clear last batch's gradients first!
        loss.backward()         # autograd computes all the gradients

        # 4. STEP — actually nudge the weights.
        optimiser.step()

        # Track stats so we can watch progress.
        total_loss += loss.item()
        predictions = logits.argmax(dim=1)   # highest score wins
        correct += (predictions == true_moods).sum().item()
        total += true_moods.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, loss_fn, device):
    model.eval()    # tells the model it's NOT training
    total_loss = 0
    correct = 0
    total = 0

    # no_grad: we're not learning here, so don't track gradients.
    # Saves memory and time.
    with torch.no_grad():
        for token_ids, true_moods in loader:
            token_ids = token_ids.to(device)
            true_moods = true_moods.to(device)

            logits = model(token_ids)
            loss = loss_fn(logits, true_moods)

            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == true_moods).sum().item()
            total += true_moods.size(0)

    return total_loss / len(loader), correct / total


if __name__ == "__main__":
    # Use the GPU if it's there.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Load and prepare the data.
    data = load_goemotions()
    train_texts, train_moods = prepare_split(data["train"])
    val_texts, val_moods = prepare_split(data["validation"])

    # Build the vocabulary from TRAINING data only.
    # Never let the model peek at validation data when building the
    # vocab — that's data leakage, and it inflates your scores.
    tok = Tokeniser(min_frequency=2)
    tok.build_vocab(train_texts)
    print(f"Vocab size: {tok.vocab_size()}")

    train_loader = make_loader(train_texts, train_moods, tok,
                               batch_size=BATCH_SIZE, shuffle=True)
    val_loader = make_loader(val_texts, val_moods, tok,
                             batch_size=BATCH_SIZE, shuffle=False)

    # Build the model and move it to the GPU.
    model = EmotionClassifier(vocab_size=tok.vocab_size()).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nStarting training for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimiser, device)
        val_loss, val_acc = evaluate(
            model, val_loader, loss_fn, device)

        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"  Train — loss: {train_loss:.4f}  acc: {train_acc:.2%}")
        print(f"  Val   — loss: {val_loss:.4f}  acc: {val_acc:.2%}")

    # Save the trained weights so we don't have to retrain every time.
    torch.save(model.state_dict(), "models/emotion_model.pt")
    print("\nModel saved to models/emotion_model.pt")