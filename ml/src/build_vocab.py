"""
Builds the real vocabulary from the full training set and reports
its size — the number the model's first layer depends on.
"""

from dataset import load_goemotions
from prepare_data import prepare_split
from tokeniser import Tokeniser


if __name__ == "__main__":
    data = load_goemotions()

    # Get the training texts (we ignore the moods here — we only
    # need the text to build a vocabulary).
    train_texts, _ = prepare_split(data["train"])
    print(f"Building vocab from {len(train_texts)} comments...")

    # Build the vocabulary. min_frequency=2 means a word must appear
    # at least twice to earn a slot — rarer words become <unk>.
    tok = Tokeniser(min_frequency=2)
    tok.build_vocab(train_texts)

    print(f"Vocabulary size: {tok.vocab_size()}")

    # Try encoding a real sentence to see it in action.
    sample = "I am so happy this made my whole day"
    print(f"\nSample: {sample}")
    print(f"Encoded: {tok.encode(sample)}")
    print(f"Decoded: {tok.decode(tok.encode(sample))}")