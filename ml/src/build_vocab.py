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

    # How long are comments actually? This decides our padding length.
    lengths = [len(tok.encode(text)) for text in train_texts]
    lengths.sort()

    print(f"\nComment lengths (in tokens):")
    print(f"  Shortest: {lengths[0]}")
    print(f"  Longest:  {lengths[-1]}")
    print(f"  Median:   {lengths[len(lengths) // 2]}")
    # The 95th percentile: 95% of comments are AT MOST this long.
    print(f"  95th pct: {lengths[int(len(lengths) * 0.95)]}")

    tok.save("models/vocab.json")