"""
A word-level tokeniser built from scratch.
Turns text into lists of integer IDs the model can process.
Two jobs: build a vocabulary from the training data, then use it
to encode any sentence into numbers (and decode back for debugging).
"""

from collections import Counter

# Special tokens get reserved slots at the very start of the vocab.
PAD_TOKEN = "<pad>"   # fills short sentences to a fixed length
UNK_TOKEN = "<unk>"   # stands in for any word not in the vocab


class Tokeniser:
    def __init__(self, min_frequency=2):
        # A word must appear at least this many times to earn a slot.
        self.min_frequency = min_frequency
        # These two dicts are the heart of the tokeniser — the lookup
        # tables in both directions.
        self.word_to_id = {}   # "cat" -> 891
        self.id_to_word = {}   # 891 -> "cat"

    def _tokenise(self, text):
        # Our splitting rule: lowercase everything, then split on spaces.
        # Lowercasing means "The" and "the" are treated as one word,
        # which keeps the vocabulary smaller. Crude but effective for v1.
        return text.lower().split()

    def build_vocab(self, texts):
        # Step 1: count how often every word appears across all texts.
        counter = Counter()
        for text in texts:
            counter.update(self._tokenise(text))

        # Step 2: reserve slots 0 and 1 for the special tokens.
        self.word_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}

        # Step 3: add every word that clears the frequency threshold.
        for word, count in counter.items():
            if count >= self.min_frequency:
                self.word_to_id[word] = len(self.word_to_id)

        # Step 4: build the reverse lookup for decoding.
        self.id_to_word = {i: w for w, i in self.word_to_id.items()}

    def encode(self, text):
        # Turn a sentence into a list of IDs. Unknown words -> <unk>'s ID.
        unk_id = self.word_to_id[UNK_TOKEN]
        return [self.word_to_id.get(word, unk_id)
                for word in self._tokenise(text)]

    def decode(self, ids):
        # Turn a list of IDs back into words. Handy for sanity-checking.
        return " ".join(self.id_to_word.get(i, UNK_TOKEN) for i in ids)

    def vocab_size(self):
        return len(self.word_to_id)


if __name__ == "__main__":
    # Test it on a tiny hand-made dataset first, before the real thing.
    sample_texts = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat and the dog",
    ]

    tok = Tokeniser(min_frequency=1)  # freq 1 so nothing is dropped here
    tok.build_vocab(sample_texts)

    print("Vocabulary size:", tok.vocab_size())
    print("Word-to-ID map:", tok.word_to_id)

    # Encode a sentence, then decode it back — should round-trip.
    encoded = tok.encode("the cat sat")
    print("\n'the cat sat' encoded:", encoded)
    print("Decoded back:", tok.decode(encoded))

    # Test the <unk> fallback with a word not in the vocab.
    print("\n'the zebra sat' encoded:", tok.encode("the zebra sat"))
    print("Decoded back:", tok.decode(tok.encode("the zebra sat")))