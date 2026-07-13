"""
Turns prepared texts and moods into padded, batched tensors
the model can train on. This is the pipeline that feeds the network.
"""

import torch
from torch.utils.data import Dataset, DataLoader

MAX_LENGTH = 32   # chosen from the length distribution: 95th pct is 24
PAD_ID = 0        # matches the tokeniser's <pad> token


class EmotionDataset(Dataset):
    """
    Wraps our texts and moods in PyTorch's Dataset interface.
    A Dataset must answer two questions: how many items are there,
    and give me item number i. That's all it does.
    """

    def __init__(self, texts, moods, tokeniser, max_length=MAX_LENGTH):
        self.texts = texts
        self.moods = moods
        self.tokeniser = tokeniser
        self.max_length = max_length

    def __len__(self):
        # How many examples in total. PyTorch calls this.
        return len(self.texts)

    def __getitem__(self, index):
        # Return ONE example, ready for the model.
        token_ids = self.tokeniser.encode(self.texts[index])

        # Truncate if too long.
        token_ids = token_ids[:self.max_length]

        # Pad with 0s if too short, until it hits max_length.
        padding_needed = self.max_length - len(token_ids)
        token_ids = token_ids + [PAD_ID] * padding_needed

        # Convert to tensors — PyTorch's native data type.
        # long = 64-bit integer, which is what embedding layers expect.
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(self.moods[index], dtype=torch.long),
        )


def make_loader(texts, moods, tokeniser, batch_size=64, shuffle=True):
    # DataLoader takes a Dataset and handles batching for us:
    # it groups examples into batches and stacks them into tensors.
    # shuffle=True reorders the data each epoch — important, so the
    # model doesn't learn the ORDER of the data instead of the content.
    dataset = EmotionDataset(texts, moods, tokeniser)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    from dataset import load_goemotions
    from prepare_data import prepare_split
    from tokeniser import Tokeniser
    from moods import MOODS

    data = load_goemotions()
    train_texts, train_moods = prepare_split(data["train"])

    tok = Tokeniser(min_frequency=2)
    tok.build_vocab(train_texts)

    loader = make_loader(train_texts, train_moods, tok)

    # Grab ONE batch and inspect it — the shapes tell us everything.
    token_batch, mood_batch = next(iter(loader))

    print(f"Token batch shape: {token_batch.shape}")
    print(f"Expected:          torch.Size([64, 32])  (64 sentences, 32 tokens)")
    print(f"Mood batch shape:  {mood_batch.shape}")
    print(f"Expected:          torch.Size([64])      (64 answers)")

    # Look at one real example, decoded back to words.
    print(f"\nFirst example in batch:")
    print(f"  IDs:  {token_batch[0].tolist()}")
    print(f"  Text: {tok.decode(token_batch[0].tolist())}")
    print(f"  Mood: {MOODS[mood_batch[0].item()]}")