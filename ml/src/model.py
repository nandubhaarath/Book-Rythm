"""
The emotion classifier: embeddings -> LSTM -> linear -> 6 mood scores.
This is the model architecture. It doesn't learn anything yet —
it just defines the SHAPE of the network and how data flows through it.
"""

import torch
import torch.nn as nn


class EmotionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100,
                 hidden_dim=128, num_moods=6, pad_id=0):
        # Every PyTorch model must call this first — it sets up the
        # machinery that tracks all our learnable weights.
        super().__init__()

        # Layer 1: the embedding table.
        # vocab_size rows (one per word), embedding_dim numbers each.
        # padding_idx tells it: ID 0 is <pad>, so keep its vector at
        # zero and never learn from it — padding carries no meaning.
        self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                      padding_idx=pad_id)

        # Layer 2: the LSTM. Reads the sequence of word vectors in
        # order and outputs a running summary at each step.
        # batch_first=True means our data is shaped (batch, words, features)
        # rather than PyTorch's odd default. Saves endless confusion.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Layer 3: the classifier head. Takes the LSTM's final summary
        # (hidden_dim numbers) and squeezes it down to 6 mood scores.
        self.fc = nn.Linear(hidden_dim, num_moods)

    def forward(self, token_ids):
        # 'forward' defines what happens when data passes through.
        # You never call this directly — PyTorch calls it for you.

        # token_ids shape: (batch_size, sequence_length)
        # e.g. 32 sentences, each 50 words long -> (32, 50)

        # Each ID becomes a vector -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(token_ids)

        # The LSTM returns two things:
        #   outputs = its summary at EVERY word (we don't need these)
        #   (hidden, cell) = its FINAL memory after the last word
        # We want the final one — it summarises the whole sentence.
        outputs, (hidden, cell) = self.lstm(embedded)

        # hidden has an extra leading dimension (for stacked LSTMs).
        # We only have one layer, so squeeze it out to get
        # (batch_size, hidden_dim).
        final_hidden = hidden.squeeze(0)

        # Map the summary to 6 raw mood scores (logits).
        logits = self.fc(final_hidden)

        return logits


if __name__ == "__main__":
    # Build the model with our real vocabulary size.
    VOCAB_SIZE = 18607
    model = EmotionClassifier(vocab_size=VOCAB_SIZE)

    print(model)

    # Count the learnable parameters — every number the model can adjust.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal learnable parameters: {total_params:,}")

    # Feed it FAKE data to check the shapes line up.
    # 4 sentences, each 10 words long, random word IDs.
    fake_batch = torch.randint(0, VOCAB_SIZE, (4, 10))
    print(f"\nInput shape:  {fake_batch.shape}")

    output = model(fake_batch)
    print(f"Output shape: {output.shape}")
    print("Expected:     torch.Size([4, 6])  (4 sentences, 6 mood scores)")

    print(f"\nRaw scores for first sentence:\n{output[0]}")