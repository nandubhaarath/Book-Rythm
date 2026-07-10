"""
Turns raw GoEmotions comments into clean training examples:
each comment paired with a single mood index (0-5).
This is the data our model will actually learn from.
"""

from dataset import load_goemotions
from moods import emotion_to_mood_index, MOODS


def prepare_split(split):
    # 'split' is one portion of the data (train, validation, or test).
    # We build two parallel lists: the texts, and their mood indices.
    texts = []
    mood_indices = []

    # The dataset stores emotion NAMES in this list, indexed by number.
    emotion_names = split.features["labels"].feature.names

    for example in split:
        labels = example["labels"]  # e.g. [27] or [0, 4] or []

        # Skip comments with no emotion label — nothing to learn from.
        if len(labels) == 0:
            continue

        # Take the FIRST labelled emotion as the dominant one.
        first_emotion_number = labels[0]
        emotion_name = emotion_names[first_emotion_number]

        # Convert that emotion into one of our 6 mood indices.
        mood_index = emotion_to_mood_index(emotion_name)

        texts.append(example["text"])
        mood_indices.append(mood_index)

    return texts, mood_indices


if __name__ == "__main__":
    data = load_goemotions()

    train_texts, train_moods = prepare_split(data["train"])
    print(f"Prepared {len(train_texts)} training examples")

    # Show a few real examples so we can eyeball whether it's sensible.
    print("\nFirst 5 examples:")
    for i in range(5):
        mood_name = MOODS[train_moods[i]]
        print(f"  [{mood_name}] {train_texts[i]}")

    # Count how many examples fall into each mood — this reveals
    # whether our data is BALANCED or lopsided. Crucial to check.
    print("\nMood distribution:")
    for index, mood_name in enumerate(MOODS):
        count = train_moods.count(index)
        print(f"  {mood_name:8s}: {count}")