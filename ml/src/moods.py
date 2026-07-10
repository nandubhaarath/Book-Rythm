"""
Collapses GoEmotions' 28 fine-grained emotions into 6 broad moods
that the music engine can actually use. This mapping is the bridge
between what the model predicts and what the reader hears.
"""

# The 6 moods we're targeting. Order matters — each mood gets an
# index (0-5) that becomes the model's actual prediction target later.
MOODS = ["joyful", "tender", "tense", "sombre", "unease", "neutral"]

# Maps each of the 28 GoEmotions labels (by NAME) to one of our moods.
# Grouped by musical intent, not dictionary meaning.
EMOTION_TO_MOOD = {
    # Joyful — bright, energetic, positive
    "amusement": "joyful", "excitement": "joyful", "joy": "joyful",
    "optimism": "joyful", "pride": "joyful", "admiration": "joyful",
    "approval": "joyful", "relief": "joyful",
    # Tender — warm, gentle, affectionate
    "love": "tender", "caring": "tender", "gratitude": "tender",
    "desire": "tender",
    # Tense — fear, threat, hostility
    "fear": "tense", "nervousness": "tense", "anger": "tense",
    "disgust": "tense",
    # Sombre — sadness, loss, low energy
    "sadness": "sombre", "grief": "sombre", "disappointment": "sombre",
    "remorse": "sombre", "embarrassment": "sombre",
    # Unease — mild friction, doubt, restlessness
    "annoyance": "unease", "disapproval": "unease", "confusion": "unease",
    # Neutral — flat, ambiguous, or purely cognitive
    "neutral": "neutral", "realization": "neutral",
    "curiosity": "neutral", "surprise": "neutral",
}


def emotion_to_mood_index(emotion_name):
    # Takes an emotion name (e.g. "grief") and returns the INDEX of its
    # mood (e.g. 3 for "sombre"). Index, not name, because the model
    # works in numbers — it predicts "mood 3", and we translate back.
    mood_name = EMOTION_TO_MOOD[emotion_name]
    return MOODS.index(mood_name)


if __name__ == "__main__":
    # Sanity check: run through every emotion and confirm it maps.
    # This catches typos — if you misspelled an emotion, it'll be
    # missing from the dict and we'd want to know now, not later.
    print("Emotion -> Mood mapping:")
    for emotion, mood in EMOTION_TO_MOOD.items():
        index = emotion_to_mood_index(emotion)
        print(f"  {emotion:15s} -> {mood} (index {index})")

    print(f"\nTotal emotions mapped: {len(EMOTION_TO_MOOD)}")
    print(f"Should be 28. Match: {len(EMOTION_TO_MOOD) == 28}")