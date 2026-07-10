"""
Loads GOEmotions dataset and shows us what we are working with.
So we can understand the data we are using.
go emotions data set made from reddit user comments and then the emotions
associated with it 
"""

from datasets import load_dataset

def load_goemotions():
    # downloads the dataset then resuses the local copy every time after
    # using simplifies version of dataset to save space , and querying quicker , exlcudes usless data like user ID
    dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
    return dataset

if __name__ == "__main__":
    data = load_goemotions()    
    
    print("Splits available:", list(data.keys()))
    print("Training examples:", len(data["train"]))

    # Print the first example to see the SHAPE of the data.
    first = data["train"][0]
    print("\nFirst example:")
    print("  Text:", first["text"])
    print("  Labels (as numbers):", first["labels"])

    # The dataset knows its own label names. Let's see all 28 label names,
    # paired with the number that represents each one.
    # this way we can then simplify the dataset into a few emotions 
    
    emotion_names = data["train"].features["labels"].feature.names
    print("\nAll emotion categories:")
    for number, name in enumerate(emotion_names):
        print(f"  {number}: {name}")
