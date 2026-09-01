#Confirming the GPU is actually attached.

import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

#Installing what this episode needs.

#!pip install -q datasets tiktoken matplotlib

#Loading the data, three ways
#1 Raw Python — download and inspect the file directly.

import urllib.request

url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
urllib.request.urlretrieve(url, "tinystories_sample.txt")

with open("tinystories_sample.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

stories = raw_text.split("<|endoftext|>")
stories = [s.strip() for s in stories if s.strip()]
print("Number of stories:", len(stories))
print("First story:\n", stories[0])

#2 PyTorch — wrapping it as a Dataset, ready for training later.

from torch.utils.data import Dataset, DataLoader

class TinyStoriesRaw(Dataset):
    def __init__(self, stories):
        self.stories = stories

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        return self.stories[idx]

ds = TinyStoriesRaw(stories)
loader = DataLoader(ds, batch_size=4, shuffle=True)
batch = next(iter(loader))
print("Batch size:", len(batch))
print("Sample from batch:\n", batch[0][:150])

#3 Production library — the same dataset via Hugging Face datasets.

from datasets import load_dataset

hf_ds = load_dataset("roneneldan/TinyStories", split="validation")
print(hf_ds)
print("\nFirst story:\n", hf_ds[0]["text"][:150])

# Seeing what's actually in the data

# 1 Story length distribution.

import matplotlib.pyplot as plt

lengths = [len(s.split()) for s in stories]

plt.figure(figsize=(8, 4))
plt.hist(lengths, bins=40, color="#2C6E7F", edgecolor="white")
plt.xlabel("Story length (words)")
plt.ylabel("Number of stories")
plt.title("TinyStories — story length distribution")
plt.tight_layout()
plt.savefig("story_lengths.png", dpi=150)
plt.show()

#2 Vocabulary size — the number that explains why this works.

import re
from collections import Counter

words = re.findall(r"\b\w+\b", raw_text.lower())
vocab = Counter(words)
print("Total words:", len(words))
print("Unique words:", len(vocab))
print("Top 10 most common:", vocab.most_common(10))