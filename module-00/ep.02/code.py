# Episode 00.02: Building a Tokenizer From Scratch
# Module 00 — Setup & Real Data

import re
import torch
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 3.1 Raw Python — verify the mechanism on a hand-checkable example
# ---------------------------------------------------------------
toy_corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}

def get_vocab(word_freq):
    return {tuple(list(w) + ["</w>"]): f for w, f in word_freq.items()}

def get_pair_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    new_vocab = {}
    for word, freq in vocab.items():
        new_word, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(word[i] + word[i + 1]); i += 2
            else:
                new_word.append(word[i]); i += 1
        new_vocab[tuple(new_word)] = freq
    return new_vocab

vocab = get_vocab(toy_corpus)
toy_merges = []
for step in range(6):
    pairs = get_pair_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    toy_merges.append(best)
    print(f"Merge {step+1}: {best}")
print(toy_merges)

# ---------------------------------------------------------------
# 3.2 Raw Python — train on the real TinyStories corpus
# ---------------------------------------------------------------
with open("tinystories_sample.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

stories = [s.strip() for s in raw_text.split("<|endoftext|>") if s.strip()]
train_text = " ".join(stories[:2000])

words = re.findall(r"\b\w+\b", train_text.lower())
word_freq = Counter(words)
print("Unique words in training slice:", len(word_freq))

vocab = get_vocab(word_freq)
num_merges = 300
merges = []
for step in range(num_merges):
    pairs = get_pair_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    merges.append(best)

print(f"Learned {len(merges)} merges")
print("First 10:", merges[:10])
print("Last 10:", merges[-10:])

# ---------------------------------------------------------------
# 3.3 Raw Python — encode / decode
# ---------------------------------------------------------------
merge_ranks = {pair: i for i, pair in enumerate(merges)}

def bpe_tokenize_word(word, merge_ranks):
    symbols = list(word) + ["</w>"]
    while True:
        pairs = [(symbols[i], symbols[i+1]) for i in range(len(symbols) - 1)]
        candidates = [(merge_ranks[p], p) for p in pairs if p in merge_ranks]
        if not candidates:
            break
        _, best_pair = min(candidates)
        new_symbols, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == best_pair:
                new_symbols.append(symbols[i] + symbols[i+1]); i += 2
            else:
                new_symbols.append(symbols[i]); i += 1
        symbols = new_symbols
    return symbols

def bpe_encode(text, merge_ranks):
    words = re.findall(r"\b\w+\b|[^\w\s]", text.lower())
    tokens = []
    for w in words:
        tokens.extend(bpe_tokenize_word(w, merge_ranks) if re.match(r"\w+", w) else [w])
    return tokens

def bpe_decode(tokens):
    return "".join(tokens).replace("</w>", " ").strip()

sentence = "The little dragon wanted to befriend the grumpy wizard."
encoded = bpe_encode(sentence, merge_ranks)
print("Encoded:", encoded)
print("Token count:", len(encoded))
print("Decoded: ", bpe_decode(encoded))

# ---------------------------------------------------------------
# 3.4 PyTorch — token IDs as tensors
# ---------------------------------------------------------------
all_tokens = sorted(set(t for w in vocab for t in w))
itos = ["<pad>", "<unk>"] + all_tokens
stoi = {tok: i for i, tok in enumerate(itos)}
print("Token-id vocab size:", len(itos))

def encode_to_ids(text, merge_ranks, stoi):
    return [stoi.get(t, stoi["<unk>"]) for t in bpe_encode(text, merge_ranks)]

ids = encode_to_ids(sentence, merge_ranks, stoi)
tensor_ids = torch.tensor(ids, dtype=torch.long)
print("Tensor:", tensor_ids)
print("Shape:", tensor_ids.shape, "dtype:", tensor_ids.dtype)

batch = [
    "The little dragon wanted to befriend the grumpy wizard.",
    "A cat sat on the mat.",
    "Once upon a time there was a very tall castle in the clouds.",
]
batch_ids = [encode_to_ids(s, merge_ranks, stoi) for s in batch]
max_len = max(len(x) for x in batch_ids)
padded = [x + [stoi["<pad>"]] * (max_len - len(x)) for x in batch_ids]
batch_tensor = torch.tensor(padded, dtype=torch.long)
print("Batch tensor shape:", batch_tensor.shape)

# ---------------------------------------------------------------
# 3.5 Production library — tiktoken (GPT-2 BPE) comparison
# ---------------------------------------------------------------
# NOTE: requires network access to openaipublic.blob.core.windows.net.
# Run this section on Colab; it is blocked in the authoring sandbox.
#
# import tiktoken
# enc = tiktoken.get_encoding("gpt2")
# gpt2_tokens = enc.encode(sentence)
# print("GPT-2 token count:", len(gpt2_tokens))
# print("GPT-2 pieces:", [enc.decode([t]) for t in gpt2_tokens])
# print("GPT-2 vocab size:", enc.n_vocab)

# ---------------------------------------------------------------
# 4.1 Vocabulary growth plot
# ---------------------------------------------------------------
steps, vocab_sizes = [], []
vocab2 = get_vocab(word_freq)
for step in range(300):
    pairs = get_pair_stats(vocab2)
    best = max(pairs, key=pairs.get)
    vocab2 = merge_vocab(best, vocab2)
    if (step + 1) % 5 == 0:
        steps.append(step + 1)
        vocab_sizes.append(len(set(t for w in vocab2 for t in w)))

plt.figure(figsize=(8, 4))
plt.plot(steps, vocab_sizes, color="#2C6E7F", linewidth=2)
plt.xlabel("Number of merges")
plt.ylabel("Subword vocabulary size")
plt.title("BPE vocabulary growth (300 merges, 2000-story training slice)")
plt.tight_layout()
plt.savefig("bpe_vocab_growth.png", dpi=150)

# ---------------------------------------------------------------
# 4.2 Whole-word vs BPE sequence length comparison
# ---------------------------------------------------------------
test_stories = stories[2000:2200]
whole_word_lens = [len(re.findall(r"\b\w+\b", s)) for s in test_stories]
bpe_lens = [len(bpe_encode(s, merge_ranks)) for s in test_stories]

print(f"Avg whole-word tokens/story: {sum(whole_word_lens)/len(whole_word_lens):.1f}")
print(f"Avg BPE tokens/story:        {sum(bpe_lens)/len(bpe_lens):.1f}")
print(f"Ratio: {sum(bpe_lens)/sum(whole_word_lens):.2f}x")

plt.figure(figsize=(8, 4))
plt.hist(whole_word_lens, bins=25, alpha=0.6, label="Whole-word split", color="#2C6E7F")
plt.hist(bpe_lens, bins=25, alpha=0.6, label="BPE (300 merges)", color="#D9822B")
plt.xlabel("Tokens per story")
plt.ylabel("Number of stories")
plt.title("Sequence length: whole-word split vs. from-scratch BPE")
plt.legend()
plt.tight_layout()
plt.savefig("bpe_vs_wordsplit_lengths.png", dpi=150)

# ---------------------------------------------------------------
# 4.3 OOV handling demo
# ---------------------------------------------------------------
unseen_word = "grumplewick"
print("Whole-word vocab lookup: NOT FOUND — would need <unk>")
print("BPE decomposition:", bpe_tokenize_word(unseen_word, merge_ranks))