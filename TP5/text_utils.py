
import struct

import numpy as np
import torch
import torch.nn as nn


# False : utiliser les embeddings Word2Vec pré-entraînés.
# True : créer des embeddings aléatoires si Word2Vec est inaccessible.
RANDOMIZED = False

ANNOTATIONS_PATH = "/mnt/ssd/flickr30k/results_20130124.token"
WORD2VEC_PATH = "/mnt/ssd/word2vec/GoogleNews-vectors-negative300.bin"


print("Computing vocabulary...")

vocabulary = set()

with open(ANNOTATIONS_PATH, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t", 1)

        if len(parts) != 2:
            continue

        _, words = parts

        for word in words.split(" "):
            if word:
                vocabulary.add(word.lower())

print("Number of words:", len(vocabulary))


def read_until(open_file):
    while True:
        char = open_file.read(1)

        if char == b" ":
            break

        if not char:
            return

        yield char




def read_word2vec():
    with open(WORD2VEC_PATH, "rb") as f:
        first_line = f.readline()
        n_words, n_dim = map(int, first_line.split())

        for _ in range(n_words):
            word = b"".join(read_until(f)).decode(
                "utf-8", errors="ignore"
            )

            vector_data = f.read(n_dim * 4)

            if len(vector_data) != n_dim * 4:
                break

            vector = struct.unpack("f" * n_dim, vector_data)

            # Passer le séparateur entre deux entrées Word2Vec.
            f.read(1)

            yield word, vector



def create_embedding_layer(vocab, randomized=False):
    if randomized:
        idx_to_word = [""]
        word_to_idx = {"": 0}
        weights = [np.zeros((1, 300), dtype=np.float32)]

        for word in sorted(vocab):
            if word == "":
                continue

            word_to_idx[word] = len(idx_to_word)
            idx_to_word.append(word)
            weights.append(np.random.rand(1, 300).astype(np.float32))

    else:
        weights = []
        idx_to_word = []
        word_to_idx = {}

        idx = 0

        for word, vector in read_word2vec():
            word = word.lower()

            if word in vocab or idx == 0:
                idx_to_word.append(word)
                word_to_idx[word] = idx
                weights.append(
                    np.asarray([vector], dtype=np.float32)
                )
                idx += 1

    weights = np.concatenate(weights, axis=0)

    emb_layer = nn.Embedding(
        num_embeddings=len(weights),
        embedding_dim=weights.shape[1]
    )

    emb_layer.load_state_dict({
        "weight": torch.from_numpy(weights)
    })

    if not randomized:
        emb_layer.requires_grad_(False)

    return emb_layer, idx_to_word, word_to_idx


def tokenize(sentence, word2idx, max_length=20):
    res = [0]

    sentence = sentence.lower()

    for word in sentence.split():
        if word in word2idx:
            res.append(word2idx[word])

    if len(res) > max_length:
        return res[:max_length]

    return res + [0] * (max_length - len(res))


def untokenize(tokens, idx2word):
    return [
        idx2word[x]
        for x in tokens
        if 0 <= x < len(idx2word)
    ]


# Création des embeddings et du vocabulaire indexé.
pretrained_embeddings, idx2word, word2idx = (
    create_embedding_layer(vocabulary, RANDOMIZED)
)

print("Vocabulary size:", len(idx2word))
print("Embedding dimensions:", pretrained_embeddings.embedding_dim)
print("Embeddings trainable:", pretrained_embeddings.weight.requires_grad)
