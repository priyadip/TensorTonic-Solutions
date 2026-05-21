import numpy as np
from typing import List, Dict


class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """

    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0

        # Special tokens — fixed IDs by convention
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words in sorted order.
        """
        # Step 1: reserve IDs 0-3 for special tokens.
        # enumerate() gives us (index, value) pairs starting at 0.
        special_tokens = [self.pad_token, self.unk_token,
                          self.bos_token, self.eos_token]
        for idx, tok in enumerate(special_tokens):
            self.word_to_id[tok] = idx
            self.id_to_word[idx] = tok

        # Step 2: gather all unique words across all texts.
        # set() deduplicates; |= is the in-place union operator.
        unique_words = set()
        for text in texts:
            unique_words |= set(text.lower().split())

        # Step 3: sort for determinism — same input → same vocab every run.
        # Important for reproducible ML experiments.
        sorted_words = sorted(unique_words)

        # Step 4: assign IDs starting from 4 (after the 4 special tokens).
        next_id = len(special_tokens)  # = 4
        for word in sorted_words:
            self.word_to_id[word] = next_id
            self.id_to_word[next_id] = word
            next_id += 1

        # Step 5: record total vocab size — needed by the embedding layer downstream.
        self.vocab_size = len(self.word_to_id)

    def encode(self, text: str) -> List[int]:
        """
        Convert text to a list of token IDs.
        Unknown words map to <UNK> (ID 1).
        """
        unk_id = self.word_to_id[self.unk_token]
        # .get(key, default) returns default if key is missing — clean UNK handling.
        return [self.word_to_id.get(word, unk_id)
                for word in text.lower().split()]

    def decode(self, ids: List[int]) -> str:
        """
        Convert a list of IDs back to a space-joined string.
        Unknown IDs map to the <UNK> token literal.
        """
        unk_token = self.unk_token
        return " ".join(self.id_to_word.get(i, unk_token) for i in ids)