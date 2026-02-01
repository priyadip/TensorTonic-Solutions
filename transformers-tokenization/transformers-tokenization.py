from typing import List, Dict

class SimpleTokenizer:
    def __init__(self):
        self.id_to_word: Dict[int, str] = {}
        self.word_to_id: Dict[str, int] = {}
        self.vocab_size = 0

        # Special tokens (fixed order)
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # Reset vocab
        self.id_to_word = {}
        self.word_to_id = {}
        self.vocab_size = 0

        # Add special tokens FIRST
        for token in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
            self.word_to_id[token] = self.vocab_size
            self.id_to_word[self.vocab_size] = token
            self.vocab_size += 1

        # Collect unique words
        words = set()
        for text in texts:
            words.update(text.lower().split())

        # Add words (sorted for determinism)
        for word in sorted(words):
            self.word_to_id[word] = self.vocab_size
            self.id_to_word[self.vocab_size] = word
            self.vocab_size += 1

    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Unknown words map to <UNK>.
        """
        tokens = text.lower().split()
        unk_id = self.word_to_id[self.unk_token]
        return [self.word_to_id.get(token, unk_id) for token in tokens]

    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        Skip <PAD> tokens.
        """
        words = []
        for idx in ids:
            word = self.id_to_word.get(idx, self.unk_token)
            if word != self.pad_token:
                words.append(word)
        return " ".join(words)
