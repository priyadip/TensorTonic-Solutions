from typing import List, Dict

class WordPieceTokenizer:
    """
    WordPiece tokenizer for BERT.
    """
    
    def __init__(self, vocab: Dict[str, int], unk_token: str = "[UNK]", max_word_len: int = 100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_word_len = max_word_len
    
    def tokenize(self, text: str) -> List[str]:
        tokens = []
        for word in text.lower().split():
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        return tokens

    def _tokenize_word(self, word: str) -> List[str]:
        if len(word) > self.max_word_len:
            return [self.unk_token]

        tokens = []
        start = 0

        while start < len(word):
            end = len(word)
            matched = None

            while end > start:
                piece = word[start:end]
                if start > 0:
                    piece = "##" + piece

                if piece in self.vocab:
                    matched = piece
                    break

                end -= 1

            if matched is None:
                return [self.unk_token]

            tokens.append(matched)
            start = end

        return tokens