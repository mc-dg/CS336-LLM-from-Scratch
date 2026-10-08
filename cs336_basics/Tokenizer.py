import pickle
from collections.abc import Iterable, Iterator

import regex as re

# Usa il pattern BPE standard di GPT-2
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens=None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.byte_to_id = {v: k for k, v in vocab.items()}

        # Costruisci una mappa rapida dei merge per accelerare l'encoding
        self.merges_ranks = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """Load tokenizer from pickled vocab and merges."""
        with open(vocab_filepath, "rb") as file:
            vocab = pickle.load(file)
        with open(merges_filepath, "rb") as file:
            merges = pickle.load(file)
        return cls(vocab, merges, special_tokens)

    def _split_on_special_tokens(self, text: str) -> list[str]:
        """Split text on special tokens, longest first (greedy)."""
        if not self.special_tokens:
            return [text]

        sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "|".join(re.escape(st) for st in sorted_tokens)
        parts = re.split(f"({pattern})", text)
        return [p for p in parts if p]

    def _encode_chunk(self, piece: bytes) -> list[bytes]:
        """Applica i merge BPE unicamente all'interno di questo singolo segmento/parola."""
        if len(piece) <= 1:
            return [piece]

        # Inizializza come lista di singoli byte
        tokens = [bytes([b]) for b in piece]

        while len(tokens) >= 2:
            # Trova la coppia di token adiacenti con il rank di merge più basso (priorità maggiore)
            stats = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges_ranks:
                    stats[pair] = self.merges_ranks[pair]

            if not stats:
                break  # Nessuna coppia presente nei merge BPE

            # Prendi la coppia con rank minimo
            best_pair = min(stats, key=stats.get)
            first, second = best_pair
            new_token = first + second

            # Sostituisci tutte le occorrenze di best_pair
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == first and tokens[i + 1] == second:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs, esattamente come tiktoken."""
        if not text:
            return []

        token_bytes = []
        segments = self._split_on_special_tokens(text)

        for segment in segments:
            if segment in self.special_tokens:
                token_bytes.append(segment.encode("utf-8"))
            else:
                # Pre-tokenizza il segmento con il Regex
                for match in PAT.finditer(segment):
                    word_bytes = match.group().encode("utf-8")
                    # Applica i merge BPE ISOLATAMENTE alla singola parola/match
                    merged_word = self._encode_chunk(word_bytes)
                    token_bytes.extend(merged_word)

        return self._bytes_to_ids(token_bytes)

    def _bytes_to_ids(self, token_bytes: list[bytes]) -> list[int]:
        """Convert list of bytes to token IDs."""
        result = []
        for b in token_bytes:
            if b in self.byte_to_id:
                result.append(self.byte_to_id[b])
            else:
                raise ValueError(f"Token {b!r} not in vocabulary")
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode iterable of texts, yielding token IDs one at a time."""
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        token_bytes = b"".join(self.vocab[tid] for tid in ids)
        return token_bytes.decode("utf-8", errors="replace")
