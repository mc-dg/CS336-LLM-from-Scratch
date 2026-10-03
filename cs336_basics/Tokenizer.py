from collections.abc import Iterable, Iterator
class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
        
def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
    pass

def encode(self, text: str) -> list[int]:
    pass

def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
    pass

def decode(self, ids: list[int]) -> str:
    pass