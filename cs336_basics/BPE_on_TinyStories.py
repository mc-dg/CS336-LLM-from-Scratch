import os
import pickle
from collections.abc import Iterator
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import BinaryIO

import regex as re

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

special_tokens = ["<|endoftext|>"]


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def remove_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """Remove special tokens from text and split on them."""
    if not special_tokens:
        return [text]
    pattern = "|".join(re.escape(token) for token in special_tokens)
    return [seg for seg in re.split(pattern, text) if seg]


def pretokenize_text(chunks: list[str], pattern: re.Pattern = PAT) -> Iterator[bytes]:
    """Pre-tokenize text chunks using regex pattern in a memory-efficient stream."""
    for chunk in chunks:
        for match in pattern.finditer(chunk):
            yield match.group().encode("utf-8")


def compute_word_frequencies(text: Iterator[bytes]) -> dict[tuple[bytes, ...], int]:
    """Compute frequency of each word (represented as tuple of bytes)."""
    word_frequencies: dict[tuple[bytes, ...], int] = {}
    for word in text:
        word_tuple = tuple(bytes([b]) for b in word)
        word_frequencies[word_tuple] = word_frequencies.get(word_tuple, 0) + 1
    return word_frequencies


def get_stats(word_frequencies: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    """Count frequency of each pair of consecutive bytes across all words."""
    couples: dict[tuple[bytes, bytes], int] = {}
    for word, freq in word_frequencies.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            couples[pair] = couples.get(pair, 0) + freq
    return couples


def merge_word(
    word: tuple[bytes, ...], pair: tuple[bytes, bytes], new_token: bytes
) -> tuple[bytes, ...]:
    """Merge a specific pair in a word."""
    new_word = []
    i = 0
    first, second = pair
    while i < len(word):
        if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
            new_word.append(new_token)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def merge_and_update(
    word_freq: dict[tuple[bytes, ...], int],
    stats: dict[tuple[bytes, bytes], int],
    best_pair: tuple[bytes, bytes],
    new_token: bytes,
) -> tuple[dict[tuple[bytes, ...], int], dict[tuple[bytes, bytes], int]]:
    """
    Merge best_pair in all words and update pair statistics.
    Returns new word_freq and updated stats.
    """
    first, second = best_pair
    new_wf = {}

    for word, freq in word_freq.items():
        # Only process words that contain both bytes of the pair
        if first not in word or second not in word:
            new_wf[word] = new_wf.get(word, 0) + freq
            continue

        merged = merge_word(word, best_pair, new_token)

        # Update stats: subtract old pairs, add new pairs
        if merged != word:
            # Subtract pairs from original word
            for p in zip(word, word[1:]):
                stats[p] = stats.get(p, 0) - freq
                if stats[p] <= 0:
                    del stats[p]
            # Add pairs from merged word
            for p in zip(merged, merged[1:]):
                stats[p] = stats.get(p, 0) + freq

        new_wf[merged] = new_wf.get(merged, 0) + freq

    return new_wf, stats


def chunk_generator(f, boundaries):
    """Generate chunks of text from file."""
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        chunk = chunk.replace("\r", "")
        yield chunk


def worker_pre_tokenization(chunk: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    """Worker: pre-tokenize chunk and compute word frequencies."""
    chunk_no_st = remove_special_tokens(chunk, special_tokens)
    pretokenized_chunk = pretokenize_text(chunk_no_st)
    return compute_word_frequencies(pretokenized_chunk)


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.

    Returns:
        vocab: dict[int, bytes] - mapping from token ID to token bytes
        merges: list[tuple[bytes, bytes]] - list of merge operations in order
    """
    num_processes = max(1, cpu_count() - 4)
    word_freq = {}

    # Step 1: Pre-tokenization with multiprocessing
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        with Pool(processes=num_processes) as p:
            worker = partial(worker_pre_tokenization, special_tokens=special_tokens)
            for partial_freq in p.imap_unordered(worker, chunk_generator(f, boundaries)):
                # Merge partial frequencies: sum, don't overwrite
                for w, c in partial_freq.items():
                    word_freq[w] = word_freq.get(w, 0) + c

    # Step 2: Initialize vocabulary with bytes
    vocab = {i: bytes([i]) for i in range(256)}

    # Step 3: Add special tokens to vocabulary
    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")

    # Step 4: Compute number of merges needed
    num_merges = vocab_size - len(vocab)
    merges: list[tuple[bytes, bytes]] = []

    # Step 5: Compute initial pair statistics
    stats = get_stats(word_freq)

    # Step 6: Iteratively merge most frequent pairs
    for i in range(num_merges):
        if not stats:
            break

        # Tie-break lexicographically (per Stanford CS336 PDF)
        best_pair = max(stats, key=lambda p: (stats[p], p))
        new_token = best_pair[0] + best_pair[1]

        # Record merge
        merges.append(best_pair)

        # Add to vocabulary
        vocab[len(vocab)] = new_token

        # Update word frequencies and statistics
        word_freq, stats = merge_and_update(word_freq, stats, best_pair, new_token)

    return vocab, merges


if __name__ == "__main__":
    dataset_path = "C:\\Users\\maric\\OneDrive\\Desktop\\CS336-LLM-from-Scratch\\CS336-LLM-from-Scratch\\data\\TinyStoriesV2-GPT4-valid.txt"
    vocab_path = "C:\\Users\\maric\\OneDrive\\Desktop\\CS336-LLM-from-Scratch\\CS336-LLM-from-Scratch\\data\\vocab.pkl"
    merge_path = "C:\\Users\\maric\\OneDrive\\Desktop\\CS336-LLM-from-Scratch\\CS336-LLM-from-Scratch\\data\\merge.pkl"

    vocab_size = 10000

    vocab, merges = train_bpe(dataset_path, vocab_size, special_tokens)

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merge_path, "wb") as f:
        pickle.dump(merges, f)

    print(f"Vocab size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print(f"First 10 merges: {merges[:10]}")
