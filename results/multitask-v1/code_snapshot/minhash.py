"""A readable MinHash baseline using independently salted SHA-256 hashes."""

import hashlib
import json


def minhash_signature(
    items: set[tuple[str, ...]], size: int = 128, seed: int = 0
) -> list[int] | None:
    """Keep one minimum per hash function. None means no shingles to compare.

    SHA-256 with a different salt per position approximates independent random
    rankings. This favors clarity over speed; it is not a production hash family.
    Every document must use the same size, seed, and shingle representation.
    """
    if size < 1 or seed < 0:
        raise ValueError("size must be positive and seed must be nonnegative")
    if not items:
        return None

    encoded = [json.dumps(item, ensure_ascii=False).encode("utf-8") for item in items]
    signature = []
    for position in range(size):
        salt = f"minhash-v1:{seed}:{position}:".encode("ascii")
        minimum = min(
            int.from_bytes(hashlib.sha256(salt + item).digest()[:8], "big")
            for item in encoded
        )
        signature.append(minimum)
    return signature


def estimate_jaccard(left: list[int] | None, right: list[int] | None) -> float:
    """Fraction of corresponding positions that match, not numeric closeness."""
    if left is None or right is None:
        return 0.0
    if not left or len(left) != len(right):
        raise ValueError("Signatures must have the same positive length")
    return sum(a == b for a, b in zip(left, right)) / len(left)
