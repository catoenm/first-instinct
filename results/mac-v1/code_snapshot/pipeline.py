"""Miniature pretraining data pipeline, built one step at a time."""

import hashlib
import json
import re
from itertools import combinations
from pathlib import Path

from minhash import estimate_jaccard, minhash_signature


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_DIR / "data" / "raw.jsonl"
OUTPUT_DIR = PROJECT_DIR / "output"
SIGNATURE_SIZES = (16, 64, 256)


def main() -> None:
    documents = load_documents(INPUT_PATH)
    documents = process_documents(documents)
    retained, decisions = deduplicate(documents)
    similarities = compare_documents(retained)

    save_documents(retained, OUTPUT_DIR / "corpus.jsonl")
    save_documents(decisions, OUTPUT_DIR / "decisions.jsonl")
    save_documents(similarities, OUTPUT_DIR / "similarities.jsonl")

    print(f"Loaded {len(documents)} documents; retained {len(retained)}.")
    for decision in decisions:
        print(f"  {decision['source_id']}: {decision['reason']}"
              f" -> {decision['retained_id']}")
    print("\nExact Jaccard similarity of word-trigram sets (survivors):")
    for pair in similarities:
        print(f"  {pair['left_id']} / {pair['right_id']}: {pair['jaccard']:.3f}"
              f" ({pair['intersection_size']}/{pair['union_size']})")
        for size, estimate in pair["minhash_estimates"].items():
            print(f"    MinHash {size:>3} positions: {estimate:.3f}"
                  f" (absolute error {abs(estimate - pair['jaccard']):.3f})")
    print(f"\nOutputs: {OUTPUT_DIR}")


def process_documents(documents: list[dict]) -> list[dict]:
    enriched = []
    for document in documents:
        normalized = normalize_text(document["text"])
        enriched.append({
            **document,
            "raw_sha256": hash_text(document["text"]),
            "normalized_text": normalized,
            "normalized_sha256": hash_text(normalized),
        })
    return enriched


def normalize_text(text: str) -> str:
    """Normalize line endings and trailing spaces/tabs; preserve indentation."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip(" \t") for line in text.split("\n"))


def deduplicate(documents: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep the first source in input order for each normalized content hash."""
    seen = {}
    retained = []
    decisions = []
    for document in documents:
        fingerprint = document["normalized_sha256"]
        representative = seen.get(fingerprint)
        if representative is None:
            seen[fingerprint] = document
            retained.append(document)
            decisions.append({
                "source_id": document["source_id"],
                "retained_id": document["source_id"],
                "reason": "retained",
            })
        else:
            reason = ("exact_match" if document["raw_sha256"] == representative["raw_sha256"]
                      else "normalized_match")
            decisions.append({
                "source_id": document["source_id"],
                "retained_id": representative["source_id"],
                "reason": reason,
            })
    return retained, decisions


def shingles(text: str, size: int = 3) -> set[tuple[str, ...]]:
    """Overlapping word sequences; this comparison view ignores case/punctuation."""
    if size < 1:
        raise ValueError("Shingle size must be positive")
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i:i + size]) for i in range(len(words) - size + 1)}


def jaccard(left: set, right: set) -> float:
    union = left | right
    # Empty shingle sets provide no evidence of shared content.
    return len(left & right) / len(union) if union else 0.0


def compare_documents(documents: list[dict]) -> list[dict]:
    """All pairs is an inspection baseline, not the eventual scalable search."""
    fingerprints = {
        document["source_id"]: shingles(document["normalized_text"])
        for document in documents
    }
    # Build once per document, then reuse across all pair comparisons.
    signatures = {
        source_id: minhash_signature(items, size=max(SIGNATURE_SIZES))
        for source_id, items in fingerprints.items()
    }
    pairs = []
    for left, right in combinations(documents, 2):
        a, b = fingerprints[left["source_id"]], fingerprints[right["source_id"]]
        signature_a = signatures[left["source_id"]]
        signature_b = signatures[right["source_id"]]
        pairs.append({
            "left_id": left["source_id"],
            "right_id": right["source_id"],
            "intersection_size": len(a & b),
            "union_size": len(a | b),
            "jaccard": jaccard(a, b),
            "minhash_estimates": {
                str(size): estimate_jaccard(
                    signature_a[:size] if signature_a is not None else None,
                    signature_b[:size] if signature_b is not None else None,
                )
                for size in SIGNATURE_SIZES
            },
        })
    return pairs


def save_documents(documents: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for document in documents:
            file.write(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n")


def load_documents(path: Path) -> list[dict]:
    documents = []
    source_ids = set()
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            document = json.loads(line)
            if not isinstance(document, dict) or not all(
                isinstance(document.get(key), str) for key in ("source_id", "text")
            ):
                raise ValueError(f"Line {line_number}: expected source_id and text strings")
            if document["source_id"] in source_ids:
                raise ValueError(f"Line {line_number}: duplicate source_id")
            source_ids.add(document["source_id"])
            documents.append(document)
    return documents


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
