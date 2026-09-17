"""MinHash banding: generate candidates with buckets, then verify their overlap."""

import argparse
from collections import defaultdict
from itertools import combinations

from minhash import minhash_signature
from pipeline import (
    INPUT_PATH, OUTPUT_DIR, deduplicate, jaccard, load_documents,
    process_documents, save_documents, shingles,
)


def candidate_pairs(
    signatures: dict[str, list[int] | None], rows_per_band: int
) -> set[tuple[str, str]]:
    """A pair qualifies when every position matches in at least one band.

    All signatures must use the same seed and hash family. Empty documents have
    no signature and do not enter buckets. Buckets can still produce quadratic
    numbers of pairs when many documents share a band.
    """
    if rows_per_band < 1:
        raise ValueError("rows_per_band must be positive")
    lengths = {len(s) for s in signatures.values() if s is not None}
    if len(lengths) > 1 or any(n == 0 or n % rows_per_band for n in lengths):
        raise ValueError("Signatures must have equal positive lengths divisible by rows_per_band")

    buckets = defaultdict(list)
    for source_id, signature in sorted(signatures.items()):
        if signature is None:
            continue
        for start in range(0, len(signature), rows_per_band):
            # Include the band index: equal values in different positions do not match.
            key = (start // rows_per_band, tuple(signature[start:start + rows_per_band]))
            buckets[key].append(source_id)

    # One pair may share several bands; emit it only once.
    return {pair for members in buckets.values() for pair in combinations(members, 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Exact Jaccard cutoff used after candidate retrieval")
    args = parser.parse_args()
    if not 0 < args.threshold <= 1:
        parser.error("threshold must be in (0, 1]")
    if args.seed < 0:
        parser.error("seed must be nonnegative")

    documents, _ = deduplicate(process_documents(load_documents(INPUT_PATH)))
    sets = {d["source_id"]: shingles(d["normalized_text"]) for d in documents}
    signatures = {key: minhash_signature(items, size=256, seed=args.seed)
                  for key, items in sets.items()}

    # Evaluation only: exhaustively find true matches in this tiny fixture.
    # The candidate generator above never calls this or compares every pair.
    exact = {(a, b): jaccard(sets[a], sets[b]) for a, b in combinations(sorted(sets), 2)}
    truth = {pair for pair, score in exact.items() if score >= args.threshold}

    print(f"Seed {args.seed}; {len(documents)} documents; {len(exact)} possible pairs.")
    print(f"Exact verification threshold: {args.threshold}; {len(truth)} true matching pairs.")
    print("\nBands x rows | Candidates | Verified | Missed | Recall")
    reports = []
    for rows in (16, 8, 4):
        bands = 256 // rows
        candidates = candidate_pairs(signatures, rows)
        # Actual verification work: exact overlap only for retrieved candidates.
        verified = {pair for pair in candidates
                    if jaccard(sets[pair[0]], sets[pair[1]]) >= args.threshold}
        missed = truth - verified
        recall = len(verified) / len(truth) if truth else None
        recall_label = f"{recall:.0%}" if recall is not None else "n/a"
        print(f"{bands:>5} x {rows:<4} | {len(candidates):>10} | {len(verified):>8}"
              f" | {len(missed):>6} | {recall_label}")
        reports.append({
            "seed": args.seed,
            "signature_size": 256,
            "bands": bands,
            "rows_per_band": rows,
            "verification_threshold": args.threshold,
            "possible_pairs": len(exact),
            "candidate_pairs": sorted(candidates),
            "verified_pairs": sorted(verified),
            "rejected_pairs": sorted(candidates - verified),
            "missed_pairs": sorted(missed),
            "recall": recall,
        })
    save_documents(reports, OUTPUT_DIR / "lsh_experiment.jsonl")
    print(f"\nReport: {OUTPUT_DIR / 'lsh_experiment.jsonl'}")
    print("Only the evaluation baseline uses all pairs. No documents are removed by LSH.")


if __name__ == "__main__":
    main()
