"""Rank candidate structures by matching measured 13C NMR peak lists."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Spectrum:
    id: str
    name: str
    formula: str
    smiles: str
    solvent: str
    peaks: tuple[float, ...]


@dataclass(frozen=True)
class Alignment:
    score: float
    matches: list[tuple[float, float]]
    extra_query_peaks: list[float]
    missing_reference_peaks: list[float]


def parse_peaks(text: str) -> tuple[float, ...]:
    parts = re.split(r"[,;\s]+", text.strip())
    peaks = tuple(sorted(float(part) for part in parts))
    if not all(math.isfinite(peak) and -100 <= peak <= 400 for peak in peaks):
        raise ValueError("Expected finite 13C chemical shifts between -100 and 400 ppm")
    return peaks


def read_spectra(path: Path, solvent: str) -> list[Spectrum]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        required = {"id", "name", "formula", "smiles", "solvent", "peaks_ppm"}
        if not required <= set(rows.fieldnames or ()):
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(required - set(rows.fieldnames or ())))}")
        spectra = []
        seen = set()
        for line, row in enumerate(rows, 2):
            if (row["solvent"] or "").strip().casefold() != solvent.casefold():
                continue
            try:
                fields = {key: (row[key] or "").strip() for key in required}
                if not all(fields.values()):
                    raise ValueError("Incomplete row")
                if fields["id"] in seen:
                    raise ValueError(f"Duplicate ID {fields['id']} in {solvent}")
                seen.add(fields["id"])
                spectra.append(Spectrum(fields["id"], fields["name"], fields["formula"],
                                        fields["smiles"], fields["solvent"],
                                        parse_peaks(fields["peaks_ppm"])))
            except ValueError as error:
                raise ValueError(f"{path}:{line}: {error}") from error
    if not spectra:
        raise ValueError(f"No spectra for {solvent!r} in {path}")
    return spectra


def align(query: tuple[float, ...], reference: tuple[float, ...], gap: float) -> Alignment:
    """Find the minimum-cost order-preserving alignment of two sorted peak lists."""
    n, m = len(query), len(reference)
    costs = [[0.0] * (m + 1) for _ in range(n + 1)]
    operations = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        costs[i][0] = i * gap
        operations[i][0] = "query"
    for j in range(1, m + 1):
        costs[0][j] = j * gap
        operations[0][j] = "reference"

    for i, query_peak in enumerate(query, 1):
        for j, reference_peak in enumerate(reference, 1):
            distance = abs(query_peak - reference_peak)
            alternatives = [(costs[i - 1][j] + gap, "query"),
                            (costs[i][j - 1] + gap, "reference")]
            if distance < 2 * gap:
                alternatives.insert(0, (costs[i - 1][j - 1] + distance, "match"))
            costs[i][j], operations[i][j] = min(alternatives, key=lambda item: item[0])

    i, j = n, m
    matches, extra, missing = [], [], []
    while i or j:
        operation = operations[i][j]
        if operation == "match":
            matches.append((query[i - 1], reference[j - 1]))
            i, j = i - 1, j - 1
        elif operation == "query":
            extra.append(query[i - 1])
            i -= 1
        else:
            missing.append(reference[j - 1])
            j -= 1
    score = max(0.0, 1 - costs[n][m] / ((n + m) * gap))
    return Alignment(score, matches[::-1], extra[::-1], missing[::-1])


def ranking(peaks: tuple[float, ...], candidates: list[Spectrum], gap: float
            ) -> list[tuple[Spectrum, Alignment]]:
    rows = [(candidate, align(peaks, candidate.peaks, gap)) for candidate in candidates]
    return sorted(rows, key=lambda row: (-row[1].score, row[0].id))


def histogram_similarity(query: tuple[float, ...], reference: tuple[float, ...]) -> float:
    """Cosine similarity of 5 ppm bins, used as a reproducible baseline."""
    query_bins = Counter(math.floor(peak / 5) for peak in query)
    ref_bins = Counter(math.floor(peak / 5) for peak in reference)
    dot = sum(count * ref_bins[index] for index, count in query_bins.items())
    norm = math.sqrt(sum(n ** 2 for n in query_bins.values()) * sum(n ** 2 for n in ref_bins.values()))
    return dot / norm


def rank_bounds(scores: list[float], correct: float) -> tuple[int, int]:
    ahead = sum(score > correct + 1e-9 for score in scores)
    tied = sum(abs(score - correct) <= 1e-9 for score in scores)
    return ahead + 1, ahead + tied


def benchmark(queries: list[Spectrum], references: list[Spectrum], gap: float,
              same_formula: bool, stress: bool, seed: int) -> dict:
    randomizer = random.Random(seed)
    details = []
    ref_ids = {candidate.id for candidate in references}
    for query in sorted(queries, key=lambda item: item.id):
        if query.id not in ref_ids:
            raise ValueError(f"No correct reference for query {query.id}")
        candidates = [ref for ref in references if not same_formula or ref.formula == query.formula]
        if len(candidates) < 2:
            continue
        peaks = list(query.peaks)
        if stress and len(peaks) >= 2:
            peaks.pop(randomizer.randrange(len(peaks)))
            peaks.append(randomizer.uniform(0, 210))
        peaks = tuple(sorted(peaks))
        ranked = ranking(peaks, candidates, gap)
        correct = next((result for ref, result in ranked if ref.id == query.id), None)
        if correct is None:
            raise ValueError(f"Reference formula differs for query {query.id}")
        hist_scores = {ref.id: histogram_similarity(peaks, ref.peaks) for ref in candidates}
        details.append({
            "id": query.id,
            "name": query.name,
            "candidates": len(candidates),
            "best_candidate": ranked[0][0].id,
            "best_score": round(ranked[0][1].score, 6),
            "correct_score": round(correct.score, 6),
            "alignment_rank": rank_bounds([result.score for _, result in ranked], correct.score),
            "histogram_rank": rank_bounds(list(hist_scores.values()), hist_scores[query.id]),
        })
    if not details:
        raise ValueError("No eligible test cases with at least two candidates")

    def metrics(key: str) -> dict:
        worst = [row[key][1] for row in details]
        return {
            "top_1": sum(rank == 1 for rank in worst),
            "top_3": sum(rank <= 3 for rank in worst),
            "mrr": statistics.mean(1 / rank for rank in worst),
        }

    return {
        "cases": len(details),
        "same_formula": same_formula,
        "stress": stress,
        "seed": seed if stress else None,
        "gap_ppm": gap,
        "reference_solvent": references[0].solvent,
        "query_solvent": queries[0].solvent,
        "alignment": metrics("alignment_rank"),
        "histogram_baseline": metrics("histogram_rank"),
        "details": details,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("rank", help="Rank candidates for a list of experimental peaks")
    measure = commands.add_parser("benchmark", help="Use real spectra measured in a different solvent")
    for command in (search, measure):
        command.add_argument("--data", type=Path, default=Path(__file__).parent / "data" / "solvents.csv")
        command.add_argument("--reference-solvent", default="CDCl3")
        command.add_argument("--gap-ppm", type=float, default=2.0)
    search.add_argument("--peaks", required=True, help="Comma-separated 13C shifts in ppm")
    search.add_argument("--formula", help="Optional molecular formula, e.g. C4H8O")
    search.add_argument("--top", type=int, default=5)
    measure.add_argument("--query-solvent", default="C6D6")
    measure.add_argument("--same-formula", action="store_true")
    measure.add_argument("--stress", action="store_true", help="Replace one signal in each list of at least two")
    measure.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    try:
        if not math.isfinite(args.gap_ppm) or args.gap_ppm <= 0:
            raise ValueError("--gap-ppm must be finite and positive")
        references = read_spectra(args.data, args.reference_solvent.strip())
        if args.command == "benchmark":
            if args.reference_solvent.strip().casefold() == args.query_solvent.strip().casefold():
                raise ValueError("Use different solvents for the query and reference spectra")
            queries = read_spectra(args.data, args.query_solvent.strip())
            result = benchmark(queries, references, args.gap_ppm, args.same_formula,
                               args.stress, args.seed)
        else:
            if args.top <= 0:
                raise ValueError("--top must be positive")
            peaks = parse_peaks(args.peaks)
            candidates = [ref for ref in references if args.formula is None or ref.formula == args.formula.strip()]
            if not candidates:
                raise ValueError("No candidates have the requested formula")
            ranked = ranking(peaks, candidates, args.gap_ppm)
            best = ranked[0][1].score
            ties = sum(abs(item.score - best) <= 1e-9 for _, item in ranked)
            result = {
                "candidates": len(ranked),
                "top_tie": ties > 1,
                "no_peak_matches": not ranked[0][1].matches,
                "results": [
                    {"id": ref.id, "name": ref.name, "formula": ref.formula, "smiles": ref.smiles,
                     "solvent": ref.solvent, **asdict(match)}
                    for ref, match in ranked[:max(args.top, ties)]
                ],
            }
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        print(f"nmr-ranker: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
