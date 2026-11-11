import unittest
from pathlib import Path

from nmr_ranker import align, benchmark, parse_peaks, rank_bounds, read_spectra

DATA = Path(__file__).resolve().parents[1] / "data" / "solvents.csv"


class RankerTests(unittest.TestCase):
    def test_match_never_reuses_a_peak(self):
        match = align((10.0, 10.5), (10.0,), 2.0)
        self.assertEqual(match.matches, [(10.0, 10.0)])
        self.assertEqual(match.extra_query_peaks, [10.5])
        self.assertAlmostEqual(match.score, 2 / 3)

    def test_extra_and_missing_signals_are_reported(self):
        match = align((10.0, 20.3, 60.0), (10.1, 20.0, 99.0), 2.0)
        self.assertEqual(match.matches, [(10.0, 10.1), (20.3, 20.0)])
        self.assertEqual(match.extra_query_peaks, [60.0])
        self.assertEqual(match.missing_reference_peaks, [99.0])

    def test_ties_have_conservative_rank(self):
        self.assertEqual(rank_bounds([1, 0.8, 0.8, 0.2], 0.8), (2, 3))

    def test_parse_accepts_common_separators_and_rejects_bad_numbers(self):
        self.assertEqual(parse_peaks("60.3, 10.1; 22.9"), (10.1, 22.9, 60.3))
        for invalid in ("", "NaN", "Infinity", "700"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_peaks(invalid)

    def test_cross_solvent_benchmark_contains_only_real_pairs(self):
        queries = read_spectra(DATA, "DMSO-d6")
        references = read_spectra(DATA, "CDCl3")
        self.assertEqual(len(references), 38)
        correct_reference = {row.id: row for row in references}
        self.assertTrue(all(row.peaks != correct_reference[row.id].peaks for row in queries))
        result = benchmark(queries, references, 2, True, False, 42)
        self.assertEqual(result["cases"], 6)
        self.assertTrue(all(row["candidates"] == 2 for row in result["details"]))


if __name__ == "__main__":
    unittest.main()
