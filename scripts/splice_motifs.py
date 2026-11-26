"""Infer the strand of an intron from its terminal dinucleotides.

Shared by ``shift_unstranded_usage.py`` and ``convert_splice_junctions_to_parquet.py``
for unstranded Spliser runs (``Strand`` = ``?``).

Coordinates are raw Spliser ``Site`` values (before the −1 left-end shift): for an
intron with left end ``s`` and right end ``p`` (``s < p``), the first two intron
bases are ``genome[s:s+2]`` and the last two are ``genome[p-2:p]`` (0-based slices).
Checked on Khrameeva human: 98.9% of junctions are GT-AG or CT-AC at this offset
and none at ±1.
"""

from __future__ import annotations

from pathlib import Path

# (first two intron bases on the + strand, last two) → strand
INTRON_MOTIFS: dict[tuple[str, str], str] = {
    ("GT", "AG"): "+", ("GC", "AG"): "+", ("AT", "AC"): "+",
    ("CT", "AC"): "-", ("CT", "GC"): "-", ("GT", "AT"): "-",
}


class MotifStrand:
    """Cached ``(chrom, left, right) → '+' | '-' | None`` lookup on a FASTA."""

    def __init__(self, genome: str | Path) -> None:
        import pyfaidx

        self._fa = pyfaidx.Fasta(str(genome), as_raw=True, sequence_always_upper=True)
        self._names = set(self._fa.keys())
        self._chrom_cache: dict[str, str | None] = {}
        self._cache: dict[tuple[str, int, int], str | None] = {}

    def _fasta_name(self, chrom: str) -> str | None:
        if chrom not in self._chrom_cache:
            name = None
            for cand in (chrom, f"chr{chrom}", chrom.removeprefix("chr")):
                if cand in self._names:
                    name = cand
                    break
            self._chrom_cache[chrom] = name
        return self._chrom_cache[chrom]

    def __call__(self, chrom: str, left: int, right: int) -> str | None:
        key = (chrom, left, right)
        if key not in self._cache:
            name = self._fasta_name(chrom)
            strand = None
            if name is not None and right - left >= 4 and left >= 0:
                seq = self._fa[name]
                strand = INTRON_MOTIFS.get((seq[left:left + 2], seq[right - 2:right]))
            self._cache[key] = strand
        return self._cache[key]
