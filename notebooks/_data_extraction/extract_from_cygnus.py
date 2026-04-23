"""Extract demo data for the vega-mir notebooks from a local CYGNUS install.

This is a one-off helper. Most users will never run it: they consume
the bundled JSON files in ``notebooks/data/`` directly. The extraction
is documented here so the data provenance is transparent.

Inputs (must exist on the local machine):
    /Users/fredmacbook/Desktop/CYGNUS/src/cygnus/profiler.py
    /Users/fredmacbook/Desktop/CYGNUS/profiles/maestro/<slug>.json
    /Users/fredmacbook/Desktop/CYGNUS/profiles/<slug>.json
    /Users/fredmacbook/Desktop/CYGNUS/corpus/<track_id>/final.json
    /Users/fredmacbook/Desktop/CYGNUS/reports/kl_analysis_20260417/*.json

Outputs:
    notebooks/data/composer_pieces.json    (5 sample pieces per composer, raw sequences)
    notebooks/data/composer_counts.json    (full aggregated marginal + joint counts per composer)
    notebooks/data/paper_reference.json    (subset of published values for parity comparison)

Methodology mirrors ``CYGNUS/scripts/shannon_zipf_v2.py``: per-piece
scale-degree sequences collapsed (consecutive duplicates merged),
mapped relative to the detected key via Cygnus's harmony helpers.

Run from the vega-mir repo root:
    PYTHONPATH=/Users/fredmacbook/Desktop/CYGNUS/src python3.11 \\
        notebooks/_data_extraction/extract_from_cygnus.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CYGNUS_REPO = Path("/Users/fredmacbook/Desktop/CYGNUS")
sys.path.insert(0, str(CYGNUS_REPO / "src"))

from cygnus.profiler import (  # noqa: E402  (sys.path manipulation)
    _chord_root_semitone,
    _interval_to_degree,
    _parse_key,
)

CORPUS_DIR = CYGNUS_REPO / "corpus"
PROFILES_DIR = CYGNUS_REPO / "profiles"
REPORTS_DIR = CYGNUS_REPO / "reports" / "kl_analysis_20260417"

# Curated 8 composers spanning the paper's flagship findings:
#   - Shannon range (3.33-3.86 bits): Bach low end, Liszt high end
#   - KL stylistic pairs: Haydn-Beethoven, Liszt-Rachmaninoff
#   - Zipf gap historical vs neoclassical: Bach/Chopin vs Glass/Richter
COMPOSERS = [
    # (display_name, slug, profile_subdir)
    ("Johann Sebastian Bach", "johann_sebastian_bach", "maestro"),
    ("Joseph Haydn", "joseph_haydn", "maestro"),
    ("Ludwig van Beethoven", "ludwig_van_beethoven", "maestro"),
    ("Frédéric Chopin", "frederic_chopin", "maestro"),
    ("Franz Liszt", "franz_liszt", "maestro"),
    ("Sergei Rachmaninoff", "sergei_rachmaninoff", "maestro"),
    ("Philip Glass", "philip_glass", ""),
    ("Max Richter", "max_richter", ""),
]

N_PIECES_PER_COMPOSER = 5
MIN_SEQUENCE_LENGTH = 30  # enough to be meaningful for KS tests, FFT, etc.


def extract_sequence(final_data: dict) -> list[str] | None:
    """Extract a collapsed scale-degree sequence from a Cygnus final.json."""
    harmony = final_data.get("harmony", {})
    unified = harmony.get("unified", [])
    key_str = harmony.get("stats", {}).get("key", "")
    parsed = _parse_key(key_str) if key_str else None
    if not parsed:
        return None
    tonic, mode = parsed

    sequence: list[str] = []
    prev: str | None = None
    for entry in unified:
        chord = entry.get("chord", "")
        root = _chord_root_semitone(chord)
        if root is None or chord == "N":
            prev = None
            continue
        interval = (root - tonic) % 12
        degree = _interval_to_degree(interval, mode)
        if degree != prev:
            sequence.append(degree)
        prev = degree
    return sequence if sequence else None


def extract_composer(display_name: str, slug: str, subdir: str) -> tuple[dict, dict]:
    """Extract sample pieces and full aggregated counts for one composer.

    Returns
    -------
    sample : dict
        ``N_PIECES_PER_COMPOSER`` representative pieces with raw scale-degree
        sequences. For the pedagogical notebook.
    counts : dict
        Aggregated marginal + joint (bigram) counts across **all** pieces of
        the composer. For exact parity with the paper artefacts.
    """
    if subdir:
        profile_path = PROFILES_DIR / subdir / f"{slug}.json"
    else:
        profile_path = PROFILES_DIR / f"{slug}.json"
    profile = json.loads(profile_path.read_text())
    track_ids = profile.get("corpus_ids", [])

    sample_pieces: list[dict] = []
    marginal: dict[str, int] = {}
    joint: dict[str, int] = {}
    n_pieces_aggregated = 0

    for tid in track_ids:
        final_path = CORPUS_DIR / tid / "final.json"
        if not final_path.exists():
            continue
        try:
            final_data = json.loads(final_path.read_text())
        except json.JSONDecodeError:
            continue
        seq = extract_sequence(final_data)
        if seq is None or len(seq) < MIN_SEQUENCE_LENGTH:
            continue

        # Aggregate counts across ALL valid pieces (marginal + bigram joint).
        for d in seq:
            marginal[d] = marginal.get(d, 0) + 1
        for src, tgt in zip(seq[:-1], seq[1:]):
            key = f"{src}>{tgt}"
            joint[key] = joint.get(key, 0) + 1
        n_pieces_aggregated += 1

        # Keep the first N as the sample bundle for the pedagogical notebook.
        if len(sample_pieces) < N_PIECES_PER_COMPOSER:
            sample_pieces.append(
                {
                    "track_id": tid,
                    "scale_degrees": seq,
                    "n_observations": len(seq),
                }
            )

    sample = {
        "display_name": display_name,
        "n_pieces_total_in_corpus": len(track_ids),
        "n_pieces_sampled": len(sample_pieces),
        "total_observations": sum(p["n_observations"] for p in sample_pieces),
        "pieces": sample_pieces,
    }
    counts = {
        "display_name": display_name,
        "n_pieces": n_pieces_aggregated,
        "marginal_counts": marginal,
        "joint_counts": joint,
        "n_marginal_observations": sum(marginal.values()),
        "n_joint_observations": sum(joint.values()),
    }
    return sample, counts


def load_paper_reference(composer_names: list[str]) -> dict:
    """Pull subset of published Shannon / Zipf / KL values for the bundled composers.

    Block layout (verified 2026-04-23):
        shannon_scale_degrees_33.json -> ["entropy_bits"][composer]
            -> {H, ci_95, n_pieces, low_sample}
        zipf_scale_degrees_33.json    -> ["zipf_marginal"][composer]
            -> {alpha, R2, intercept, n_points, ranked_degrees}
        zipf_transitions_33.json      -> ["zipf_transitions"][composer]
            -> {alpha, R2, intercept, n_points, top_transitions}
        kl_scale_degrees_33x33.json   -> ["matrix"][src][tgt] = float
    """
    reference: dict = {}

    sources = [
        ("shannon_scale_degrees_33.json", "entropy_bits", "shannon_published"),
        ("zipf_scale_degrees_33.json", "zipf_marginal", "zipf_marginal_published"),
        ("zipf_transitions_33.json", "zipf_transitions", "zipf_transitions_published"),
    ]
    for fname, block_key, ref_key in sources:
        path = REPORTS_DIR / fname
        if not path.exists():
            continue
        published = json.loads(path.read_text())
        block = published.get(block_key, {})
        reference[ref_key] = {
            name: block[name] for name in composer_names if name in block
        }

    kl_path = REPORTS_DIR / "kl_scale_degrees_33x33.json"
    if kl_path.exists():
        published = json.loads(kl_path.read_text())
        matrix_block = published.get("matrix", {})
        sub_matrix: dict = {}
        for src in composer_names:
            if src not in matrix_block:
                continue
            sub_matrix[src] = {
                tgt: matrix_block[src][tgt]
                for tgt in composer_names
                if tgt in matrix_block[src]
            }
        reference["kl_marginal_published"] = sub_matrix

    return reference


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_data: dict = {}
    counts_data: dict = {}
    for display, slug, subdir in COMPOSERS:
        print(f"Extracting {display}...")
        sample, counts = extract_composer(display, slug, subdir)
        sample_data[display] = sample
        counts_data[display] = counts
        print(
            f"  -> sample: {sample['n_pieces_sampled']} pieces, "
            f"{sample['total_observations']} obs | "
            f"counts: {counts['n_pieces']} pieces, "
            f"{counts['n_marginal_observations']} obs"
        )

    composer_pieces_path = out_dir / "composer_pieces.json"
    composer_pieces_path.write_text(
        json.dumps(sample_data, indent=2, ensure_ascii=False)
    )
    print(
        f"\nWrote {composer_pieces_path} "
        f"({composer_pieces_path.stat().st_size / 1024:.1f} KB)"
    )

    composer_counts_path = out_dir / "composer_counts.json"
    composer_counts_path.write_text(
        json.dumps(counts_data, indent=2, ensure_ascii=False)
    )
    print(
        f"Wrote {composer_counts_path} "
        f"({composer_counts_path.stat().st_size / 1024:.1f} KB)"
    )

    composer_names = [c[0] for c in COMPOSERS]
    reference = load_paper_reference(composer_names)
    paper_ref_path = out_dir / "paper_reference.json"
    paper_ref_path.write_text(json.dumps(reference, indent=2, ensure_ascii=False))
    print(
        f"Wrote {paper_ref_path} "
        f"({paper_ref_path.stat().st_size / 1024:.1f} KB)"
    )


if __name__ == "__main__":
    main()
