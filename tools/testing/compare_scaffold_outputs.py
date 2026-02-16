#!/usr/bin/env python3
"""Compare two scaffold-experiment outputs for reproducibility.

Walks both directory trees, normalizes path-dependent content, and reports
whether files are identical (byte-level) or semantically equivalent (YAML).

Usage:
    python tools/testing/compare_scaffold_outputs.py DIR_A DIR_B
"""

import argparse
import os
import re
import sys

import yaml


# ---------------------------------------------------------------------------
# File categorisation
# ---------------------------------------------------------------------------

SKIP_EXTENSIONS = {".log"}

SCRIPT_GENERATED = {"finetune.yaml", "finetune.slurm"}

# Agent-written files are identified by name patterns
AGENT_WRITTEN_NAMES = {"setup_finetune.yaml", "eval_config.yaml"}


def categorise(rel_path: str) -> str:
    """Return 'skip', 'script', or 'agent' for a relative file path."""
    _, ext = os.path.splitext(rel_path)
    if ext in SKIP_EXTENSIONS:
        return "skip"
    basename = os.path.basename(rel_path)
    if basename in SCRIPT_GENERATED:
        return "script"
    # Everything else (eval slurm scripts, setup_finetune.yaml, eval_config.yaml)
    return "agent"


# ---------------------------------------------------------------------------
# Path normalisation
# ---------------------------------------------------------------------------

def build_normaliser(dir_a: str, dir_b: str):
    """Return a function that replaces directory-specific strings with _X."""
    # Extract the suffix (_A or _B) from the directory names and build
    # replacement pairs.  We normalise both the full directory paths and
    # the experiment names that appear inside config files.
    basename_a = os.path.basename(dir_a.rstrip("/"))
    basename_b = os.path.basename(dir_b.rstrip("/"))

    # Find the differing suffix (everything after the common prefix)
    common = os.path.commonprefix([basename_a, basename_b])
    suffix_a = basename_a[len(common):]
    suffix_b = basename_b[len(common):]

    if not suffix_a or not suffix_b:
        sys.exit(
            f"Cannot determine differing suffixes.\n"
            f"  Dir A basename: {basename_a}\n"
            f"  Dir B basename: {basename_b}\n"
            f"  Common prefix:  {common}"
        )

    # We need to replace long strings first to avoid partial replacements.
    # Use the full directory paths, then basenames, then bare suffixes.
    parent_a = os.path.dirname(dir_a.rstrip("/"))
    parent_b = os.path.dirname(dir_b.rstrip("/"))
    full_a = os.path.join(parent_a, basename_a)
    full_b = os.path.join(parent_b, basename_b)
    # common prefix already includes the separator (e.g. "..._2026-02-16_")
    # so we just append "X" to avoid double underscores
    normalised_full = os.path.join(parent_a, common + "X")

    replacements = [
        # Full paths (experiments dir and outputs dir)
        (full_a, normalised_full),
        (full_b, normalised_full),
        # Also handle the output directory variant (ck-outputs instead of ck-experiments)
        (full_a.replace("ck-experiments", "ck-outputs"), normalised_full.replace("ck-experiments", "ck-outputs")),
        (full_b.replace("ck-experiments", "ck-outputs"), normalised_full.replace("ck-experiments", "ck-outputs")),
        # Basenames (e.g. in experiment name fields)
        (basename_a, common + "X"),
        (basename_b, common + "X"),
    ]

    def normalise(text: str) -> str:
        for old, new in replacements:
            text = text.replace(old, new)
        return text

    return normalise


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def compare_bytes(text_a: str, text_b: str) -> bool:
    """Return True if normalised texts are byte-identical."""
    return text_a == text_b


def compare_yaml_semantic(text_a: str, text_b: str) -> tuple[bool, str]:
    """Parse YAML and compare data structures. Returns (match, detail)."""
    try:
        data_a = yaml.safe_load(text_a)
        data_b = yaml.safe_load(text_b)
    except yaml.YAMLError as e:
        return False, f"YAML parse error: {e}"

    if data_a == data_b:
        return True, "semantically identical"
    else:
        return False, "YAML structures differ"


def compare_slurm_directives(text_a: str, text_b: str) -> tuple[bool, str]:
    """Compare SBATCH directives and command lines in SLURM scripts."""
    def extract_significant(text):
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            # Keep SBATCH directives and non-empty, non-comment lines
            if stripped.startswith("#SBATCH") or (stripped and not stripped.startswith("#")):
                lines.append(stripped)
        return lines

    lines_a = extract_significant(text_a)
    lines_b = extract_significant(text_b)

    if lines_a == lines_b:
        return True, "directives and commands match"
    else:
        # Find first difference for reporting
        for i, (la, lb) in enumerate(zip(lines_a, lines_b)):
            if la != lb:
                return False, f"line {i+1} differs:\n  A: {la}\n  B: {lb}"
        if len(lines_a) != len(lines_b):
            return False, f"different line counts: {len(lines_a)} vs {len(lines_b)}"
        return False, "unknown difference"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_files(directory: str) -> dict[str, str]:
    """Return {relative_path: absolute_path} for all files under directory."""
    files = {}
    for root, _dirs, filenames in os.walk(directory):
        for fn in filenames:
            abs_path = os.path.join(root, fn)
            rel_path = os.path.relpath(abs_path, directory)
            files[rel_path] = abs_path
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dir_a", help="First scaffold output directory")
    parser.add_argument("dir_b", help="Second scaffold output directory")
    args = parser.parse_args()

    dir_a = os.path.abspath(args.dir_a)
    dir_b = os.path.abspath(args.dir_b)

    for d in (dir_a, dir_b):
        if not os.path.isdir(d):
            sys.exit(f"Not a directory: {d}")

    normalise = build_normaliser(dir_a, dir_b)
    files_a = collect_files(dir_a)
    files_b = collect_files(dir_b)

    all_paths = sorted(set(files_a.keys()) | set(files_b.keys()))

    # Counters
    results = {"pass": 0, "fail": 0, "skip": 0}
    failures = []

    print(f"Comparing scaffold outputs")
    print(f"  A: {dir_a}")
    print(f"  B: {dir_b}")
    print(f"  Files found: {len(files_a)} (A), {len(files_b)} (B)")
    print()

    # Check for missing files
    only_a = set(files_a.keys()) - set(files_b.keys())
    only_b = set(files_b.keys()) - set(files_a.keys())
    if only_a:
        print(f"FAIL: Files only in A: {sorted(only_a)}")
        results["fail"] += len(only_a)
        for p in sorted(only_a):
            failures.append((p, "only in A"))
    if only_b:
        print(f"FAIL: Files only in B: {sorted(only_b)}")
        results["fail"] += len(only_b)
        for p in sorted(only_b):
            failures.append((p, "only in B"))

    # Compare shared files
    shared = sorted(set(files_a.keys()) & set(files_b.keys()))
    for rel_path in shared:
        category = categorise(rel_path)

        if category == "skip":
            print(f"  SKIP  {rel_path}")
            results["skip"] += 1
            continue

        # Read and normalise
        with open(files_a[rel_path]) as f:
            content_a = normalise(f.read())
        with open(files_b[rel_path]) as f:
            content_b = normalise(f.read())

        byte_match = compare_bytes(content_a, content_b)

        if byte_match:
            print(f"  PASS  {rel_path}  [{category}] (byte-identical)")
            results["pass"] += 1
            continue

        # Not byte-identical — try semantic comparison for known types
        basename = os.path.basename(rel_path)
        _, ext = os.path.splitext(basename)

        if ext == ".yaml":
            sem_match, detail = compare_yaml_semantic(content_a, content_b)
            if sem_match:
                print(f"  PASS  {rel_path}  [{category}] (semantic match, byte differs)")
                results["pass"] += 1
            else:
                print(f"  FAIL  {rel_path}  [{category}] ({detail})")
                results["fail"] += 1
                failures.append((rel_path, detail))
        elif ext == ".slurm":
            dir_match, detail = compare_slurm_directives(content_a, content_b)
            if dir_match:
                print(f"  PASS  {rel_path}  [{category}] (directives match, whitespace differs)")
                results["pass"] += 1
            else:
                print(f"  FAIL  {rel_path}  [{category}] ({detail})")
                results["fail"] += 1
                failures.append((rel_path, detail))
        else:
            print(f"  FAIL  {rel_path}  [{category}] (byte mismatch, no semantic check)")
            results["fail"] += 1
            failures.append((rel_path, "byte mismatch"))

    # Summary
    print()
    print("=" * 60)
    print(f"SUMMARY: {results['pass']} passed, {results['fail']} failed, {results['skip']} skipped")
    print("=" * 60)

    if failures:
        print()
        print("Failures:")
        for path, reason in failures:
            print(f"  - {path}: {reason}")
        print()
        print("To inspect differences, run:")
        print(f"  diff <(sed 's/_A/_X/g' DIR_A/FILE) <(sed 's/_B/_X/g' DIR_B/FILE)")

    sys.exit(0 if results["fail"] == 0 else 1)


if __name__ == "__main__":
    main()
