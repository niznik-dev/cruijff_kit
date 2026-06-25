#!/usr/bin/env python3
"""privatize_fill — guided fill + verify for a privatize-experiment clone.

Run this YOURSELF, in your own terminal (not through Claude). It asks for your
private data folder, fills the clone's `__FILL_REAL_DATA_DIR__` placeholders, and
verifies the result.

Privacy by construction: the real paths are written only to
`<DST>/verify.runbook_filled.md` — which Claude is denied from reading (the
`Read(**/*runbook_filled.md)` rule) — and to your terminal. Everything printed to
stdout is aggregate/boolean (counts, pass/fail), never a real path, so the process
stays Claude-blind even on the off chance Claude runs it.

Usage:
    python scripts/privatize_fill.py [DST]      # DST = the staged clone directory
"""

import json
import os
import sys

PLACEHOLDER = "__FILL_REAL_DATA_DIR__"
REPORT_NAME = "verify.runbook_filled.md"  # suffix matches the Read-deny glob


def ask(prompt, default=None):
    suffix = f" [{default}]" if default else ""
    return input(f"{prompt}{suffix}: ").strip() or (default or "")


def text_files(root):
    for dirpath, _, names in os.walk(root):
        for n in names:
            yield os.path.join(dirpath, n)


def read(path):
    try:
        with open(path) as fh:
            return fh.read()
    except (UnicodeDecodeError, OSError):
        return None


def fill(dst, real_dir):
    """Replace the data-dir placeholder across every text file in the clone."""
    changed = 0
    for p in text_files(dst):
        content = read(p)
        if content is not None and PLACEHOLDER in content:
            with open(p, "w") as fh:
                fh.write(content.replace(PLACEHOLDER, real_dir))
            changed += 1
    return changed


def verify(dst):
    """Collect the eval data_path values and any files still holding a placeholder."""
    json_paths, leftovers = set(), []
    for p in text_files(dst):
        if p.endswith(REPORT_NAME):
            continue
        content = read(p)
        if content is None:
            continue
        if PLACEHOLDER in content:
            leftovers.append(os.path.relpath(p, dst))
        if os.path.basename(p) == "eval.yaml":
            for line in content.splitlines():
                if line.strip().startswith("data_path:"):
                    json_paths.add(line.split(":", 1)[1].strip())
    return sorted(json_paths), sorted(leftovers)


def splits(json_path):
    """Aggregate-only split sizes + positive rate; never touches record content."""
    try:
        d = json.load(open(json_path))
    except (OSError, ValueError):
        return None
    out = []
    for k, v in d.items():
        pos = sum(e.get("output") == "1" for e in v)
        out.append((k, len(v), pos, pos / len(v) if v else 0.0))
    return out


def main():
    print("🔐 privatize_fill — fill a private-data clone and verify it\n")

    dst = sys.argv[1] if len(sys.argv) > 1 else ask("Staged clone directory (DST)")
    dst = os.path.abspath(os.path.expanduser(dst))
    if not os.path.isfile(os.path.join(dst, "experiment_summary.yaml")):
        sys.exit(
            f"❌ {dst} doesn't look like an experiment clone (no experiment_summary.yaml)."
        )

    _, pending = verify(dst)
    print(f"📂 clone: {dst}")
    print(f"   placeholders awaiting a real path: {len(pending)} file(s)\n")

    real_dir = ask(
        "Your PRIVATE data folder (holds the .json — the assistant can't see it)"
    )
    real_dir = os.path.abspath(os.path.expanduser(real_dir)).rstrip("/")
    if not real_dir:
        sys.exit("❌ no folder given; nothing filled.")

    changed = fill(dst, real_dir)
    json_paths, leftovers = verify(dst)
    print(f"\n✏️  filled {changed} file(s).")

    # Build the Claude-denied report (full real paths live here, not on stdout).
    lines = [f"# privatize_fill report — {dst}", "", "## data_path values"]
    lines += [f"- {p}" for p in json_paths] or ["- (none found)"]
    missing = [p for p in json_paths if not os.path.isfile(p)]
    lines += ["", "## JSON present on disk?"]
    lines += [
        f"- {p}  ->  {'FOUND' if os.path.isfile(p) else 'MISSING — build it first'}"
        for p in json_paths
    ]
    if leftovers:
        lines += ["", "## ⚠️ files still holding __FILL_REAL_DATA_DIR__"] + [
            f"- {x}" for x in leftovers
        ]
    for jp in json_paths:
        s = splits(jp)
        if s:
            lines += ["", f"## splits & balance — {jp}"]
            lines += [
                f"- {k}: n={n} pos={pos} rate={rate:.3f}" for (k, n, pos, rate) in s
            ]
    report = os.path.join(dst, REPORT_NAME)
    with open(report, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # stdout summary: counts + booleans only, never a real path.
    print("\n──── verification ────")
    print(
        f"  remaining placeholders : {len(leftovers)}   {'✅' if not leftovers else '❌ re-run / check'}"
    )
    print(f"  eval cells point at JSON: {len(json_paths)} unique path(s)")
    print(
        f"  JSON present on disk    : {'✅ yes' if json_paths and not missing else '❌ missing — run the build (Step 1) first'}"
    )
    ok = (not leftovers) and json_paths and not missing
    print(f"\n{'✅ READY to submit.' if ok else '⚠️  NOT ready — see the report.'}")
    print(f"📄 details (real paths, splits) → {report}")
    print("   (open it yourself; Claude is denied from reading *runbook_filled.md)")


if __name__ == "__main__":
    main()
