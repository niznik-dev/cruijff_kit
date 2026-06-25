#!/usr/bin/env python3
"""privatize_fill — guided fill + pre-submit checks for a privatize-experiment clone.

Run this YOURSELF, in your own terminal (not through Claude). It asks for your
private data folder, fills the clone's `__FILL_REAL_DATA_DIR__` placeholders, then
runs the whole pre-submit check suite so you never paste a snippet into ipython:

  • placeholders cleared           • split sizes & class balance
  • eval cells point at the JSON   • max input tokens vs max_seq_len
  • the JSON exists on disk         • warmup vs total training steps
  • input_formatting unchanged

Privacy by construction: real paths go only to `<DST>/verify.runbook_filled.md`
(which Claude is denied from reading). stdout shows counts / booleans / aggregate
numbers — never a real path — so it stays Claude-blind even if Claude runs it.

Usage:
    python scripts/privatize_fill.py [DST]        # DST = the staged clone directory
"""

import glob
import json
import math
import os
import re
import sys

PLACEHOLDER = "__FILL_REAL_DATA_DIR__"
REPORT_NAME = "verify.runbook_filled.md"  # suffix matches the Read-deny glob


def ask(prompt, default=None):
    suffix = f" [{default}]" if default else ""
    return input(f"{prompt}{suffix}: ").strip() or (default or "")


def text_files(root):
    for dirpath, _, names in os.walk(root):
        for n in names:
            if n.endswith(".md"):  # never touch docs (the runbook, the report)
                continue
            yield os.path.join(dirpath, n)


def read(path):
    try:
        with open(path) as fh:
            return fh.read()
    except (UnicodeDecodeError, OSError):
        return None


def first_glob(dst, pattern):
    hits = sorted(glob.glob(os.path.join(dst, pattern)))
    return hits[0] if hits else None


def scan(path, key, cast=str, default=None):
    """Value of the first `key:` line in a YAML-ish file (quotes stripped)."""
    if not path or not os.path.isfile(path):
        return default
    pat = re.compile(rf"^\s*{re.escape(key)}\s*:\s*(.*?)\s*$")
    for line in open(path):
        m = pat.match(line)
        if m:
            try:
                return cast(m.group(1).strip().strip("'\""))
            except ValueError:
                return default
    return default


def fill(dst, real_dir):
    changed = 0
    for p in text_files(dst):
        content = read(p)
        if content is not None and PLACEHOLDER in content:
            with open(p, "w") as fh:
                fh.write(content.replace(PLACEHOLDER, real_dir))
            changed += 1
    return changed


def scan_clone(dst):
    """Return (sorted data_path values, sorted files still holding a placeholder)."""
    json_paths, leftovers = set(), []
    for p in text_files(dst):
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


def max_input_tokens(records, model_path):
    """Best-effort max prompt length; None if transformers/model unavailable."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        return None
    return max(len(tok(e["input"])["input_ids"]) for e in records)


def main():
    print(
        "🔐 privatize_fill — fill a private-data clone and run the pre-submit checks\n"
    )

    dst = sys.argv[1] if len(sys.argv) > 1 else ask("Staged clone directory (DST)")
    dst = os.path.abspath(os.path.expanduser(dst))
    if not os.path.isfile(os.path.join(dst, "experiment_summary.yaml")):
        sys.exit(f"❌ {dst} is not an experiment clone (no experiment_summary.yaml).")

    _, pending = scan_clone(dst)
    print(f"📂 clone: {dst}")
    print(f"   config files awaiting your data folder: {len(pending)}\n")

    real_dir = ask(
        "Your PRIVATE data folder (holds the .json — the assistant can't see it)"
    )
    real_dir = os.path.abspath(os.path.expanduser(real_dir)).rstrip("/")
    if not real_dir:
        sys.exit("❌ no folder given; nothing filled.")

    changed = fill(dst, real_dir)
    json_paths, leftovers = scan_clone(dst)
    print(f"✏️  filled {changed} config file(s).\n")

    # ---- parameters, read from the clone's own configs ----
    setup = first_glob(dst, "*_ft/setup_finetune.yaml")
    ft = first_glob(dst, "*_ft/finetune.yaml")
    max_seq_len = scan(setup, "max_seq_len", int, 512)
    epochs = scan(setup, "epochs", int, 1)
    batch = scan(setup, "batch_size", int, 1)
    grad_accum = scan(setup, "gradient_accumulation_steps", int, 1)
    fmt = scan(setup, "input_formatting", str, "")
    warmup = scan(ft, "num_warmup_steps", int, 100)
    model_path = scan(setup, "model_checkpoint", str, None)

    checks = []  # (icon, name, stdout-safe summary)
    hints = []  # ready-to-paste fix commands for failed checks
    report = [f"# privatize_fill report — {dst}", ""]

    # 1. placeholders cleared (critical)
    checks.append(
        (
            "✅" if not leftovers else "❌",
            "placeholders cleared",
            f"{len(leftovers)} remaining",
        )
    )
    report += [
        "## placeholders still unfilled",
        *([f"- {x}" for x in leftovers] or ["- none ✓"]),
        "",
    ]

    # 2. eval cells point at a JSON that exists (critical)
    missing = [p for p in json_paths if not os.path.isfile(p)]
    json_ok = bool(json_paths) and not missing
    checks.append(
        (
            "✅" if json_ok else "❌",
            "JSON on disk",
            f"{len(json_paths)} path(s), {len(missing)} missing",
        )
    )
    report += ["## data_path → exists?"]
    report += [
        f"- {p}  ->  {'FOUND' if os.path.isfile(p) else 'MISSING — check the --out name'}"
        for p in json_paths
    ] or ["- (no data_path found)"]
    report += [""]

    # data-derived checks need a loadable JSON
    data = None
    for p in json_paths:
        if os.path.isfile(p):
            try:
                data = json.load(open(p))
                break
            except (OSError, ValueError):
                pass

    if data:
        # 3. splits & balance (info)
        bal = []
        for k, v in data.items():
            pos = sum(e.get("output") == "1" for e in v)
            bal.append((k, len(v), pos, pos / len(v) if v else 0.0))
        checks.append(
            (
                "ℹ️ ",
                "splits & balance",
                " | ".join(f"{k} n={n} rate={r:.3f}" for k, n, _, r in bal),
            )
        )
        report += [
            "## splits & balance",
            *[f"- {k}: n={n} pos={pos} rate={r:.3f}" for k, n, pos, r in bal],
            "",
        ]

        n_train = len(data.get("train", []))

        # 4. max tokens vs max_seq_len (critical when computable)
        recs = [e for v in data.values() for e in v]
        mt = max_input_tokens(recs, model_path) if model_path else None
        if mt is None:
            checks.append(
                ("ℹ️ ", "max tokens", "skipped (transformers/model unavailable)")
            )
        else:
            ok = mt < max_seq_len
            checks.append(
                (
                    "✅" if ok else "❌",
                    "max tokens < max_seq_len",
                    f"{mt} vs {max_seq_len}",
                )
            )
            report += [
                f"## max input tokens: {mt} vs max_seq_len {max_seq_len} — {'OK' if ok else 'BUMP max_seq_len'}",
                "",
            ]
            if not ok:
                # round up to the next multiple of 128 above the real max (headroom)
                new = ((mt // 128) + 1) * 128
                cmd = (
                    f"grep -rl 'max_seq_len: {max_seq_len}' \"{dst}\" | "
                    f"xargs -r sed -i 's#max_seq_len: {max_seq_len}#max_seq_len: {new}#g'"
                )
                hints.append(
                    f"max tokens ({mt}) ≥ max_seq_len ({max_seq_len}) — raise it to {new}, then re-run:\n    {cmd}"
                )
                report += [
                    f"## FIX: bump max_seq_len {max_seq_len} → {new}",
                    f"    {cmd}",
                    "",
                ]

        # 5. warmup vs total steps (advisory)
        total_steps = epochs * math.ceil(n_train / max(batch * grad_accum, 1))
        ok = total_steps > warmup
        checks.append(
            ("✅" if ok else "⚠️ ", "warmup < total steps", f"{warmup} vs {total_steps}")
        )
        report += [
            f"## warmup {warmup} vs total_steps {total_steps} (epochs {epochs} × ceil({n_train}/({batch}×{grad_accum}))) — {'OK' if ok else 'warmup too long'}",
            "",
        ]
    else:
        checks.append(("⏭️ ", "data checks", "skipped — build the JSON first (Step 1)"))

    # 6. input_formatting unchanged (advisory — flat JSON wants empty)
    ok = fmt == ""
    checks.append(("✅" if ok else "⚠️ ", "input_formatting", repr(fmt)))
    report += [
        f"## input_formatting: {fmt!r} — {'empty (OK for flat JSON)' if ok else 'non-empty: appends raw/, may break _setup_data'}",
        "",
    ]

    with open(os.path.join(dst, REPORT_NAME), "w") as fh:
        fh.write("\n".join(report) + "\n")

    # ---- dashboard (stdout-safe: no real paths) ----
    print("──── pre-submit checks ────")
    for icon, name, summary in checks:
        print(f"  {icon} {name:<26} {summary}")
    ready = not leftovers and json_ok and not any(i == "❌" for i, _, _ in checks)
    if hints:
        print("\n──── suggested fixes ────")
        for h in hints:
            print(f"  • {h}")
    print(f"\n{'✅ READY to submit.' if ready else '⛔ NOT ready — fix the ❌ rows.'}")
    print(f"📄 details + real paths → {os.path.join(dst, REPORT_NAME)}")
    print("   (open it yourself; Claude is denied from reading *runbook_filled.md)")


if __name__ == "__main__":
    main()
