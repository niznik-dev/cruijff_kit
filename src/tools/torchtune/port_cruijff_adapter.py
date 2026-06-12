"""Restore portability to a cruijff_kit-saved LoRA adapter directory.

By default, cruijff_kit rewrites `adapter_config.json`'s `base_model_name_or_path`
to a local absolute path so the adapter dir is self-loading on offline compute
nodes. That trade-off breaks portability — the rewritten path is meaningless on
a different machine.

This utility restores a portable name into `adapter_config.json`. It prefers the
repo name from `original_repo_id.json` (which torchtune copies in when the base
model was HF-downloaded with that marker present). If that file is missing,
supply the value explicitly with `--repo-id`.

Usage:
    python -m cruijff_kit.tools.torchtune.port_cruijff_adapter <epoch_directory>
    python -m cruijff_kit.tools.torchtune.port_cruijff_adapter <run_directory>  # recurses over epoch_*
    python -m cruijff_kit.tools.torchtune.port_cruijff_adapter <dir> --repo-id meta-llama/Llama-3.2-1B-Instruct

After running, `AutoModelForCausalLM.from_pretrained(<dir>)` will resolve the
base via the HF Hub (or HF cache) as PEFT intends.
"""

import argparse
import json
import sys
from pathlib import Path


def _resolve_repo_id(epoch_directory: Path, repo_id_override: str | None) -> str | None:
    if repo_id_override:
        return repo_id_override
    repo_id_file = epoch_directory / "original_repo_id.json"
    if repo_id_file.exists():
        return json.loads(repo_id_file.read_text())["repo_id"]
    return None


def restore_one(epoch_directory: Path, repo_id_override: str | None = None) -> bool:
    """Restore adapter_config.json's base_model_name_or_path to a portable value.

    Returns True if the file was rewritten, False if skipped.
    """
    adapter_cfg = epoch_directory / "adapter_config.json"
    if not adapter_cfg.exists():
        print(f"  [skip] {epoch_directory}: no adapter_config.json")
        return False

    repo_id = _resolve_repo_id(epoch_directory, repo_id_override)
    if repo_id is None:
        print(
            f"  [skip] {epoch_directory}: no original_repo_id.json — pass --repo-id to override"
        )
        return False

    cfg = json.loads(adapter_cfg.read_text())
    before = cfg.get("base_model_name_or_path")

    if before == repo_id:
        print(f"  [ok]   {epoch_directory}: already {repo_id}")
        return False

    cfg["base_model_name_or_path"] = repo_id
    adapter_cfg.write_text(json.dumps(cfg, indent=2))
    print(f"  [done] {epoch_directory}: {before} -> {repo_id}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "path",
        type=Path,
        help="Adapter epoch dir (with adapter_config.json) or run dir (with epoch_* children).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HF Hub repo name to write (e.g. 'meta-llama/Llama-3.2-1B-Instruct'). "
        "Required when the adapter dir has no original_repo_id.json; overrides it when present.",
    )
    args = parser.parse_args()
    root = args.path

    if not root.exists():
        print(f"ERROR: {root} does not exist", file=sys.stderr)
        return 1

    # If the directory itself has adapter_config.json, treat as a single epoch dir.
    # Otherwise look for epoch_* children.
    if (root / "adapter_config.json").exists():
        targets = [root]
    else:
        targets = sorted(
            p for p in root.iterdir() if p.is_dir() and p.name.startswith("epoch_")
        )

    if not targets:
        print(f"ERROR: no adapter dirs found under {root}", file=sys.stderr)
        return 1

    print(f"Restoring portable HF Hub repo names under {root}:")
    changed = 0
    for t in targets:
        if restore_one(t, args.repo_id):
            changed += 1
    print(f"Rewrote {changed}/{len(targets)} adapter_config.json files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
