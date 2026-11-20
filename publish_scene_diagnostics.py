"""Publish a bounded snapshot of the latest Scene Engine diagnostics to Git.

Run after scene_engine.py. Only the allowlisted files are staged.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE = Path("output/scene-engine")
TARGET = Path("diagnostics/latest")
FILES = (
    "ink-centerlines.png",
    "color-regions.png",
    "ink-mask.png",
    "layered-preview.png",
    "layer-validation.json",
    "stroke-reconstruction.png",
    "stroke-difference.png",
    "stroke-validation.json",
    "simplified-paths-preview.png",
    "simplification-validation.json",
    "validation.json",
)


def git(*args):
    return subprocess.run(["git", *args], check=True, text=True,
                          capture_output=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description="Publish latest diagnostic snapshot to GitHub")
    parser.add_argument("--push", action="store_true",
                        help="Push the resulting commit to origin/main")
    args = parser.parse_args()
    if not SOURCE.is_dir():
        parser.error("Missing output/scene-engine; run scene_engine.py first")
    if git("branch", "--show-current") != "main":
        parser.error("Diagnostics publishing is restricted to branch main")
    if git("status", "--porcelain", "--", str(TARGET)):
        parser.error("diagnostics/latest has uncommitted changes; review them first")
    available = [name for name in FILES if (SOURCE / name).is_file()]
    if not available:
        parser.error("No diagnostic files found")
    TARGET.mkdir(parents=True, exist_ok=True)
    for old in TARGET.iterdir():
        if old.is_file() and old.name not in available and old.name in FILES:
            old.unlink()
    for name in available:
        shutil.copy2(SOURCE / name, TARGET / name)
    git("add", "--", str(TARGET))
    changed = subprocess.run(["git", "diff", "--cached", "--quiet", "--", str(TARGET)])
    if changed.returncode == 0:
        print("Diagnostic snapshot unchanged; nothing to commit")
        return
    if changed.returncode != 1:
        raise RuntimeError("Unable to inspect staged diagnostic changes")
    print(git("commit", "-m", "Update latest scene diagnostics", "--", str(TARGET)))
    if args.push:
        print(git("push", "origin", "main"))
    else:
        print("Committed locally. Use --push to publish to GitHub.")


if __name__ == "__main__":
    main()
