#!/usr/bin/env python3
"""Small helpers shared by compact-evidence exporters.

The exporters are optional.  Using the release needs only the compact JSON
files and the matching aggregate scripts; ``--export`` is for maintainers who
still have the source repository available.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(root: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip() or None


def source_descriptor(root: Path, path: Path) -> dict:
    """Return content and repository provenance for one source file."""
    path = path.resolve()
    root = root.resolve()
    relative = path.relative_to(root).as_posix()
    tracked = git_output(root, "ls-files", "-s", "--", relative)
    blob = tracked.split()[1] if tracked else None
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "git_blob": blob,
    }


def repo_descriptor(root: Path) -> dict:
    return {
        # A stable logical identifier is portable and does not disclose the
        # maintainer's workstation path.  File paths below are repository-
        # relative and the commit pins their exact content.
        "repository": ("this repository" if root.resolve().name == "PowerCodeBench"
                       else "frozen experimental pipeline"),
        "commit": git_output(root, "rev-parse", "HEAD"),
    }


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        stream.write("\n")
