"""Download a model from ModelScope and verify it against a Hugging Face revision.

ModelScope is fast from the server (~20-80 MB/s) and Hugging Face is slow (~1.5 MB/s), but
the experiments name a Hugging Face revision. So weights come from ModelScope and every file
is checked against that revision's hashes: sha256 for LFS files, the git blob sha1 otherwise.
Files land in $ROOT/models/<name>.

On the server:
    uv run --no-project --with modelscope==1.40.2 python $REPO/scripts/fetch_model.py qwen3-4b
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

from modelscope import snapshot_download

# name -> (Hugging Face id, revision); the ModelScope id is the same string
MODELS = {
    "qwen3-4b": ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554"),
    "olmo3-7b": ("allenai/Olmo-3-7B-Instruct", "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"),
}


def file_hash(path: Path, lfs: bool) -> str:
    h = hashlib.sha256() if lfs else hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=MODELS)
    model, revision = MODELS[parser.parse_args().name]

    target = Path(os.environ["ROOT"]) / "models" / model.split("/")[1]
    snapshot_download(model, local_dir=str(target))

    url = f"https://huggingface.co/api/models/{model}/tree/{revision}"
    # .gitattributes holds each hub's own LFS rules, not model content
    expected = [e for e in json.load(urllib.request.urlopen(url)) if e["path"] != ".gitattributes"]
    bad = []
    for entry in expected:
        path = target / entry["path"]
        lfs = entry.get("lfs")
        want = lfs["oid"] if lfs else entry["oid"]
        got = file_hash(path, bool(lfs)) if path.exists() else "missing"
        status = "ok" if got == want else "MISMATCH"
        print(f"{status:8s} {entry['path']}")
        if status != "ok":
            bad.append(entry["path"])
    if bad:
        sys.exit(f"{len(bad)} files differ from {model}@{revision}: {bad}")
    print(f"all {len(expected)} files match {model}@{revision}")


if __name__ == "__main__":
    main()
