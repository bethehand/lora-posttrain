"""Download Qwen3-4B-Instruct-2507 from ModelScope and verify it against Hugging Face.

ModelScope is fast from the server (~20 MB/s) and Hugging Face is slow (~1.5 MB/s), but the
experiments name a Hugging Face revision. So weights come from ModelScope and every file is
checked against that revision's hashes: sha256 for LFS files, the git blob sha1 otherwise.

On the server:
    uv run --no-project --with modelscope==1.40.2 python $REPO/scripts/fetch_model.py
"""

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

from modelscope import snapshot_download

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
HF_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


def file_hash(path: Path, lfs: bool) -> str:
    h = hashlib.sha256() if lfs else hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    target = Path(os.environ["ROOT"]) / "models" / MODEL.split("/")[1]
    snapshot_download(MODEL, local_dir=str(target))

    url = f"https://huggingface.co/api/models/{MODEL}/tree/{HF_REVISION}"
    expected = json.load(urllib.request.urlopen(url))
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
        sys.exit(f"{len(bad)} files differ from {MODEL}@{HF_REVISION}: {bad}")
    print(f"all {len(expected)} files match {MODEL}@{HF_REVISION}")


if __name__ == "__main__":
    main()
