"""Download a pinned official safetensors snapshot without duplicate HF caches.

Concurrent HTTP ranges write disjoint regions of one .part; completed files are
verified against Hub LFS SHA256 (or Git blob SHA1 for small files) before rename.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.parse import urlparse

import requests

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="checkpoints/Qwen3-VL-2B-Instruct-bf16")
    parser.add_argument("--proxy", default="http://127.0.0.1:7897")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--direct-cdn", action="store_true", help="Resolve through proxy, stream official HF CDN directly")
    parser.add_argument("--reserve-gib", type=float, default=8.)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    project = Path(__file__).resolve().parents[1]
    allowed = (project / "checkpoints").resolve()
    if root == allowed or allowed not in root.parents:
        raise ValueError("Model output must be a subdirectory of this project's checkpoints directory.")
    if args.workers < 1 or args.workers > 12 or args.reserve_gib < 0:
        raise ValueError("Require 1..12 workers and a nonnegative disk reserve.")
    root.mkdir(parents=True, exist_ok=True)

    def session():
        value = requests.Session()
        value.trust_env = False
        if args.proxy:
            value.proxies = {"https": args.proxy}
        return value

    def range_response(url, start, end):
        headers = {"Range": "bytes=%d-%d" % (start, end)}
        if not args.direct_cdn:
            return session().get(url, headers=headers, stream=True, timeout=(30, 60))
        redirect = session().get(url, headers=headers, allow_redirects=False, timeout=30)
        redirect.raise_for_status()
        destination = redirect.headers.get("Location")
        host = urlparse(destination or "").hostname or ""
        if not destination or urlparse(destination).scheme != "https" or not host.endswith((".hf.co", ".huggingface.co")):
            raise ValueError("Expected an official Hugging Face HTTPS CDN redirect")
        direct = requests.Session()
        direct.trust_env = False
        return direct.get(destination, headers=headers, stream=True, timeout=(20, 60))

    metadata = session().get("https://huggingface.co/api/models/" + MODEL + "/revision/" + REVISION + "?blobs=true", timeout=40)
    metadata.raise_for_status()
    info = metadata.json()
    if info["sha"] != REVISION:
        raise ValueError("Unexpected model revision")
    files = [f for f in info["siblings"] if f["rfilename"].endswith((".safetensors", ".json", ".txt", ".jinja")) and "/" not in f["rfilename"]]
    needed = sum(f["size"] for f in files if not (root / f["rfilename"]).exists())
    # .part may already occupy space; this conservative check can overestimate on resume.
    occupied = sum(p.stat().st_size for p in root.glob("*.part"))
    if shutil.disk_usage(root).free < max(0, needed - occupied) + args.reserve_gib * 1024**3:
        raise ValueError("Insufficient space for model plus requested reserve.")
    records = []
    for entry in files:
        name, total = entry["rfilename"], entry["size"]
        path = root / name
        url = "https://huggingface.co/" + MODEL + "/resolve/" + REVISION + "/" + name
        expected = (entry.get("lfs") or {}).get("sha256")
        if path.exists():
            if path.stat().st_size != total or (expected and digest(path) != expected):
                raise ValueError("Existing file has wrong size/hash: " + name)
        else:
            partial = path.with_name(name + ".part")
            if total < 16 * 1024**2:
                response = session().get(url, timeout=60)
                response.raise_for_status()
                partial.write_bytes(response.content)
            else:
                if not partial.exists():
                    with partial.open("wb") as stream:
                        stream.truncate(total)
                if partial.stat().st_size != total:
                    raise ValueError("Partial model size differs; do not overwrite an unknown partial file.")
                block = 128 * 1024**2
                ranges = [(offset, min(total, offset + block) - 1) for offset in range(0, total, block)]

                def fetch(bounds):
                    start, end = bounds
                    marker = root / (name + ".range_%d.done" % start)
                    if marker.exists() and marker.read_text() == str(end):
                        return
                    for attempt in range(4):
                        try:
                            if shutil.disk_usage(root).free < args.reserve_gib * 1024**3:
                                raise ValueError("Disk reserve reached during download")
                            with range_response(url, start, end) as response:
                                response.raise_for_status()
                                if response.status_code != 206 or response.headers.get("Content-Range") != "bytes %d-%d/%d" % (start, end, total):
                                    raise ValueError("Server did not honor the exact requested range")
                                count = 0
                                with partial.open("r+b") as stream:
                                    stream.seek(start)
                                    for chunk in response.iter_content(1024**2):
                                        count += len(chunk)
                                        if count > end - start + 1:
                                            raise ValueError("Range body exceeds declared size")
                                        stream.write(chunk)
                                if count != end - start + 1:
                                    raise ValueError("Truncated range")
                            marker.write_text(str(end))
                            print("%s: verified transfer range %d-%d" % (name, start, end), flush=True)
                            return
                        except (requests.RequestException, ValueError) as error:
                            if attempt == 3 or "Disk reserve" in str(error):
                                raise
                            time.sleep(1)
                    raise RuntimeError("Range exhausted retries")

                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    list(pool.map(fetch, ranges))
            if partial.stat().st_size != total:
                raise ValueError("Downloaded size mismatch: " + name)
            actual = digest(partial)
            if expected and actual != expected:
                raise ValueError("Downloaded SHA256 mismatch: " + name)
            if not expected and entry.get("blobId"):
                blob = partial.read_bytes()
                git_hash = hashlib.sha1(("blob %d\0" % len(blob)).encode() + blob).hexdigest()
                if git_hash != entry["blobId"]:
                    raise ValueError("Downloaded Git blob mismatch: " + name)
            partial.replace(path)
        records.append({"file": name, "bytes": total, "sha256": digest(path), "source_url": url})
        print("Verified " + name, flush=True)
    (root / "source_provenance.json").write_text(json.dumps({"model": MODEL, "revision": REVISION, "files": records}, indent=2) + "\n", encoding="utf-8")
    print("Training model ready: " + str(root), flush=True)


if __name__ == "__main__":
    main()
