#!/usr/bin/env python3
"""Download YOLO pretrained weights in advance (stdlib only, no ultralytics needed).

Files are saved to the target directory (project root by default) so that
`YOLO("yolo11n.pt")` / `YOLO("yolo26n.pt")` and `python train.py` pick them up
without any network access.

Examples:
    python download_models.py                          # yolo11 n, s, m, l, x
    python download_models.py --family yolo26          # yolo26 n, s, m, l, x
    python download_models.py --family both            # both families
    python download_models.py --models n s             # only two sizes
    python download_models.py --models yolo11n yolo26n # mix families explicitly
    python download_models.py --check                  # only verify the URLs
    python download_models.py --force                  # re-download existing files
    python download_models.py --dir weights            # save to ./weights
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SIZES = ("n", "s", "m", "l", "x")
# family -> GitHub release tag that hosts its assets
FAMILIES = {"yolo11": "v8.3.0", "yolo26": "v8.4.0"}
BASE_URL = "https://github.com/ultralytics/assets/releases/download"
CHUNK = 1024 * 256
FULL_NAME = re.compile(r"^(yolo\d+)([nsmlx])$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="download_models.py",
        description="Download YOLO pretrained checkpoints for offline training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--family",
        default="yolo11",
        choices=(*FAMILIES, "both"),
        help="Model family; 'both' = yolo11 and yolo26",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Sizes (n s m l x) or full names (yolo11n, yolo26s); default = all sizes of --family",
    )
    parser.add_argument(
        "--dir",
        default=str(PROJECT_ROOT),
        help="Directory to save the .pt files into",
    )
    parser.add_argument("--check", action="store_true", help="Only check availability, do not download")
    parser.add_argument("--force", action="store_true", help="Overwrite already downloaded files")
    return parser.parse_args(argv)


def resolve_models(family: str, models: list[str] | None) -> list[str]:
    """Return a de-duplicated list of file stems such as 'yolo26n'."""
    families = list(FAMILIES) if family == "both" else [family]

    if models is None:
        return [f"{fam}{size}" for fam in families for size in SIZES]

    stems: list[str] = []
    for token in models:
        token = token.lower().removesuffix(".pt")
        if token in SIZES:
            stems.extend(f"{fam}{token}" for fam in families)
            continue
        match = FULL_NAME.match(token)
        if match and match.group(1) in FAMILIES:
            stems.append(token)
            continue
        raise SystemExit(
            f"[error] unknown model '{token}'; use one of {list(SIZES)} "
            f"or a full name like {', '.join(f + 'n' for f in FAMILIES)}"
        )
    return list(dict.fromkeys(stems))


def url_for(stem: str) -> str:
    family = FULL_NAME.match(stem).group(1)
    return f"{BASE_URL}/{FAMILIES[family]}/{stem}.pt"


def remote_size(url: str) -> int | None:
    """Return file size in bytes from a HEAD request, or None on failure."""
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        print(f"[error] {url} -> {exc}")
        return None


def human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


def download(url: str, dest: Path, expected: int | None) -> bool:
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as out:
            total = expected or int(response.headers.get("Content-Length") or 0)
            done = 0
            last_report = -1
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    percent = done * 100 // total
                    if percent != last_report and percent % 10 == 0:
                        last_report = percent
                        print(f"    {percent:>3}%  {human(done)} / {human(total)}", flush=True)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        print(f"[error] download failed: {exc}")
        return False

    if expected is not None and tmp.stat().st_size != expected:
        print(f"[error] size mismatch: got {tmp.stat().st_size} bytes, expected {expected}")
        tmp.unlink(missing_ok=True)
        return False

    tmp.replace(dest)
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stems = resolve_models(args.family, args.models)
    target_dir = Path(args.dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] families: {args.family}")
    print(f"[info] target  : {target_dir}")
    print(f"[info] models  : {', '.join(stems)}")
    print("")

    failures = 0
    for stem in stems:
        name = f"{stem}.pt"
        url = url_for(stem)
        dest = target_dir / name

        if dest.is_file() and not args.force:
            print(f"[skip] {name} already exists ({human(dest.stat().st_size)})")
            continue

        print(f"[get ] {name}")
        expected = remote_size(url)
        if expected is None:
            failures += 1
            print(f"       check your internet connection or proxy, URL: {url}")
            continue
        print(f"       size {human(expected)}")

        if args.check:
            print("       available")
            continue

        if not download(url, dest, expected):
            failures += 1
            continue
        print(f"[ok  ] saved {dest}")

    print("")
    if failures:
        print(f"[done] failed: {failures} of {len(stems)}")
        return 1
    print("[done] all requested models are available locally")
    return 0


if __name__ == "__main__":
    sys.exit(main())
