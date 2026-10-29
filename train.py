#!/usr/bin/env python3
"""Train a YOLO detector (YOLO11 / YOLO26) on a dataset configured in config.json.

Cross-platform (Windows / Linux). All console output is ASCII-only.
The dataset path comes from config.json, results go to
`result/<dataset directory name>/<model name>/runs/<run name>`.

Examples:
    python train.py
    python train.py --model yolo26s.pt --epochs 200 --batch 16 --device 0
    python train.py --epochs 5 --imgsz 320 --batch 4 --name smoke_test   # quick check
    python train.py --resume
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_FILE = PROJECT_ROOT / "config.json"
RESULT_ROOT = PROJECT_ROOT / "result"
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CONFIG_EXAMPLE = '{"data": "datasets/<dataset name>/data.yaml"}'


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Train YOLO11 / YOLO26 on the dataset configured in config.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Checkpoint or config: yolo11n/s/m/l/x.pt, yolo26n/s/m/l/x.pt, or a path",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to data.yaml; overrides the 'data' value from config.json",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (-1 = auto)")
    parser.add_argument(
        "--device",
        default=None,
        help="CUDA device id(s) such as 0 or 0,1; use 'cpu' for CPU; empty = auto",
    )
    parser.add_argument("--workers", type=int, default=8, help="Dataloader worker processes")
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (epochs without mAP improvement)",
    )
    parser.add_argument(
        "--cache",
        choices=("none", "ram", "disk"),
        default="none",
        help="Cache images for faster epochs (needs RAM/disk)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--project",
        default=None,
        help="Directory for runs; empty = result/<dataset name>/<model name>/runs",
    )
    parser.add_argument("--name", default="train", help="Run name")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse an existing run directory")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume from a checkpoint (auto = newest weights/last.pt of this dataset)",
    )
    return parser.parse_args(argv)


def load_config(path: Path = CONFIG_FILE) -> dict | None:
    """Read config.json. Returns None (and prints a hint) if it is unusable."""
    if not path.is_file():
        print(f"[error] config file not found: {path}")
        print(f"        create it with: {CONFIG_EXAMPLE}")
        return None

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        print(f"[error] config file is empty: {path}")
        print(f"        fill it with: {CONFIG_EXAMPLE}")
        return None

    try:
        cfg = json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"[error] invalid JSON in {path}: {exc}")
        print(f"        expected something like: {CONFIG_EXAMPLE}")
        return None

    if not isinstance(cfg, dict):
        print(f"[error] {path} must contain a JSON object, e.g. {CONFIG_EXAMPLE}")
        return None
    return cfg


def resolve_path(value: str) -> Path:
    """Resolve a config path: relative paths are taken from the project root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def find_last_run(search_dir: Path) -> Path | None:
    """Newest resumable checkpoint under search_dir, used by --resume auto.

    Layouts searched: <model>/runs/<run>/weights/last.pt and <run>/weights/last.pt.
    Finished runs are skipped when an interrupted one is available.
    """
    candidates = list(search_dir.glob("*/runs/*/weights/last.pt"))
    if not candidates:
        candidates = list(search_dir.glob("*/weights/last.pt"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for checkpoint in candidates:
        if is_resumable(checkpoint):
            return checkpoint
    return candidates[0]


def tag_from_checkpoint(checkpoint: Path) -> str | None:
    """Model directory name from a run layout: .../<model>/runs/<run>/weights/x.pt -> <model>."""
    parts = checkpoint.parts
    if (
        len(parts) >= 5
        and parts[-2] == "weights"
        and parts[-1].endswith(".pt")
        and parts[-4] == "runs"
    ):
        return parts[-5]
    return None


def model_tag(model: str) -> str:
    """Directory-safe model name for result/<dataset>/<model>/runs.

    'yolo26s.pt' -> 'yolo26s', 'weights/x.pt' -> 'x',
    'result/D/yolo26l/runs/train/weights/best.pt' -> 'yolo26l' (fine-tuning keeps the tag).
    """
    from_checkpoint = tag_from_checkpoint(Path(model))
    if from_checkpoint:
        return from_checkpoint
    stem = Path(model).stem if model.endswith((".pt", ".yaml", ".yml")) else model
    return stem.replace(" ", "_") or "model"


def is_resumable(checkpoint: Path) -> bool:
    """True if the .pt still holds optimizer/epoch state (i.e. training was interrupted).

    Ultralytics strips these keys from last.pt/best.pt once a run finishes, and
    `train(resume=True)` on such a file silently starts a new run instead.
    """
    import torch

    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:  # corrupt or unreadable file
        print(f"[warn] cannot read checkpoint {checkpoint}: {exc}")
        return False
    if not isinstance(ckpt, dict):
        return False
    epoch = ckpt.get("epoch")
    optimizer = ckpt.get("optimizer")
    # a finished run is saved with epoch=-1 and optimizer=None
    return isinstance(epoch, int) and epoch >= 0 and bool(optimizer)


def resolve_split(yaml_dir: Path, value: str) -> Path:
    """Resolve a split path the same way Ultralytics does.

    Roboflow exports contain paths like '../train/images' while data.yaml sits in
    the dataset root, so the leading '../' is dropped when the direct path is missing.
    """
    candidate = (yaml_dir / value).resolve()
    if not candidate.exists() and value.startswith("../"):
        candidate = (yaml_dir / value[3:]).resolve()
    return candidate


def count_files(directory: Path, suffixes: set[str]) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for f in directory.iterdir() if f.is_file() and f.suffix.lower() in suffixes)


def inspect_dataset(data_path: Path) -> bool:
    """Print a short dataset summary. Returns False if the layout is unusable."""
    try:
        import yaml
    except ImportError:
        print("[warn] PyYAML is not installed, skipping the dataset check")
        return True

    with data_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    print(f"[info] data.yaml : {data_path}")
    print(f"[info] classes   : {len(names)} -> {list(names)}")

    ok = True
    for split in SPLITS:
        value = cfg.get(split)
        if not value:
            continue
        images_dir = resolve_split(data_path.parent, str(value))
        labels_dir = images_dir.parent / "labels"
        n_images = count_files(images_dir, IMAGE_SUFFIXES)
        n_labels = count_files(labels_dir, {".txt"})
        print(f"[info] {split:<7}: {n_images:>6} images / {n_labels:>6} labels  ({images_dir})")
        if split in ("train", "val"):
            if n_images == 0:
                print(f"[warn] split '{split}' has no images")
                ok = False
            elif n_images != n_labels:
                print(f"[warn] split '{split}': image/label count mismatch")
    return ok


def pick_device(requested: str | None, torch_module) -> str:
    if requested not in (None, ""):
        return requested
    return "0" if torch_module.cuda.is_available() else "cpu"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cfg = load_config()
    if cfg is None:
        return 2

    raw_data = args.data if args.data else cfg.get("data")
    if not raw_data:
        print(f"[error] no dataset path: config.json has no 'data' key")
        print(f"        fill {CONFIG_FILE} with: {CONFIG_EXAMPLE}")
        return 2

    data_path = resolve_path(str(raw_data))
    if not data_path.is_file():
        print(f"[error] data.yaml not found: {data_path}")
        print(f"        check the 'data' value in {CONFIG_FILE}")
        return 2

    dataset_name = data_path.parent.name
    dataset_root = RESULT_ROOT / dataset_name

    last: Path | None = None
    if args.resume:
        if args.resume != "auto":
            last = resolve_path(args.resume)
            if not last.is_file():
                print(f"[error] checkpoint to resume not found: {last}")
                return 2
        else:
            last = find_last_run(dataset_root)
            if last is None:
                print(f"[error] nothing to resume: no weights/last.pt under {dataset_root}")
                return 2
        # the model directory is taken from the checkpoint layout:
        # result/<dataset>/<model>/runs/<run>/weights/last.pt
        tag = tag_from_checkpoint(last) or model_tag(args.model)
    else:
        tag = model_tag(args.model)

    project_dir = resolve_path(args.project) if args.project else dataset_root / tag / "runs"

    print(f"[info] config    : {CONFIG_FILE}")
    print(f"[info] dataset   : {dataset_name}")
    print(f"[info] model dir : {tag}")
    print(f"[info] results   : {project_dir / args.name}")

    if not inspect_dataset(data_path):
        print("[error] dataset layout looks broken, fix the paths in data.yaml")
        return 2

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"[error] missing dependency: {exc.name}")
        print("        install the environment first, see SETUP.md")
        return 1

    device = pick_device(args.device, torch)
    cuda_ok = torch.cuda.is_available()
    print(f"[info] torch     : {torch.__version__} (cuda available: {cuda_ok})")
    if device != "cpu" and not cuda_ok:
        print("[warn] CUDA was requested but is not available, falling back to CPU")
        device = "cpu"

    run_dir = project_dir / args.name

    if last is not None:
        if not is_resumable(last):
            print(f"[error] not a resumable checkpoint: {last}")
            print("        the run is already finished (last.pt has no optimizer state).")
            print(f"        to fine-tune from it use: --model \"{last}\"")
            return 2
        print(f"[info] resuming from {last}")
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        model_path = Path(args.model).expanduser()
        if args.model.endswith(".pt"):
            if not model_path.is_file() and (PROJECT_ROOT / model_path).is_file():
                # allow running the script from any directory
                model_path = PROJECT_ROOT / model_path
            if model_path.is_file():
                print(f"[info] weights   : {model_path}")
            else:
                print(f"[info] weights   : {args.model} (not found locally, will be downloaded)")
        print(f"[info] model     : {args.model}")
        print(f"[info] device    : {device}")
        print(f"[info] epochs    : {args.epochs}  imgsz: {args.imgsz}  batch: {args.batch}")
        model = YOLO(str(model_path) if model_path.is_file() else args.model)
        model.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=args.workers,
            patience=args.patience,
            cache=False if args.cache == "none" else args.cache,
            seed=args.seed,
            project=str(project_dir),
            name=args.name,
            exist_ok=args.exist_ok,
            amp=device != "cpu",
            verbose=True,
        )

    save_dir = Path(getattr(model.trainer, "save_dir", run_dir))
    best = save_dir / "weights" / "best.pt"
    print("")
    print("[done] run directory :", save_dir)
    print("[done] best weights  :", best if best.is_file() else "(not found)")
    print("[done] check results :", save_dir / "results.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
