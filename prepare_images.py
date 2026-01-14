import argparse
import os
import re
import uuid
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_images(images_dir: Path):
    files = []
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    files.sort(key=lambda p: natural_key(p.name))
    return files


def prepare(images_dir: Path, dry_run: bool = False):
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    images = list_images(images_dir)
    if not images:
        print(f"No images found in {images_dir}")
        return 0

    # Phase 1: rename to unique temp names to avoid collisions (e.g., existing 1.jpg)
    token = uuid.uuid4().hex[:8]
    temp_paths = []
    for idx, src in enumerate(images, start=1):
        tmp = images_dir / f"__tmp_{token}_{idx}{src.suffix.lower()}"
        temp_paths.append((src, tmp))

    # Phase 2: rename temps to final 1.jpg, 2.jpg... always .jpg unless original wasn't jpg
    final_paths = []
    for idx, (_, tmp) in enumerate(temp_paths, start=1):
        final = images_dir / f"{idx}.jpg"
        final_paths.append((tmp, final))

    # Print plan
    print(f"Found {len(images)} image(s) in {images_dir}")
    for (src, tmp), (_, final) in zip(temp_paths, final_paths):
        print(f"{src.name} -> {tmp.name} -> {final.name}")

    if dry_run:
        print("Dry-run: no files renamed.")
        return len(images)

    # Execute renames
    for src, tmp in temp_paths:
        src.rename(tmp)

    # If original extension wasn't jpg, we've still temp-renamed it; now convert extension naming only.
    # (We are NOT re-encoding the file; we just rename the file to .jpg for consistency.)
    for tmp, final in final_paths:
        # If a final already exists (shouldn't), fail loudly to avoid data loss.
        if final.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {final}")
        tmp.rename(final)

    print("Done.")
    return len(images)


def main():
    parser = argparse.ArgumentParser(description="Rename images in ./images to 1.jpg,2.jpg,3.jpg,... safely.")
    parser.add_argument("--images-dir", default="images", help="Path to images directory (default: images)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without renaming")
    args = parser.parse_args()

    count = prepare(Path(args.images_dir), dry_run=args.dry_run)
    return 0 if count >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


