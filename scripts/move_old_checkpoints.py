#!/usr/bin/env python3
"""Moves old checkpoint models, merged trees, and GGUF files from C: to E: to free up disk space."""

import os
import shutil
import sys

SOURCE_DIR = os.path.abspath("checkpoints")
DEST_DIR = r"E:\classone_checkpoints_archive"

# Active models and critical metadata to KEEP on C:
KEEP_ITEMS = {
    "classone_composite_70pct",
    "classone_composite_70pct_merged",
    "snapshots",
}


def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"[!] Source checkpoints directory does not exist: {SOURCE_DIR}")
        sys.exit(1)

    os.makedirs(DEST_DIR, exist_ok=True)
    print(f"[*] Moving old checkpoints from:\n    {SOURCE_DIR}\n  to:\n    {DEST_DIR}\n")

    items_to_move = []
    for name in sorted(os.listdir(SOURCE_DIR)):
        if name in KEEP_ITEMS:
            print(f"  [KEEP] {name} (active champion model / metadata)")
            continue
        items_to_move.append(name)

    print(f"\n[*] Found {len(items_to_move)} items to move to E: drive.\n")

    moved_count = 0
    total_bytes = 0

    for name in items_to_move:
        src_path = os.path.join(SOURCE_DIR, name)
        dst_path = os.path.join(DEST_DIR, name)

        if os.path.exists(dst_path):
            print(f"  [!] Destination already exists, skipping: {dst_path}")
            continue

        try:
            # Measure size
            if os.path.isdir(src_path):
                sz = sum(os.path.getsize(os.path.join(r, f)) for r, d, files in os.walk(src_path) for f in files)
            else:
                sz = os.path.getsize(src_path)

            print(f"  --> Moving {name} ({sz / 1e9:.2f} GB)...", end="", flush=True)
            shutil.move(src_path, dst_path)
            print(" [DONE]")

            moved_count += 1
            total_bytes += sz
        except Exception as exc:
            print(f" [FAILED: {exc}]")

    print(f"\n[✓] Successfully moved {moved_count} items ({total_bytes / 1e9:.2f} GB) to {DEST_DIR}!")

    # Check free disk space
    c_usage = shutil.disk_usage("C:\\")
    e_usage = shutil.disk_usage("E:\\")

    print("\nUpdated Disk Free Space:")
    print(f"  C: drive: {c_usage.free / 1e9:.2f} GB free (total: {c_usage.total / 1e9:.2f} GB)")
    print(f"  E: drive: {e_usage.free / 1e9:.2f} GB free (total: {e_usage.total / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
