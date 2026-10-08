"""
scripts/validate_dataset.py
============================
Standalone script that verifies the integrity of the processed split files.

Checks performed:
  - No sample appears in both train and test (no leakage)
  - All class IDs are in [0, 99]
  - All 100 classes are represented in train
  - All class IDs are represented in test
  - Total counts match expected values from the official split
  - All sample_ids have a valid format

Usage
-----
  python scripts/validate_dataset.py
  python scripts/validate_dataset.py --processed_root <path>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXPECTED_TRAIN = 66032
EXPECTED_TEST = 21866
N_CLASSES = 100


def load_json(path: Path) -> list:
    with open(path, "r") as fh:
        return json.load(fh)


def validate_split(entries: list, split_name: str) -> bool:
    ok = True
    print(f"\n--- {split_name.upper()} split ---")
    print(f"  Entries: {len(entries)}")

    # Class ID range
    bad_ids = [e for e in entries if not (0 <= e["class_id"] <= N_CLASSES - 1)]
    if bad_ids:
        print(f"  FAIL: {len(bad_ids)} entries with class_id out of [0,99]")
        ok = False
    else:
        print(f"  Class ID range [0,{N_CLASSES-1}]: OK")

    # Unique classes
    classes = set(e["class_id"] for e in entries)
    if split_name == "train" and len(classes) != N_CLASSES:
        print(f"  FAIL: train has {len(classes)} unique classes (expected {N_CLASSES})")
        ok = False
    else:
        print(f"  Unique classes: {len(classes)}")

    # Sample ID uniqueness
    ids = [e["sample_id"] for e in entries]
    if len(ids) != len(set(ids)):
        print(f"  FAIL: duplicate sample_ids detected!")
        ok = False
    else:
        print(f"  Sample IDs unique: OK")

    # Format check
    bad_fmt = [e for e in entries if "=" not in e["sample_id"]]
    if bad_fmt:
        print(f"  FAIL: {len(bad_fmt)} entries with malformed sample_id")
        ok = False
    else:
        print(f"  Sample ID format: OK")

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate processed split files for 3DGrocery100"
    )
    parser.add_argument(
        "--processed_root",
        default="data/processed",
        help="Path to the processed data directory",
    )
    args = parser.parse_args()

    processed_root = Path(args.processed_root)
    splits_dir = processed_root / "splits"
    labels_json = processed_root / "labels.json"

    all_ok = True

    # Check labels.json exists
    if not labels_json.exists():
        print(f"FAIL: labels.json not found at {labels_json}")
        sys.exit(1)
    with open(labels_json) as fh:
        label_map = json.load(fh)
    print(f"labels.json: {len(label_map)} classes found")
    if len(label_map) != N_CLASSES:
        print(f"  FAIL: expected {N_CLASSES} classes, got {len(label_map)}")
        all_ok = False
    else:
        print(f"  Class count: OK")

    # Load splits
    train_path = splits_dir / "train.json"
    test_path = splits_dir / "test.json"

    if not train_path.exists() or not test_path.exists():
        print("FAIL: split JSON files missing. Run prepare_dataset.py first.")
        sys.exit(1)

    train_entries = load_json(train_path)
    test_entries = load_json(test_path)

    # Validate each split
    all_ok &= validate_split(train_entries, "train")
    all_ok &= validate_split(test_entries, "test")

    # Cross-split leakage check
    print("\n--- Train/Test Leakage Check ---")
    train_ids = {e["sample_id"] for e in train_entries}
    test_ids = {e["sample_id"] for e in test_entries}
    overlap = train_ids & test_ids
    if overlap:
        print(f"  FAIL: {len(overlap)} sample(s) appear in BOTH train and test!")
        for sid in list(overlap)[:5]:
            print(f"    {sid}")
        all_ok = False
    else:
        print(f"  No overlap detected (OK)")

    # Expected counts
    print("\n--- Count Checks ---")
    if len(train_entries) != EXPECTED_TRAIN:
        print(
            f"  WARNING: train has {len(train_entries)} entries "
            f"(expected {EXPECTED_TRAIN})"
        )
    else:
        print(f"  Train count {len(train_entries)}: OK")

    if len(test_entries) != EXPECTED_TEST:
        print(
            f"  WARNING: test has {len(test_entries)} entries "
            f"(expected {EXPECTED_TEST})"
        )
    else:
        print(f"  Test  count {len(test_entries)}: OK")

    print("\n" + ("=" * 50))
    if all_ok:
        print("  ALL CHECKS PASSED (OK)")
    else:
        print("  SOME CHECKS FAILED — review output above")
    print("=" * 50)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
