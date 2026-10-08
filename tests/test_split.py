"""
tests/test_split.py
====================
Unit tests for fixed train/test split integrity.

Run with:
  cd <project_root>
  pytest tests/test_split.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SPLITS_DIR = Path("data/processed/splits")
LABELS_JSON = Path("data/processed/labels.json")

EXPECTED_TRAIN = 66032
EXPECTED_TEST = 21866
N_CLASSES = 100


def splits_exist():
    return (SPLITS_DIR / "train.json").exists() and (SPLITS_DIR / "test.json").exists()


pytestmark = pytest.mark.skipif(
    not splits_exist(),
    reason="Run prepare_dataset.py first to generate splits",
)


@pytest.fixture(scope="module")
def train_entries():
    with open(SPLITS_DIR / "train.json") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def test_entries():
    with open(SPLITS_DIR / "test.json") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def label_map():
    with open(LABELS_JSON) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Fixed Split Integrity Tests
# ---------------------------------------------------------------------------

class TestSplitIntegrity:
    def test_train_count(self, train_entries):
        assert len(train_entries) == EXPECTED_TRAIN, (
            f"Expected {EXPECTED_TRAIN} train entries, got {len(train_entries)}"
        )

    def test_test_count(self, test_entries):
        assert len(test_entries) == EXPECTED_TEST, (
            f"Expected {EXPECTED_TEST} test entries, got {len(test_entries)}"
        )

    def test_no_train_test_overlap(self, train_entries, test_entries):
        """No sample ID should appear in both train and test."""
        train_ids = {e["sample_id"] for e in train_entries}
        test_ids = {e["sample_id"] for e in test_entries}
        overlap = train_ids & test_ids
        assert len(overlap) == 0, (
            f"Train/test overlap detected: {list(overlap)[:5]}"
        )

    def test_all_train_ids_unique(self, train_entries):
        ids = [e["sample_id"] for e in train_entries]
        assert len(ids) == len(set(ids)), "Duplicate IDs in train split"

    def test_all_test_ids_unique(self, test_entries):
        ids = [e["sample_id"] for e in test_entries]
        assert len(ids) == len(set(ids)), "Duplicate IDs in test split"

    def test_train_has_all_100_classes(self, train_entries):
        classes = set(e["class_id"] for e in train_entries)
        assert len(classes) == N_CLASSES, (
            f"Train split has {len(classes)} classes (expected {N_CLASSES})"
        )

    def test_test_has_all_100_classes(self, test_entries):
        classes = set(e["class_id"] for e in test_entries)
        assert len(classes) == N_CLASSES, (
            f"Test split has {len(classes)} classes (expected {N_CLASSES})"
        )

    def test_all_class_ids_valid_train(self, train_entries):
        bad = [e for e in train_entries if not (0 <= e["class_id"] <= N_CLASSES - 1)]
        assert len(bad) == 0, f"Invalid class IDs in train: {bad[:3]}"

    def test_all_class_ids_valid_test(self, test_entries):
        bad = [e for e in test_entries if not (0 <= e["class_id"] <= N_CLASSES - 1)]
        assert len(bad) == 0, f"Invalid class IDs in test: {bad[:3]}"

    def test_sample_id_format_train(self, train_entries):
        """Every sample_id should match <class>=<rest>."""
        bad = [e for e in train_entries if "=" not in e["sample_id"]]
        assert len(bad) == 0, f"Malformed sample IDs in train: {bad[:3]}"

    def test_sample_id_format_test(self, test_entries):
        bad = [e for e in test_entries if "=" not in e["sample_id"]]
        assert len(bad) == 0, f"Malformed sample IDs in test: {bad[:3]}"

    def test_class_name_consistent_with_id(self, train_entries, label_map):
        """class_name in entry must match labels.json for class_id."""
        for e in train_entries[::1000]:  # check every 1000th entry
            expected_name = label_map[str(e["class_id"])]["name"]
            assert e["class_name"] == expected_name, (
                f"class_name mismatch: entry={e['class_name']}, "
                f"labels.json={expected_name}"
            )

    def test_required_fields_present(self, train_entries, test_entries):
        required = {"index", "sample_id", "class_id", "class_name", "status"}
        for e in train_entries[:10] + test_entries[:10]:
            missing = required - set(e.keys())
            assert not missing, f"Missing fields: {missing}"


# ---------------------------------------------------------------------------
# Labels JSON Tests
# ---------------------------------------------------------------------------

class TestLabelsJSON:
    def test_has_100_entries(self, label_map):
        assert len(label_map) == N_CLASSES

    def test_keys_are_string_ints(self, label_map):
        for k in label_map:
            assert k.isdigit(), f"Non-integer key: {k}"

    def test_ids_range_0_to_99(self, label_map):
        ids = [int(k) for k in label_map]
        assert min(ids) == 0
        assert max(ids) == N_CLASSES - 1

    def test_no_duplicate_names(self, label_map):
        names = [v["name"] for v in label_map.values()]
        assert len(names) == len(set(names)), "Duplicate class names in labels.json"

    def test_all_entries_have_name_and_category(self, label_map):
        for k, v in label_map.items():
            assert "name" in v, f"Missing 'name' in class {k}"
            assert "category" in v, f"Missing 'category' in class {k}"
            assert v["name"], f"Empty 'name' in class {k}"
