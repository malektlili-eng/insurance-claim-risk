"""
Data provenance: no reported number may be mistaken for real-data results.

Checks that the generated/real flag is computed correctly, travels with the
model bundle, is exposed by the API and app, and is stated directly under the
Results heading of the README (not buried at the bottom).
"""
from __future__ import annotations

import os
import re
import shutil

import pytest

from conftest import ROOT, needs_model
from data.make_dataset import (dataset_provenance, TRAIN_PATH, FINGERPRINT_PATH,
                               _sha256)

README = os.path.join(ROOT, "README.md")


def test_generated_data_is_reported_as_generated():
    assert dataset_provenance(TRAIN_PATH) == "generated"


def test_any_other_training_file_is_reported_as_real(tmp_path):
    """Dropping a different CSV into data/ (e.g. the real one) flips the flag."""
    shutil.copy(FINGERPRINT_PATH, tmp_path / os.path.basename(FINGERPRINT_PATH))
    fake_real = tmp_path / "train_Insurance.csv"
    with open(TRAIN_PATH, encoding="utf-8") as src:
        lines = src.readlines()
    lines[1] = lines[1].replace("2014", "2015", 1) if "2014" in lines[1] else lines[1] + " "
    fake_real.write_text("".join(lines), encoding="utf-8")
    assert dataset_provenance(str(fake_real)) == "real"


def test_missing_fingerprint_means_real(tmp_path):
    copy = tmp_path / "train_Insurance.csv"
    shutil.copy(TRAIN_PATH, copy)
    assert dataset_provenance(str(copy)) == "real"


def test_fingerprint_matches_shipped_training_file():
    with open(FINGERPRINT_PATH) as f:
        assert f.read().split()[0] == _sha256(TRAIN_PATH)


@needs_model
def test_bundle_records_the_data_it_was_trained_on(metadata):
    assert metadata["data_source"] == dataset_provenance(TRAIN_PATH)
    assert metadata["metrics"]["data_source"] == metadata["data_source"]


@needs_model
def test_api_health_exposes_provenance(client):
    body = client.get("/health").json()
    assert body["data_source"] == dataset_provenance(TRAIN_PATH)
    if body["data_source"] == "generated":
        assert "generated" in body["data_warning"].lower()


def test_streamlit_app_warns_on_generated_data():
    with open(os.path.join(ROOT, "app", "streamlit_app.py"), encoding="utf-8") as f:
        src = f.read()
    assert 'data_source == "generated"' in src and "st.warning(" in src


def test_figures_are_stamped_with_data_source():
    with open(os.path.join(ROOT, "train.py"), encoding="utf-8") as f:
        src = f.read()
    # every figure is saved through _save(), which applies the provenance footer
    assert "_footer(fig, source)" in src
    assert src.count("_save(fig,") >= 5
    assert "savefig(" in src and src.count("savefig(") == 1


def _section(md: str, heading: str) -> str:
    m = re.search(rf"^## {re.escape(heading)}\s*$(.*?)(?=^## |\Z)", md, re.M | re.S)
    assert m, f"README has no '## {heading}' section"
    return m.group(1)


def test_readme_flags_provenance_directly_under_results():
    """Reviewer requirement: the flag sits right under the Results heading."""
    with open(README, encoding="utf-8") as f:
        md = f.read()
    body = _section(md, "Results")
    first_lines = [l for l in body.strip().splitlines() if l.strip()][:2]
    head = " ".join(first_lines).lower()
    assert "generated" in head and "about the data" in head, first_lines


def test_readme_flags_provenance_before_any_metric():
    with open(README, encoding="utf-8") as f:
        md = f.read()
    first_number = re.search(r"\b0\.\d{3}\b|\d+(\.\d+)?%", md)
    first_flag = md.lower().find("generated")
    assert first_number and 0 <= first_flag < first_number.start()
