from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ettd.samples import write_sample_pcap


@pytest.fixture
def sample_pcap(tmp_path: Path) -> Path:
    path = tmp_path / "sample.pcap"
    write_sample_pcap(path)
    return path


@pytest.fixture
def repo(tmp_path: Path):
    from ettd.db.repository import Repository

    return Repository(tmp_path / "test.db")
