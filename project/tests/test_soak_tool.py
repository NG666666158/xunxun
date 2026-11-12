"""packaging/soak_test.py: the pieces that decide pass or fail, and the synthetic folder.
The run itself takes 30+ minutes and is a release step (packaging/README_RELEASE.md)."""

import importlib.util
import os
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "packaging" / "soak_test.py"
spec = importlib.util.spec_from_file_location("soak_test", SCRIPT)
soak = importlib.util.module_from_spec(spec)
sys.modules["soak_test"] = soak  # dataclasses look their module up there
spec.loader.exec_module(soak)


def _rows(values, key="rss_mb", every=60):
    return [{"elapsed_s": i * every, key: value} for i, value in enumerate(values)]


def test_growth_compares_medians_of_first_and_last_samples():
    rows = _rows([300, 302, 299, 301, 340, 350, 345])
    start, end, grew = soak.growth(rows, "rss_mb", 3)
    assert (start, end, grew) == (300, 345, 45)
    # One spike at either end does not decide it.
    assert soak.growth(_rows([300, 900, 300, 300, 300, 300]), "rss_mb", 3)[2] == 0


def test_slope_is_per_hour():
    rows = _rows([100 + i for i in range(31)])  # +1 MB a minute
    assert soak.slope_per_hour(rows, "rss_mb") == pytest.approx(60)
    assert soak.slope_per_hour(_rows([5]), "rss_mb") == 0


def test_sampler_reads_this_process():
    how, sample = soak.sampler()
    numbers = sample(os.getpid())
    assert how in ("psutil", "win32", "proc")
    assert numbers.rss_mb > 10 and numbers.private_mb > 0 and numbers.handles > 0 and numbers.threads >= 1


@pytest.mark.skipif(sys.platform != "win32", reason="the venv launcher is a Windows thing")
def test_child_processes_finds_a_started_program():
    import subprocess

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert dict(soak.child_processes(os.getpid()))[child.pid].lower().startswith("python")
    finally:
        child.kill()
        child.wait()


def test_synthetic_folder_has_mixed_small_files(tmp_path):
    made = soak.make_corpus(tmp_path, 60)
    files = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert len(files) == 60 == sum(made.values())
    assert {"txt", "md"} <= set(made) and len(made) >= 4
    assert all(path.stat().st_size < 200_000 for path in files)


def test_file_changes_cycle_through_add_edit_rename_delete(tmp_path):
    soak.make_corpus(tmp_path, 20)
    rng = soak.random.Random(1)
    actions = [soak.change_files(tmp_path, step, rng).split()[0] for step in range(4)]
    assert actions == ["add", "edit", "rename", "delete"]
