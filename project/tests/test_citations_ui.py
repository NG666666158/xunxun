"""The UI's citation linking (marker forms, matching to source cards) runs under Node's test runner."""

from pathlib import Path
import shutil
import subprocess

import pytest

NODE = shutil.which("node")
TESTS = Path(__file__).resolve().parent / "js" / "citations.test.cjs"


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_citation_forms_matching_and_drafts():
    result = subprocess.run(
        [NODE, "--test", str(TESTS)], capture_output=True, text=True, encoding="utf-8", timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
