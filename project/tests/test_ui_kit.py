"""The UI's pure logic runs under Node's test runner: the interaction kit (wheel picker,
tri-state selection, deck, file stacks), the status row / preview preparation, adding
files (image choice, change notes), the guided start, the data-safety rows and when the
page looks for a new version by itself."""

from pathlib import Path
import shutil
import subprocess

import pytest

NODE = shutil.which("node")
JS = Path(__file__).resolve().parent / "js"


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
@pytest.mark.parametrize("name", ["ui.test.cjs", "status.test.cjs", "library.test.cjs", "guide.test.cjs", "safety.test.cjs", "privacy.test.cjs", "updatecheck.test.cjs"])
def test_ui_logic(name):
    result = subprocess.run(
        [NODE, "--test", str(JS / name)], capture_output=True, text=True, encoding="utf-8", timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
