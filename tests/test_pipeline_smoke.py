"""Entry-point smoke: the CLI shape must work before the pipeline does."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cli_help():
    out = subprocess.run(
        [sys.executable, str(ROOT / "run_pipeline.py"), "--help"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert out.returncode == 0
    assert "self-consistency" in out.stdout
