"""Root conftest — adds src paths so tests can import rules and shared."""

import sys
from pathlib import Path

_root = Path(__file__).parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root.parent / "shared" / "src"))
