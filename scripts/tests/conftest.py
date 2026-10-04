import sys
from pathlib import Path

# Tests import the scripts as top-level modules, the same way CI runs them.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
