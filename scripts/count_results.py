from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "model_results"

counts = Counter(p.parts[-4] for p in ROOT.glob("*/Predictions/*/testResample*.csv"))
for name, n in sorted(counts.items()):
    print(f"{n:6d}  {name}")
print(f"{sum(counts.values()):6d}  TOTAL")
