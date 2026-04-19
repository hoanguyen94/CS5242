import subprocess
from datetime import datetime
from pathlib import Path

dir_ = Path(__file__).parent
src = dir_ / "presentation.md"
ts = datetime.now().strftime('%y%m%d%H%M%S')

dst = dir_ / f"presentation_{ts}.html"
subprocess.run(
    ["marp", "--html", str(src), "-o", str(dst), "--allow-local-files"],
    check=True,
)
print(f"Exported: {dst}")
