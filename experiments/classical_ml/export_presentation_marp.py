import subprocess
from datetime import datetime
from pathlib import Path

dir_ = Path(__file__).parent
src = dir_ / "presentation.md"
ts = datetime.now().strftime('%y%m%d%H%M%S')

for fmt, flag in [("pptx", "--pptx"), ("html", "--html")]:
    dst = dir_ / f"presentation_{ts}.{fmt}"
    subprocess.run(
        ["marp", flag, str(src), "-o", str(dst), "--allow-local-files"],
        check=True,
    )
    print(f"Exported: {dst}")
