import subprocess
import shutil
import sys
from datetime import datetime
from pathlib import Path

dir_ = Path(__file__).parent
src = dir_ / "presentation.md"
ts = datetime.now().strftime('%y%m%d%H%M%S')

marp = shutil.which("marp")
npx = shutil.which("npx")
if marp:
    marp_cmd = [marp]
elif npx:
    marp_cmd = [npx, "@marp-team/marp-cli"]
else:
    raise SystemExit(
        "Marp CLI is not installed. Install Node.js/npm, then run:\n"
        "  npm install -g @marp-team/marp-cli\n"
        "or, if npm/npx is available, use:\n"
        "  npx @marp-team/marp-cli --html report/presentation.md -o report/presentation.html --allow-local-files"
    )

for ext in ("html", "pdf"):
    dst = dir_ / f"presentation_{ts}.{ext}"
    result = subprocess.run(
        [*marp_cmd, "--html", str(src), "-o", str(dst), "--allow-local-files"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        combined_output = "\n".join(
            output for output in (result.stdout, result.stderr) if output
        )
        if combined_output:
            print(combined_output, file=sys.stderr)
        if "require is not defined in ES module scope" in combined_output:
            raise SystemExit(
                "Marp failed because the active Node.js runtime is incompatible with "
                "one of Marp's CommonJS dependencies. Use a stable LTS Node version, "
                "then reinstall Marp:\n"
                "  conda install -c conda-forge 'nodejs>=20,<23'\n"
                "  npm install -g @marp-team/marp-cli"
            )
        raise SystemExit(f"Marp export failed for {dst}")
    print(f"Exported: {dst}")
