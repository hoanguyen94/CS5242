import subprocess
import tempfile
from datetime import datetime
from pathlib import Path


dir_ = Path(__file__).parent
ts = datetime.now().strftime('%y%m%d%H%M%S')

# --- Slides: Marp → html + pdf ---
src = dir_ / "part1_presentation.md"
for ext in ("html", "pdf"):
    dst = dir_ / f"part1_presentation_{ts}.{ext}"
    subprocess.run(
        ["marp", "--html", str(src), "-o", str(dst), "--allow-local-files"],
        check=True,
    )
    print(f"Exported: {dst}")

# --- Speaker script: pandoc → html → Chrome headless → pdf ---
script_src = dir_ / "part1_script.md"
script_css = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 780px; margin: 2em auto; padding: 0 1em; line-height: 1.55; color: #1a1a1a; }
h1 { font-size: 1.6em; border-bottom: 2px solid #333; padding-bottom: 0.3em; }
h2 { font-size: 1.2em; margin-top: 1.6em; color: #2a4a7a; }
table { border-collapse: collapse; margin: 1em 0; }
th, td { border: 1px solid #bbb; padding: 0.35em 0.7em; }
th { background: #f0f0f0; }
hr { border: none; border-top: 1px dashed #ccc; margin: 2em 0; }
code { background: #f4f4f4; padding: 0.1em 0.3em; border-radius: 3px; }
strong { color: #1a1a1a; }
"""
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
script_pdf = dir_ / f"part1_script_{ts}.pdf"
with tempfile.TemporaryDirectory() as tmp:
    tmp_dir = Path(tmp)
    css_path = tmp_dir / "script.css"
    css_path.write_text(script_css)
    html_path = tmp_dir / "script.html"
    subprocess.run(
        [
            "pandoc", str(script_src), "-s",
            "--metadata", "title=Speaker Script — Presenter 1",
            "-c", str(css_path),
            "-o", str(html_path),
        ],
        check=True,
    )
    subprocess.run(
        [
            chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
            f"--print-to-pdf={script_pdf}",
            f"file://{html_path}",
        ],
        check=True,
    )
print(f"Exported: {script_pdf}")
