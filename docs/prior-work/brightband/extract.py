"""Split the downloaded Next.js chunks into one pretty-printed file per webpack module,
and pull the readable text out of the methodology page.

Run with:  uv run --no-project --with jsbeautifier --with beautifulsoup4 python extract.py

Writes source/modules/<chunk>/<module-id>.js and source/methodology.txt. Prints which
modules mention the scorecard so they can be read first.
"""

from __future__ import annotations

import re
from pathlib import Path

import jsbeautifier
from bs4 import BeautifulSoup

HERE = Path(__file__).parent
SRC = HERE / "source"

# A webpack module entry: `1234:(e,s,a)=>{` or `1234:function(e,t,n){`.
MODULE_START = re.compile(
    r"(?<![\w$.])(\d{2,6}):(?:\([\w$,]*\)=>|function\([\w$,]*\))\{"
)

KEYWORDS = ("scorecard", "baseline", "lead_time", "/api/", "rmse", "significan")


def split_modules(js: str) -> dict[str, str]:
    """Return {module id: source}, cutting at each module start in the chunk."""
    starts = [(m.start(), m.group(1)) for m in MODULE_START.finditer(js)]
    out: dict[str, str] = {}
    for (pos, mid), (end, _) in zip(starts, starts[1:] + [(len(js), None)]):
        out.setdefault(mid, js[pos:end])
    return out


def main() -> None:
    opts = jsbeautifier.default_options()
    opts.indent_size = 2
    hits = []
    for chunk in sorted((SRC / "chunks").glob("*.js")):
        if chunk.name.startswith(("polyfills", "webpack", "4bd1b696")):
            continue  # core-js, the webpack runtime, react-dom
        outdir = SRC / "modules" / chunk.stem
        outdir.mkdir(parents=True, exist_ok=True)
        for mid, code in split_modules(chunk.read_text()).items():
            pretty = jsbeautifier.beautify(code, opts)
            (outdir / f"{mid}.js").write_text(pretty)
            found = [k for k in KEYWORDS if k in code.lower()]
            if found:
                hits.append((f"{chunk.stem}/{mid}", len(code), found))

    soup = BeautifulSoup((SRC / "methodology.html").read_text(), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    main_el = soup.find("main") or soup
    text = "\n".join(
        line.strip() for line in main_el.get_text("\n").splitlines() if line.strip()
    )
    (SRC / "methodology.txt").write_text(text + "\n")

    print("modules mentioning scorecard keywords:")
    for name, size, found in hits:
        print(f"  {name:40s} {size:7d}  {', '.join(found)}")
    print(f"methodology.txt: {len(text)} chars")


if __name__ == "__main__":
    main()
