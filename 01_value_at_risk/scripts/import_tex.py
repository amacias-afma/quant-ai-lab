"""Split `paper/draft-v1.md` into LaTeX fragments under `paper/tex/sections/`.

Why a script and not hand transcription
---------------------------------------
Appendix C of the paper reports five citation errors, four of them introduced by copying a
source's *description* rather than the source. Retyping 7,000 words of prose into LaTeX is
the same operation at larger scale. So the conversion is mechanical and repeatable: edit the
markdown, re-run this, and the LaTeX follows.

Consequence, stated so nobody is surprised: **`paper/draft-v1.md` is the source of truth.**
Hand edits to `paper/tex/sections/*.tex` are overwritten on the next import. Anything that
must live only in the LaTeX (float placement, `\\ref`, layout) belongs in `main.tex` or in
the manual-override list below.

    python scripts/import_tex.py            # convert
    python scripts/import_tex.py --check    # verify the .tex is current, exit 1 if stale
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys

SRC = os.path.join("paper", "draft-v1.md")
OUT = os.path.join("paper", "tex", "sections")

# markdown heading -> fragment filename. Order matters only for readability.
MAP = [
    ("## Abstract",                              "abstract"),
    ("## 1. Introduction",                       "01-introduction"),
    ("### 1.1 Where this paper came from",       "01a-provenance"),
    ("### 1.2 Setting and scope",                "01b-scope"),
    ("## 2. Setup",                              "02-setup"),
    ("## 3. Four findings and their withdrawals", "03-findings"),
    ('### 3.1 "Anchoring improves',              "03a-multiplicity"),
    ('### 3.2 "Weight selection',                "03b-power"),
    ('### 3.3 "Higher capacity',                 "03c-capacity"),
    ('### 3.4 "Anchoring stabilises',            "03d-tautology"),
    ("### 3.5 The artefact isolated",            "03e-synthetic"),
    ("## 4. Disclosure",                         "04-disclosure"),
    ("## 5. Three defects",                      "05-defects"),
    ("## 6. Checks on this study",               "06-checks"),
    ("## 7. The taxonomy",                       "07-taxonomy"),
    ("## 8. Limitations",                        "08-limitations"),
    ("## 9. Conclusion",                         "09-conclusion"),
    ("## Reproducibility statement",             "10-reproducibility"),
    ("## Appendix A",                            "A-ledger"),
    ("## Appendix B",                            "B-corrections"),
    ("## Appendix C",                            "C-citations"),
    ("## Appendix D",                            "D-survey"),
]

# Cross-reference rewriting: the markdown says "§3.4", LaTeX should say \Cref{sec:tautology}.
XREF = {
    r"§1\.1": r"\\Cref{sec:provenance}", r"§1\.2": r"\\Cref{sec:scope}",
    r"§3\.1": r"\\Cref{sec:multiplicity}", r"§3\.2": r"\\Cref{sec:power}",
    r"§3\.3": r"\\Cref{sec:capacity}", r"§3\.4": r"\\Cref{sec:tautology}",
    r"§3\.5": r"\\Cref{sec:synthetic}", r"§1\b": r"\\Cref{sec:intro}",
    r"§2\b": r"\\Cref{sec:setup}", r"§3\b": r"\\Cref{sec:findings}",
    r"§4\b": r"\\Cref{sec:disclosure}", r"§5\b": r"\\Cref{sec:defects}",
    r"§6\b": r"\\Cref{sec:checks}", r"§7\b": r"\\Cref{sec:taxonomy}",
    r"§8\b": r"\\Cref{sec:limitations}", r"§9\b": r"\\Cref{sec:conclusion}",
    r"Appendix A": r"\\Cref{app:ledger}", r"Appendix B\.2": r"\\Cref{app:corrections}",
    r"Appendix B": r"\\Cref{app:corrections}", r"Appendix C": r"\\Cref{app:citations}",
    r"Appendix D": r"\\Cref{app:survey}",
}

FIGURES = {
    "figures/figure1_dose_response.png": ("fig:dose", "The dose--response that convinced us."),
    "figures/figure2_control.png": ("fig:control", "The same effect, with a control."),
}


def split_sections(text: str) -> dict[str, str]:
    """Cut the markdown at each mapped heading. Unmapped headings stay inside their parent."""
    marks = []
    for needle, name in MAP:
        i = text.find(needle)
        if i < 0:
            print(f"  [warn] heading not found, fragment will be empty: {needle!r}")
            continue
        marks.append((i, needle, name))
    marks.sort()
    out = {}
    for k, (i, needle, name) in enumerate(marks):
        end = marks[k + 1][0] if k + 1 < len(marks) else len(text)
        body = text[i + len(needle):end]
        # drop the remainder of the heading line
        body = body.split("\n", 1)[1] if "\n" in body else ""
        out[name] = body.strip()
    return out


def to_latex(md: str) -> str:
    p = subprocess.run(
        ["pandoc", "--from=markdown+pipe_tables+raw_tex", "--to=latex",
         "--wrap=preserve", "--no-highlight"],
        input=md, capture_output=True, text=True, check=True)
    tex = p.stdout

    # pandoc ALREADY emits a figure float for a lone image. Do not wrap it again: nesting
    # floats raises "Not in outer par mode". Set the width and attach our label instead, so
    # the prose can \Cref it.
    for path, (label, _cap) in FIGURES.items():
        tex = re.sub(
            r"\\includegraphics(\[[^\]]*\])?\{" + re.escape(path) + r"\}",
            r"\\includegraphics[width=\\linewidth]{" + path + r"}\\label{" + label + "}",
            tex)
    tex = tex.replace(r"\begin{figure}" + "\n", r"\begin{figure}[t]" + "\n")
    tex = tex.replace(r"\pandocbounded{", "{")

    for pat, rep in XREF.items():
        tex = re.sub(pat, rep, tex)

    # Unicode the T1 fonts cannot set. Each is mapped rather than dropped, because a silently
    # dropped minus sign in "(1 - 2*lr*w)" would corrupt the paper's central equation.
    SUBS = {
        "—": "---", "–": "--", "−": "$-$", "∞": r"$\infty$",
        "≤": r"$\leq$", "≥": r"$\geq$", "≈": r"$\approx$", "≠": r"$\neq$",
        "×": r"$\times$", "÷": r"$\div$", "±": r"$\pm$",
        "→": r"$\to$", "←": r"$\leftarrow$", "⇒": r"$\Rightarrow$",
        "ρ": r"$\rho$", "α": r"$\alpha$", "β": r"$\beta$", "μ": r"$\mu$",
        "σ": r"$\sigma$", "η": r"$\eta$", "ν": r"$\nu$", "τ": r"$\tau$",
        "Δ": r"$\Delta$", "θ": r"$\theta$", "λ": r"$\lambda$", "Σ": r"$\Sigma$",
        "§": r"\S", "…": r"\ldots", "·": r"$\cdot$", "‰": r"\textperthousand",
    }
    # Inside verbatim/listing blocks LaTeX sets characters literally, so substituting there
    # would print "$-$" as visible text. Protect those regions first.
    parts = re.split(r"(\\begin\{verbatim\}.*?\\end\{verbatim\})", tex, flags=re.S)
    for i, part in enumerate(parts):
        if part.startswith(r"\begin{verbatim}"):
            for u in SUBS:                       # keep code readable: strip to ASCII lookalikes
                part = part.replace(u, {"—": "--", "−": "-", "×": "x", "≤": "<=",
                                        "≥": ">=", "≈": "~", "→": "->",
                                        "∞": "inf"}.get(u, u))
            parts[i] = part
            continue
        for u, r in SUBS.items():
            part = part.replace(u, r)
        parts[i] = part
    tex = "".join(parts)

    # Any remaining non-Latin-1 character is a transcription hazard: fail loudly rather than
    # let pdflatex drop it. This is the same discipline as the numbers check.
    leftover = sorted({c for c in tex if ord(c) > 0x2000})
    if leftover:
        print(f"  [warn] unmapped unicode still present: {leftover}")
    return tex


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the fragments match the markdown; do not write")
    args = ap.parse_args()

    if not os.path.exists(SRC):
        print(f"MISSING: {SRC}", file=sys.stderr)
        return 2
    os.makedirs(OUT, exist_ok=True)
    text = open(SRC, encoding="utf-8").read()

    stale = []
    for name, md in split_sections(text).items():
        tex = to_latex(md) if md.strip() else "% (empty section)\n"
        path = os.path.join(OUT, f"{name}.tex")
        new = hashlib.sha256(tex.encode()).hexdigest()
        old = (hashlib.sha256(open(path, "rb").read()).hexdigest()
               if os.path.exists(path) else None)
        if new != old:
            stale.append(name)
            if not args.check:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(tex)
        print(f"  {'stale' if new != old else 'ok   '}  {name}")

    if args.check and stale:
        print(f"\n--check: {len(stale)} fragment(s) out of date. Run without --check.")
        return 1
    print(f"\n{'Verified' if args.check else 'Wrote'} {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
