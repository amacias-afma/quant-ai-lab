"""Reconcile `paper/tex/refs.bib` against the audited bibliography in `paper/references.md`.

The point of `references.md` is that every entry is marked verified against the published
record. A `.bib` file that drifts from it silently reintroduces exactly the errors Appendix C
documents — so the two are checked against each other rather than trusted to agree.

Checks:
  1. every bib entry's first author surname and year appear in references.md
  2. no bib entry is a placeholder (`Anonymous`, `TO BE`, empty author)
  3. every bib key is actually cited somewhere in paper/tex/

    python scripts/check_bib.py
"""
from __future__ import annotations

import os
import re
import sys
import unicodedata

BIB = os.path.join("paper", "tex", "refs.bib")
REFS = os.path.join("paper", "references.md")
TEXDIR = os.path.join("paper", "tex")

ENTRY = re.compile(r"@(\w+)\{([^,]+),(.*?)\n\}", re.S)
FIELD = lambda body, name: (                                            # noqa: E731
    m.group(1).strip() if (m := re.search(rf"{name}\s*=\s*\{{(.*?)\}},?\s*\n", body, re.S))
    else "")


def _fold(s: str) -> str:
    """Base letters only: decompose accents, drop combining marks and punctuation."""
    d = unicodedata.normalize("NFKD", s)
    d = "".join(c for c in d if not unicodedata.combining(c))
    return re.sub(r"[^A-Za-z]", "", d).lower()


def surname(author_field: str) -> str:
    """First author's surname, from either 'Last, First' or brace-protected forms."""
    first = author_field.split(" and ")[0].strip()
    # Strip LaTeX accent commands: {\"a} -> a, {\'a} -> a, {\`e} -> e, {\ldots}
    first = re.sub(r"\\[a-zA-Z]+", "", first)
    first = re.sub(r"[{}\\'`\"^~]", "", first)
    if "," in first:
        return first.split(",")[0].strip()
    parts = first.split()
    return parts[-1].strip() if parts else ""


def main() -> int:
    for p in (BIB, REFS):
        if not os.path.exists(p):
            print(f"MISSING: {p}", file=sys.stderr)
            return 2

    bib = open(BIB, encoding="utf-8").read()
    refs = open(REFS, encoding="utf-8").read()
    tex = ""
    for root, _, files in os.walk(TEXDIR):
        for f in files:
            if f.endswith(".tex"):
                tex += open(os.path.join(root, f), encoding="utf-8").read()

    problems, entries = [], []
    for _kind, key, body in ENTRY.findall(bib):
        key = key.strip()
        author, year = FIELD(body, "author"), FIELD(body, "year")
        entries.append(key)

        if not author or "Anonymous" in author or "TO BE" in body.upper():
            problems.append(f"{key}: placeholder or missing author")
            continue

        sn = surname(author)
        # Accent-insensitive containment. references.md renders diacritics as real unicode
        # ("Petnehazi" with an acute a); the bib escapes them as LaTeX. Decompose both to
        # base letters before comparing, or the check fails on spelling that is correct.
        plain = _fold(sn)
        hay = _fold(refs)
        if plain and plain not in hay:
            problems.append(f"{key}: first author '{sn}' not found in references.md")
        elif year and year not in refs:
            problems.append(f"{key}: year {year} not found in references.md")

    uncited = [k for k in entries if f"\\cite" not in tex or k not in tex]

    print(f"{len(entries)} entries in {BIB}")
    if problems:
        print(f"\n{len(problems)} reconciliation problem(s):")
        for p in problems:
            print(f"  - {p}")
    if uncited:
        print(f"\n{len(uncited)} entry/entries not yet cited in the LaTeX "
              f"(expected while the draft is being imported):")
        for k in uncited:
            print(f"  - {k}")

    if problems:
        print("\nFAIL: the bibliography does not reconcile with the audited reference list.")
        return 1
    print("\nOK: every bib entry reconciles with paper/references.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
