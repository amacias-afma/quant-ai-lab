# Build the paper. Two targets, one source.
#
#   make named   arXiv / Zenodo build, with authors
#   make anon    TMLR submission, anonymised (double-blind)
#   make import  regenerate sections/*.tex from ../draft-v1.md
#   make check   verify the LaTeX is current and the numbers match the result files
#   make clean
#
# The anonymisation switch is a file, not an edit: `anon.flag` exists or it does not.
# That way the two builds cannot drift apart through someone forgetting to toggle a macro.

# -g forces a rebuild. latexmk cannot see that anon.flag changed — it is read by \IfFileExists
# at compile time, not \input — so without -g a switch between builds silently reuses the
# previous PDF. That failure mode (a "new" artefact that is actually the old one) is the same
# class as the stale result files in Appendix B, so it is closed here rather than remembered.
LATEX  = latexmk -pdf -g -interaction=nonstopmode -halt-on-error
REPO   = ../..
NAMED  = four-findings-named
ANON   = four-findings-anon

.PHONY: named anon import check clean all verify

all: named

import:
	cd $(REPO) && python scripts/import_tex.py

named: import
	@echo '\\anonymousfalse' > anon.tex
	$(LATEX) -jobname=$(NAMED) main.tex
	@echo "-> $(NAMED).pdf  (authors shown; for arXiv/Zenodo)"

anon: import
	@echo '\\anonymoustrue' > anon.tex
	$(LATEX) -jobname=$(ANON) main.tex
	@echo '\\anonymousfalse' > anon.tex
	@echo "-> $(ANON).pdf   (anonymised; for TMLR)"

# `cat >`, not `mv` or `cp`. Both must unlink the destination, which some mounts forbid;
# a redirect truncates in place and always works. When `mv` failed here, the build printed
# "Output written" and then silently kept serving the PREVIOUS pdf.
# That is the third instance in this project of a partial regeneration producing a stale
# artefact (see Appendix B for the first two), so `verify` below checks the contents of the
# pdf rather than trusting that the build did what it said.

# Assert what each PDF actually contains. TMLR review is double-blind; shipping the named
# PDF would be unrecoverable.
verify: named anon
	@test $(NAMED).pdf -nt main.tex \
	  && echo "OK  named pdf is newer than the source" \
	  || { echo "FAIL: named pdf is STALE - the copy step did not take"; exit 1; }
	@pdftotext $(ANON).pdf - | grep -q "Anonymous authors" \
	  && echo "OK  anon build is anonymised" \
	  || { echo "FAIL: anon build is NOT anonymised"; exit 1; }
	@pdftotext $(ANON).pdf - | grep -qi "afmaing.cl" \
	  && { echo "FAIL: anon build leaks an author email"; exit 1; } \
	  || echo "OK  anon build carries no author contact"
	@pdftotext $(NAMED).pdf - | grep -qi "afmaing.cl" \
	  && echo "OK  named build carries author details" \
	  || { echo "FAIL: named build is missing author details"; exit 1; }

check:
	cd $(REPO) && python scripts/import_tex.py --check
	cd $(REPO) && python scripts/refresh_paper_figures.py --check
	cd $(REPO) && python scripts/check_bib.py
	@echo "LaTeX is current, figures file matches the CSVs, bibliography reconciles."

clean:
	latexmk -C
	@echo '\\anonymousfalse' > anon.tex
	rm -f *.bbl *.blg *.aux *.log *.out *.fdb_latexmk *.fls
