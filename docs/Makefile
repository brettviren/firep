# Build the technical note.
#
#   make          - regenerate figures and values, then build the PDF
#   make pdf      - build the PDF from whatever is already in figs/
#   make figs     - regenerate figs/*.pdf and figs/values.tex only
#   make clean    - remove LaTeX build products
#   make realclean- also remove the generated figures
#
# `make figs` re-reads the solved models in ../runs, so improving a model and
# re-running the pipeline updates every figure and every quoted number.

DOC     := siren-for-lartpc
RUNDIR  := ../runs
FIGDIR  := figs
PYTHON  := ../.venv/bin/python
SECTIONS:= $(wildcard sections/*.tex)
STAMP   := $(FIGDIR)/values.tex

export MPLCONFIGDIR ?= $(CURDIR)/.mplconfig

.PHONY: all pdf figs clean realclean

all: figs pdf

figs: $(STAMP)

$(STAMP): ../scripts/make_figs.py ../src/firep/plot.py ../src/firep/electronics.py
	$(PYTHON) ../scripts/make_figs.py $(RUNDIR) $(FIGDIR)

pdf: $(DOC).pdf

$(DOC).pdf: $(DOC).tex $(SECTIONS) $(STAMP)
	pdflatex -interaction=nonstopmode $(DOC).tex >/dev/null
	pdflatex -interaction=nonstopmode $(DOC).tex >/dev/null
	@echo "built $(DOC).pdf"

clean:
	rm -f $(DOC).aux $(DOC).log $(DOC).out $(DOC).toc $(DOC).pdf
	rm -rf .mplconfig

realclean: clean
	rm -f $(FIGDIR)/*.pdf $(STAMP)
