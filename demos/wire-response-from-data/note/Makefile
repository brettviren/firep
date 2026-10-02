# Build the note from whatever figures/macros the recipes have made.
# Regenerate those with snakemake from the demo directory, not from here.

DOC      := wrfd
SECTIONS := $(wildcard sections/*.tex)
FIGTEX   := $(wildcard figs/*.tex)

.PHONY: pdf clean

pdf: $(DOC).pdf

$(DOC).pdf: $(DOC).tex $(SECTIONS) $(FIGTEX)
	pdflatex -interaction=nonstopmode $(DOC).tex >/dev/null
	pdflatex -interaction=nonstopmode $(DOC).tex >/dev/null
	@echo "built $(DOC).pdf"

clean:
	rm -f $(DOC).aux $(DOC).log $(DOC).out $(DOC).toc $(DOC).pdf
