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

# Sec. 16: evaluated from the PDSP learned-response runs, when they exist.
BWRUN   := ../runs/pdsp
figs: $(FIGDIR)/bandwidth.tex
$(FIGDIR)/bandwidth.tex: ../scripts/bandwidth_figs.py ../src/firep/direct.py
	@if [ -f $(BWRUN)/direct.pt ] && [ -d $(BWRUN)/exp ]; then \
	    $(PYTHON) ../scripts/bandwidth_figs.py $(BWRUN) $(FIGDIR); \
	else echo "no $(BWRUN)/direct.pt: skipping Sec. 16"; fi

# Sec. 17: the 3D pixel reproduction, when its runs exist (wants a GPU).
PIXRUN  := ../runs/pixel
figs: $(FIGDIR)/pixel.tex
$(FIGDIR)/pixel.tex: ../scripts/pixel_pochoir.py ../src/firep/pixel.py
	@if [ -f $(PIXRUN)/weighting9.pt ]; then \
	    $(PYTHON) ../scripts/pixel_pochoir.py $(PIXRUN) $(FIGDIR); \
	else echo "no $(PIXRUN): skipping Sec. 17"; fi

# Sec. 17: drift lines through a gap, three treatments of the PCB (wants a GPU).
figs: $(FIGDIR)/pixel-gap.tex
$(FIGDIR)/pixel-gap.tex: ../scripts/pixel_gap.py ../src/firep/pixel.py
	@if [ -f $(PIXRUN)/weighting9.pt ]; then \
	    $(PYTHON) ../scripts/pixel_gap.py $(PIXRUN)/gap $(FIGDIR); \
	else echo "no $(PIXRUN): skipping the gap study"; fi

# Sec. 17: against R. Fan's pochoir store with the PCB (wants a GPU and the store).
RADOSTORE ?= /nfs/data/1/bviren/rado/results/store_larpix_v2a_5x5_15cm_pathsongrid
figs: $(FIGDIR)/pixel-rado.tex
$(FIGDIR)/pixel-rado.tex: ../scripts/pixel_rado.py ../src/firep/pixel.py
	@if [ -d $(RADOSTORE) ]; then \
	    RADO_STORE=$(RADOSTORE) $(PYTHON) ../scripts/pixel_rado.py $(PIXRUN)/rado $(FIGDIR); \
	else echo "no $(RADOSTORE): skipping the PCB comparison"; fi

# Sec. 17: against R. Fan's latest (two-grid) pochoir solves of the standard pad.
RADORESULTS ?= /nfs/data/1/bviren/rado/results
figs: $(FIGDIR)/pixel-hybrid.tex
$(FIGDIR)/pixel-hybrid.tex: ../scripts/pixel_hybrid.py ../src/firep/pixel.py
	@if [ -d $(RADORESULTS) ]; then \
	    RADO_RESULTS=$(RADORESULTS) $(PYTHON) ../scripts/pixel_hybrid.py $(PIXRUN)/hybrid $(FIGDIR); \
	else echo "no $(RADORESULTS): skipping the standard-design comparison"; fi

# Sec. 17: pad shape and size (wants a GPU; about two hours from scratch).
figs: $(FIGDIR)/pixel-shapes.tex
$(FIGDIR)/pixel-shapes.tex: ../scripts/pixel_shapes.py ../src/firep/pixel.py
	@if [ -f $(PIXRUN)/weighting9.pt ]; then \
	    $(PYTHON) ../scripts/pixel_shapes.py $(PIXRUN)/shapes $(FIGDIR); \
	else echo "no $(PIXRUN): skipping the shape study"; fi

# Sec. 17: optimising the grid design by differentiation (GPU, ~30 min; needs scipy).
figs: $(FIGDIR)/pixel-opt.tex
$(FIGDIR)/pixel-opt.tex: ../scripts/pixel_optimize.py ../src/firep/pixel_opt.py
	$(PYTHON) ../scripts/pixel_optimize.py $(PIXRUN)/opt $(FIGDIR)

# Sec. 17.3: the diagram of the spectral single layer (CPU, seconds, no runs).
figs: $(FIGDIR)/pixel-basis.pdf
$(FIGDIR)/pixel-basis.pdf: ../scripts/pixel_basis_fig.py ../src/firep/pixel.py
	$(PYTHON) ../scripts/pixel_basis_fig.py $(FIGDIR)

# Appendix B: the effective-plane model against the 2D response (CPU, seconds).
figs: $(FIGDIR)/effplane.pdf
$(FIGDIR)/effplane.pdf: ../scripts/effective_plane.py ../src/firep/bias.py $(RUNDIR)/response.npz
	$(PYTHON) ../scripts/effective_plane.py $(RUNDIR) $(FIGDIR)

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
