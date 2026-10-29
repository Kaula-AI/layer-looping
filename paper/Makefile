paper.pdf: main.tex references.bib placeholders.tex $(wildcard generated/*.tex figures/*)
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	bibtex main
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	mv main.pdf paper.pdf

clean:
	rm -f *.aux *.bbl *.blg *.log *.out main.pdf

.PHONY: clean
