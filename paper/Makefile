paper.pdf: main.tex references.bib $(wildcard figures/*)
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	bibtex main
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	mv main.pdf paper.pdf

clean:
	rm -f *.aux *.bbl *.blg *.log *.out main.pdf

.PHONY: clean

# arXiv bundle: source, compiled bibliography and figures (arXiv compiles the .tex itself).
arxiv: paper.pdf
	pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
	bibtex main >/dev/null
	tar -czf arxiv-submission.tar.gz main.tex main.bbl references.bib figures
	rm -f *.aux *.blg *.log *.out main.pdf main.bbl

.PHONY: arxiv
