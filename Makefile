# Regra genérica: .tex -> .pdf (slides e sucintas)
# Duas passadas explícitas: a barra de progresso do rodapé depende da 2ª,
# e o latexmk sozinho para na 1ª passada (overflow aritmético do total=0).
# Compila dentro da pasta do .tex para resolver o tema (../) e as figuras locais.
%.pdf: %.tex
	cd $(dir $<) && xelatex -interaction=nonstopmode $(notdir $<) >/dev/null 2>&1
	cd $(dir $<) && xelatex -interaction=nonstopmode $(notdir $<) >/dev/null 2>&1

# Extensões de arquivos temporários gerados pela compilação LaTeX (beamer)
clean:
	find . -type f \
		\( -name '*.aux' -o -name '*.log' -o -name '*.out' \
		 -o -name '*.toc' -o -name '*.nav' -o -name '*.snm' \
		 -o -name '*.vrb' -o -name '*.bbl' -o -name '*.blg' \
		 -o -name '*.fls' -o -name '*.fdb_latexmk' \
		 -o -name '*.synctex.gz' -o -name '*.xdv' -o -name '*.dvi' \) \
		-delete

AULAS = aula01 aula02 aula03 aula04 aula05 aula06 aula07 \
        aula08 aula09 aula10 aula11 aula12 aula13 aula14

# Atalho: make aula04 compila o deck principal da aula
.PHONY: $(AULAS)
$(AULAS):
	$(MAKE) --no-print-directory $@/$@.pdf

# Recompila todas as aulas e relata linhas Overfull (padrão: zero)
verificar:
	@for d in $(AULAS) docs; do \
	  (cd $$d && latexmk -xelatex -interaction=nonstopmode *.tex >/dev/null 2>&1); \
	  n=$$(grep -i overfull $$d/*.log 2>/dev/null | wc -l); \
	  echo "$$d: $$n linha(s) Overfull"; \
	done

.PHONY: clean verificar
