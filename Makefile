PANDOC = pandoc
PFLAGS =
STYLE = .
REPO = ../autodiff
RSYNC = rsync -a -i --update

all: \
	index.html \

.md.html:
	$(PANDOC) -s --css $(STYLE)/pandoc.css $(PFLAGS) "$<" -o "$@"

wasm:
	(cd "$(REPO)" && git rev-parse --short HEAD) > .gitrev
	$(RSYNC) $(REPO)/build_wasm/{poisson{.js,.wasm,_inc.js,.css},libs} poisson/
	$(RSYNC) $(REPO)/build_wasm/poisson.html poisson/index.html

commit:
	git commit --edit -m "update from devel `cat .gitrev`"

clean:
	rm -vf *.pdf *.html

.PHONY: all clean
.SUFFIXES: .md .html
