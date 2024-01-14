PANDOC = pandoc
STYLE = .
REPO = ..
RSYNC = rsync -a -i --update

all: \
	index.html \

.md.html:
	$(PANDOC) -s --css $(STYLE)/pandoc.css "$<" -o "$@"

wasm:
	(cd "$(REPO)" && git rev-parse --short HEAD) > .gitrev
	$(RSYNC) $(REPO)/build_wasm/{poisson{.js,.wasm,_inc.js,.css},favicon.png,libs} poisson/
	$(RSYNC) $(REPO)/build_wasm/poisson.html poisson/index.html

commit:
	git commit --edit -m "update from devel `cat .gitrev`"

clean:
	rm -vf *.pdf *.html

.PHONY: all clean
.SUFFIXES: .md .html
