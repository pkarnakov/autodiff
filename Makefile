PANDOC = pandoc
STYLE = .
MAINREPO = ..
RSYNC = rsync -a -i --update

all: \
	index.html \

.md.html:
	$(PANDOC) -s --css $(STYLE)/pandoc.css "$<" -o "$@"

demos:
	mkdir -p "$@"
	git -C $(MAINREPO) rev-parse --short HEAD > $@/.gitrev
	$(RSYNC) $(MAINREPO)/build_wasm/{*.js,*.css,*.wasm,*.html,*.svg,favicon.png} $@/

demos_commit:
	git commit --edit -m "update demos from $$(cat demos/.gitrev)"

clean:
	rm -vf *.pdf *.html

.PHONY: all clean demos demos_commit
.SUFFIXES: .md .html
