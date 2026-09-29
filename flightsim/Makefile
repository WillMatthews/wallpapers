CFLAGS ?= -O2 -Wall -Wextra

# Wayland protocol glue (generated): wlr-layer-shell for --layer, and the
# xdg-shell it refers to for popups
XDG_SHELL = $(shell pkg-config --variable=pkgdatadir wayland-protocols)/stable/xdg-shell/xdg-shell.xml
PROTO_H = proto-layer-shell.h
PROTO_C = proto-layer-shell.c proto-xdg-shell.c

flightsim: flightsim.c $(PROTO_H) $(PROTO_C)
	$(CC) $(CFLAGS) -o $@ flightsim.c $(PROTO_C) -lGL -lX11 -lEGL -lwayland-client -lwayland-egl -lm -lpthread

proto-layer-shell.h: protocols/wlr-layer-shell-unstable-v1.xml
	wayland-scanner client-header $< $@
proto-layer-shell.c: protocols/wlr-layer-shell-unstable-v1.xml
	wayland-scanner private-code $< $@
proto-xdg-shell.c: $(XDG_SHELL)
	wayland-scanner private-code $< $@

clean:
	rm -f flightsim $(PROTO_H) $(PROTO_C)
