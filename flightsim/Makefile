CFLAGS ?= -O2 -Wall -Wextra

flightsim: flightsim.c
	$(CC) $(CFLAGS) -o $@ $< -lGL -lX11 -lm

clean:
	rm -f flightsim
