CFLAGS ?= -O2 -Wall -Wextra

flightsim: flightsim.c
	$(CC) $(CFLAGS) -o $@ $< -lGL -lX11 -lm -lpthread

clean:
	rm -f flightsim
