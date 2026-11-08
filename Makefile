CC     = gcc
CFLAGS = -O2 -Wall -Wextra -pthread
LDLIBS = -lrt -lm

Proyecto-de-matrices-inversas: main.c matrix.c invert.c matrix.h
	$(CC) $(CFLAGS) -o $@ main.c matrix.c invert.c $(LDLIBS)

clean:
	rm -f Proyecto-de-matrices-inversas results.csv *.bin
