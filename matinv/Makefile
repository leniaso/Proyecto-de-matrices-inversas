CC     = gcc
CFLAGS = -O2 -Wall -Wextra -pthread
LDLIBS = -lrt -lm

matinv: main.c matrix.c invert.c matrix.h
	$(CC) $(CFLAGS) -o $@ main.c matrix.c invert.c $(LDLIBS)

clean:
	rm -f matinv results.csv *.bin
