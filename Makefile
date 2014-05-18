CC = gcc
CFLAGS = -O3 -Wall -Winline -Wshadow -fopenmp -lm
BIN = rbgs

all: $(BIN)

clean:
	rm -rf $(BIN)

rebuild: clean all

$(BIN): rbgs.c
	$(CC) $(CFLAGS) -o $@ $^
