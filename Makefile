CC = gcc
CFLAGS = -O3 -Wall -Winline -Wshadow -fopenmp -lm -llikwid -fdump-tree-optimized -Wno-unused-variable -Wno-unused-but-set-variable
BIN = rbgs

all: $(BIN)

clean:
	rm -rf $(BIN)

rebuild: clean all

$(BIN): rbgs.c
	$(CC) $(CFLAGS) -o $@ $^
