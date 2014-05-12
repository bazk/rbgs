CC = gcc
CFLAGS = -O3 -Wall -Winline -Wshadow -fopenmp
LDFLAGS = -lm

OBJS = rbgs.o
BIN = rbgs

all: $(BIN)

clean:
	rm -rf $(BIN) $(OBJS)

rebuild: clean all

$(BIN): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<
