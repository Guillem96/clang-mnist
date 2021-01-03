IDIR=include
CC=gcc
CFLAGS=-I$(IDIR)

ODIR=out
SRC=src

LIBS=-lm

_DEPS=tensor.h
DEPS=$(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ=tensor.o main.o
OBJ=$(patsubst %,$(ODIR)/%,$(_OBJ))

all: dirs mnist

$(ODIR)/%.o: $(SRC)/%.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

.PHONY: clean

dirs:
	mkdir -p $(ODIR)

run: mnist
	./mnist

mnist: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

clean:
	rm -f $(ODIR)/*.o

