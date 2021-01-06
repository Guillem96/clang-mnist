IDIR=include
CC=gcc
CFLAGS=`sdl2-config --libs --cflags` --std=c99 -Wall -I$(IDIR)

ODIR=out
SRC=src

LIBS=-lm -lSDL2_image

_DEPS=tensor.h mnist.h plot.h
DEPS=$(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ=tensor.o mnist.o plot.o main.o
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

