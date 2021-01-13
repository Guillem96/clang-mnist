IDIR=include
CC=gcc
CFLAGS=`sdl2-config --libs --cflags` --std=c99 -Wall -I$(IDIR)

ODIR=out
SRC=src

LIBS=-lm -lSDL2_image

_DEPS=tensor.h tensor_pool.h mnist.h plot.h nn.h
DEPS=$(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ=tensor.o tensor_pool.o mnist.o plot.o nn.o main.o
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

