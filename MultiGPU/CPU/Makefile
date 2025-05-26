# Makefile pour compiler le solveur Jacobi séquentiel
CC      := gcc
CFLAGS  := -O3 -Wall -march=native -funroll-loops -ffast-math -std=c99 -fopenmp
TARGET  := jacobi_seq
SRC     := jacobi.c laplace2d.c
HDR     := laplace2d.h

all: $(TARGET)

$(TARGET): $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) -lm

clean:
	rm -f $(TARGET)
