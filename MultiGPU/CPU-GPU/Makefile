# Makefile pour compiler le solveur Jacobi en CUDA
NVCC       := nvcc
NVCCFLAGS  := -O3 -arch=sm_90 -std=c++11       
LDFLAGS    := -lm                         
TARGET     := jacobi_cuda
SRC        := jacobi.cu

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
