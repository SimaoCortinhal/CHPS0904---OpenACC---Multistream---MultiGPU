# Makefile pour compiler le solveur Jacobi MPI + CUDA (multi‐GPU, sans overlap)
MPICXX    := mpicxx
NVCC      := nvcc
# Ajustez -arch=sm_90 
NVCCFLAGS := -O3 -arch=sm_90 -std=c++11 -ccbin=$(MPICXX) -use_fast_math 
LDFLAGS   := -lm -lmpi 
TARGET    := jacobi_cuda
SRC       := jacobi.cu

.PHONY: all run clean nsight

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)

# Profilage avec Nsight Systems (nsys)
nsight: $(TARGET)
	mkdir -p nsight
	nsys profile -o nsight/$(TARGET)_profile \
		mpirun -np 4 ./jacobi_cuda 4096 4096 10000 1e-6 0

