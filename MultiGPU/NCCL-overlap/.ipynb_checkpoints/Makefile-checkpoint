# Makefile pour compiler le solveur Jacobi MPI + CUDA (multi-GPU, sans overlap)
MPICXX    := mpicxx
NVCC      := nvcc

NCCL_PATH := $(shell spack location -i nccl)

NVCCFLAGS := -O3 -arch=sm_90 -std=c++11 -ccbin=$(MPICXX) -use_fast_math \
             -I$(NCCL_PATH)/include

LDFLAGS   := -L$(NCCL_PATH)/lib -lm -lmpi -lnccl -lcudart

TARGET    := jacobi_nccl
SRC       := jacobi.cu

.PHONY: all clean nsight

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)

nsight: $(TARGET)
	mkdir -p nsight
	nsys profile -o nsight/$(TARGET)_profile \
		mpirun -np 4 ./$(TARGET) 20 20 10000 1e-6 0
