# Makefile pour Jacobi MPI+CUDA+NVSHMEM (sans LTO)
MPICXX    := mpicxx
NVCC      := nvcc


NVSHMEM_INC  := $(NVSHMEM_HOME)/include
NVSHMEM_LIB  := $(NVSHMEM_HOME)/lib

CUDA_HOME   ?= /usr/local/cuda
CUDA_ARCH   ?= sm_90 

NVCCFLAGS := -O3 -arch=$(CUDA_ARCH) -std=c++17 --expt-relaxed-constexpr -rdc=true
NVCCFLAGS += -gencode=arch=compute_90,code=sm_90
NVCCFLAGS += -Xcompiler "-Wall -Wextra" -ccbin=$(MPICXX)
NVCCFLAGS += -I$(NVSHMEM_INC) -I$(CUDA_HOME)/include

LDFLAGS   := -L$(NVSHMEM_LIB) -lnvshmem -lnvshmem_host
LDFLAGS   += -L$(CUDA_HOME)/lib64 -lcudart -lmpi

TARGET    := jacobi_nvshmem
SRC       := jacobi.cu

.PHONY: all clean nsight

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o *.qdrep *.sqlite

nsight: $(TARGET)
	nsys profile -t cuda,mpi --force-overwrite=true \
	-o nsight/jacobi_profile_$(CUDA_ARCH) \
	mpirun -np 4 ./$(TARGET) 512 512 1000 1e-6 0
