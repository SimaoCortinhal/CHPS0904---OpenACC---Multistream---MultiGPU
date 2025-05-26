# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved. 

FC := nvfortran
FLAGS := -O3 -w -ldl
ACCFLAGS := -ta=tesla:managed,lineinfo -Minfo=accel
NVTXLIB := -I/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/lib64 -lnvToolsExt

rdf: nvtx.f90 rdf.f90
	${FC} ${FLAGS} ${ACCFLAGS} -o rdf nvtx.f90 rdf.f90 ${NVTXLIB} 

clean:
	rm -f *.o rdf
