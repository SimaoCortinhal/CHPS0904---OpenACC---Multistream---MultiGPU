# Copyright (c) 2021 NVIDIA Corporation.  All rights reserved. 

CC := nvc++
ACCFLAGS := -acc -fast -Minfo=accel

rdf: rdf.cpp
	${CC} ${CFLAGS} ${ACCFLAGS} -o rdf rdf.cpp  -I/apps/2025/manual_install/nvhpc/24.11/Linux_aarch64/24.11/cuda/include 

clean:
	rm -f *.o rdf
