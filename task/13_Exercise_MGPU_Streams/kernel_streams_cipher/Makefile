CUDACXX=nvcc
CUDACXXFLAGS=-arch=sm_70 -O3
CXXFLAGS=-march=native -fopenmp
NSYS=nsys profile
NSYSFLAGS=--stats=true --force-overwrite=true

all: mgpu_streams

mgpu_streams: mgpu_streams.cu
	$(CUDACXX) $(CUDACXXFLAGS) -Xcompiler="$(CXXFLAGS)" mgpu_streams.cu -o mgpu_streams

mgpu_solution: mgpu_solution.cu
	$(CUDACXX) $(CUDACXXFLAGS) -Xcompiler="$(CXXFLAGS)" mgpu_solution.cu -o mgpu_solution

profile: mgpu_streams
	$(NSYS) $(NSYSFLAGS) -o mgpu_streams-report ./mgpu_streams

profile_solution: mgpu_solution
	$(NSYS) $(NSYSFLAGS) -o mgpu_streams-solution-report ./mgpu_solution

clean:
	rm -f mgpu_streams mgpu_solution *.qdrep *.sqlite
