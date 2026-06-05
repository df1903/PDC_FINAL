NVCC      = nvcc
NVCCFLAGS = -O2 -arch=sm_75

all: scoring_kernel

scoring_kernel: scoring_kernel.cu
	$(NVCC) $(NVCCFLAGS) -o scoring_kernel scoring_kernel.cu

clean:
	rm -f scoring_kernel
