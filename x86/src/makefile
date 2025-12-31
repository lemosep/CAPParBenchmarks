#
# Copyright(C) 2025 Eduardo Lemos Paschoalini
#

#					CAPbench			        | 	  ParRes									
KERNELS := fast fn gf is km lu tsp rt nb fft bls bc dgemm sp sten gemm 

.PHONY: fast fn gf is km lu tsp rt nb fft bls bc dgemm sp sten gemm

# Builds all kernels.
all: 
	for kernel in $(KERNELS); do \
		mkdir -p $(BINDIR)/$$kernel; \
		DIR=$$(printf '%s' $$kernel | tr '[:lower:]' '[:upper:]'); \
		$(MAKE) -C $$DIR all BINDIR=$(BINDIR)/$$kernel; \
	done

get-compile-time: 
	for kernel in $(KERNELS); do \
		DIR=$$(printf '%s' $$kernel | tr '[:lower:]' '[:upper:]'); \
		echo "======================================="; \
		echo "Generating results for $$DIR Kernel"; \
		$(MAKE) -C $$DIR get-compile-time BINDIR=$(BINDIR)/$$kernel; \
	done

stats-collect:
	for kernel in $(KERNELS); do \
		DIR=$$(printf '%s' $$kernel | tr '[:lower:]' '[:upper:]'); \
		$(MAKE) -C $$DIR stats-collect BINDIR=$(BINDIR)/$$kernel; \
	done

# Cleans compilation files.
clean:
	for kernel in $(KERNELS); do \
		DIR=$$(printf '%s' $$kernel | tr '[:lower:]' '[:upper:]'); \
		$(MAKE) -C $$DIR clean; \
	done
