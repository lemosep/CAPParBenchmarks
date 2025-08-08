#
# Copyright(C) 2014 Pedro H. Penna <pedrohenriquepenna@gmail.com>
#

KERNELS := fast fn gf is km lu tsp rt nb

.PHONY: fast fn gf is km lu tsp rt nb

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

# Get amount of virtual registers
vregs-count: 
	for kernel in $(KERNELS); do \
		DIR=$$(printf '%s' $$kernel | tr '[:lower:]' '[:upper:]'); \
		$(MAKE) -C $$DIR vregs-count $(BINDIR)/$$kernel; \
	done

stats-collect:
	@cd FAST && $(MAKE) stats-collect $(BINDIR)/fast

# Cleans compilation files.
clean:
	for kernel in $(KERNELS); do \
		DIR=$$(printf '%s' $$kernel | tr '[:lower:]' '[:upper:]'); \
		$(MAKE) -C $$DIR clean; \
	done
