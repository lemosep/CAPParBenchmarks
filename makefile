#
# Copyright(C) 2014 Pedro H. Penna <pedrohenriquepenna@gmail.com>
#

# Directories.
CURRDIR = $(PWD)
RESULTSDIR = $(CURRDIR)/results

# Builds all kernels for Intel x86.
all-x86:
	cd x86 && $(MAKE) all

# Builds all kernels for MPPA-256.
all-mppa256: 
	mkdir -p bin
	cd mppa256 && $(MAKE) all BINDIR=$(BINDIR)

# Builds all kernels for Gem5 Simulator
# IMPORTANT: Must use a compatible Kernel
all-gem5:
	mkdir -p bin
	cd gem5 && $(MAKE) all BINDIR=$(BINDIR)

# Cleans compilation files.
clean:
	rm -rf $(RESULTSDIR)
	cd x86 && $(MAKE) clean