#
# Copyright(C) 2025
# Pedro H. Penna <pedrohenriquepenna@gmail.com>
# Eduardo Lemos Paschoalini
#

# Directories.
CURRDIR = $(PWD)
RESULTSDIR = $(CURRDIR)/results

###################	x86 ###################

# Builds all kernels for Intel x86.
all-x86:
	cd x86 && $(MAKE) all

# Get compile-time metrics.
cpt-x86:
	cd x86 && $(MAKE) get-compile-time

stats-x86:
	cd x86 && $(MAKE) stats-collect


###################	ARM ###################

# Builds all kernels for ARM.
all-arm:
	cd ARM && $(MAKE) all

# Get compile-time metrics.
cpt-arm:
	cd ARM && $(MAKE) get-compile-time

stats-arm:
	cd arm && $(MAKE) stats-collect

###################	MPPA-256 ###################

# Builds all kernels for MPPA-256.
all-mppa256: 
	mkdir -p bin
	cd mppa256 && $(MAKE) all BINDIR=$(BINDIR)

###################	Gem5 ###################

# Builds all kernels for Gem5 Simulator
# IMPORTANT: Must use a compatible Kernel
all-gem5:
	mkdir -p bin
	cd gem5 && $(MAKE) all BINDIR=$(BINDIR)

# Cleans compilation files.
clean:
	rm -rf $(RESULTSDIR)
	cd x86 && $(MAKE) clean
	cd ARM && $(MAKE) clean