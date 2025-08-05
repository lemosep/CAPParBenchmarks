#
# Copyright(C) 2014 Pedro H. Penna <pedrohenriquepenna@gmail.com>
#

.PHONY: fast fn gf is km

# Builds all kernels.
all: fast fn gf is

get-compile-time: fast-cpt fn-cpt gf-cpt is-cpt

# km

# km lu tsp rt nb

# FAST kernel.
fast:
	@mkdir -p $(BINDIR)/fast
	@cd FAST && $(MAKE) all BINDIR=$(BINDIR)/fast

fast-cpt:
	@echo "======================================="
	@echo "Generating results for FAST Kernel"
	@cd FAST && $(MAKE) get-compile-time BINDIR=$(BINDIR)/fast

# FN kernel.
fn:
	@mkdir -p $(BINDIR)/fn
	@cd FN && $(MAKE) all BINDIR=$(BINDIR)/fn

fn-cpt:
	@echo "======================================="
	@echo "Generating results for FN Kernel"
	@cd FN && $(MAKE) get-compile-time BINDIR=$(BINDIR)/fn

# GF kernel.
gf:
	@mkdir -p $(BINDIR)/gf
	@cd GF && $(MAKE) all BINDIR=$(BINDIR)/gf

gf-cpt:
	@echo "======================================="
	@echo "Generating results for GF Kernel"
	@cd GF && $(MAKE) get-compile-time BINDIR=$(BINDIR)/gf

# IS kernel.
is:
	@mkdir -p $(BINDIR)/is
	@cd IS && $(MAKE) all BINDIR=$(BINDIR)/is

is-cpt:
	@echo "======================================="
	@echo "Generating results for IS Kernel"
	@cd IS && $(MAKE) get-compile-time BINDIR=$(BINDIR)/is

# Builds KM kernel.
km:	
	@mkdir -p $(BINDIR)/km
	@cd KM && $(MAKE) all BINDIR=$(BINDIR)/km

# Builds LU kernel.
lu:
	@mkdir -p $(BINDIR)/lu
	@cd LU && $(MAKE) all BINDIR=$(BINDIR)/lu

# Builds TSP kernel.
tsp:
	cd TSP && $(MAKE) all

# Builds RT kernel.
rt:
	cd RT && $(MAKE) all

# Builds NB kernel.
nb:
	cd NB && $(MAKE) all

# Cleans compilation files.
clean:
	cd FAST && $(MAKE) clean
	cd FN && $(MAKE) clean
	cd GF && $(MAKE) clean
	cd IS && $(MAKE) clean
	cd KM && $(MAKE) clean
	cd LU && $(MAKE) clean
	cd TSP && $(MAKE) clean
	cd RT && $(MAKE) clean
	cd NB && $(MAKE) clean
