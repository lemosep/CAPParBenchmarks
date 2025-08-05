#
# Copyright(C) 2014 Pedro H. Penna <pedrohenriquepenna@gmail.com>
#

.PHONY: fast fn gf is km

# Builds all kernels.
all: fast fn

get-compile-time: fast-cpt fn-cpt

# gf is km

# gf is km lu tsp rt nb

# FAST kernel.
fast:
	@mkdir -p $(BINDIR)/fast
	@cd FAST && $(MAKE) all BINDIR=$(BINDIR)/fast

fast-cpt:
	@cd FAST && $(MAKE) get-compile-time BINDIR=$(BINDIR)/fast

# FN kernel.
fn:
	@mkdir -p $(BINDIR)/fn
	@cd FN && $(MAKE) all BINDIR=$(BINDIR)/fn

fn-cpt:
	@cd FN && $(MAKE) get-compile-time BINDIR=$(BINDIR)/fn

# Builds GF kernel.
gf:
	@mkdir -p $(BINDIR)/gf
	@cd GF && $(MAKE) all BINDIR=$(BINDIR)/gf

# Builds IS kernel.
is:
	@mkdir -p $(BINDIR)/is
	@cd IS && $(MAKE) all BINDIR=$(BINDIR)/is

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
