#
# Copyright(C) 2014 Pedro H. Penna <pedrohenriquepenna@gmail.com>
#

.PHONY: fast fn gf is km lu tsp rt

# Builds all kernels.
all: fast fn gf is km lu tsp rt

get-compile-time: fast-cpt fn-cpt gf-cpt is-cpt km-cpt lu-cpt tsp-cpt rt-cpt

#  nb

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

# KM kernel.
km:	
	@mkdir -p $(BINDIR)/km
	@cd KM && $(MAKE) all BINDIR=$(BINDIR)/km

km-cpt:
	@echo "======================================="
	@echo "Generating results for KM Kernel"
	@cd KM && $(MAKE) get-compile-time BINDIR=$(BINDIR)/km

# LU kernel.
lu:
	@mkdir -p $(BINDIR)/lu
	@cd LU && $(MAKE) all BINDIR=$(BINDIR)/lu

lu-cpt:
	@echo "======================================="
	@echo "Generating results for LU Kernel"
	@cd LU && $(MAKE) get-compile-time BINDIR=$(BINDIR)/lu

# TSP kernel.
tsp:
	@mkdir -p $(BINDIR)/tsp
	@cd TSP && $(MAKE) all BINDIR=$(BINDIR)/tsp

tsp-cpt:
	@echo "======================================="
	@echo "Generating results for TSP Kernel"
	@cd TSP && $(MAKE) get-compile-time BINDIR=$(BINDIR)/tsp 

# RT kernel.
rt:
	@mkdir -p $(BINDIR)/rt
	@cd RT && $(MAKE) all BINDIR=$(BINDIR)/rt

rt-cpt:
	@echo "======================================="
	@echo "Generating results for RT Kernel"
	@cd RT && $(MAKE) get-compile-time BINDIR=$(BINDIR)/rt

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
