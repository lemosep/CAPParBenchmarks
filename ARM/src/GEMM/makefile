#
# Copyright(C) 2025 Eduardo Lemos Paschoalini
#

# Directories.
LOCAL_BINDIR = ./bin
BC_DIR = $(LOCAL_BINDIR)/bytecode

# Source files.
SRC = $(wildcard *.c)

EXEC = gemm

# Bytecode object files.
BC_OBJ = $(patsubst %.c, $(BC_DIR)/%.bc, $(SRC))

# Final linked bytecode file.
LINKED_BC = $(BC_DIR)/$(EXEC)_linked.bc
OBJ_FILES = $(foreach alg, $(REG_ALLOC_ALGOS), $(BINDIR)/$(alg))

# Ensure required directories exist.
$(LOCAL_BINDIR) $(BC_DIR) $(BINDIR):
	@mkdir -p $@

# Compile each .c file into LLVM bytecode.
$(BC_DIR)/%.bc: %.c | $(BC_DIR)
	$(CC) $(CFLAGS) $(LLVMFLAGS) -c $< -o $@

# Link bytecode files.
$(LINKED_BC): $(BC_OBJ)
	llvm-link $^ -o $@

# Function to compile for different register allocation algorithms.
define compile_for_regalloc
$(BINDIR)/$(1): $(LINKED_BC) | $(BINDIR)
	$(LLC) --relocation-model=pic --regalloc=$(1) --filetype=obj $(LINKED_BC) \
		-o $(LOCAL_BINDIR)/$(EXEC)_$(1).o

	$(CC) $(LOCAL_BINDIR)/$(EXEC)_$(1).o $(LIBS) -o $(BINDIR)/$(1) $(LIBARGS)
endef

$(foreach alg, $(REG_ALLOC_ALGOS), $(eval $(call compile_for_regalloc,$(alg))))

# Build all object files.
all: $(LINKED_BC) $(OBJ_FILES)

# Fetches compile times for all algorithms 10 times
get-compile-time: clean all
	@rm -rf llc_times_$(EXEC).txt
	@for i in $$(seq 1 10); do \
		echo "Run $$i" >> llc_times_$(EXEC).txt; \
		for alg in $(REG_ALLOC_ALGOS); do \
			echo -n "$$alg: " >> llc_times_$(EXEC).txt; \
			perf stat $(LLC) -relocation-model=pic --regalloc=$$alg -filetype=obj $(LINKED_BC) \
				-o $(LOCAL_BINDIR)/$(EXEC)_$$alg.o 2>&1 | \
			grep 'seconds time elapsed' >> llc_times_$(EXEC).txt; \
		done; \
		echo "" >> llc_times_$(EXEC).txt; \
	done

stats-collect: $(LINKED_BC)
	@for alg in $(REG_ALLOC_ALGOS); do \
		echo "== $(EXEC), $$alg =="; \
		$(LLC) --regalloc=$$alg -filetype=obj $(LINKED_BC) \
			-o /dev/null --stats 1>/dev/null 2> $(LOCAL_BINDIR)/$(EXEC)_stats_$$alg.txt || true; \
		echo ""; \
		cat $(LOCAL_BINDIR)/$(EXEC)_stats_$$alg.txt; \
	done

# Cleans compilation files.
clean:
	@rm -rf $(LOCAL_BINDIR) llc_times_$(EXEC).txt
