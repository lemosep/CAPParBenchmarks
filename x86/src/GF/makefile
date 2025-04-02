#
# Copyright(C) 2025 Eduardo Lemos Paschoalini
#

# Directories.
LOCAL_BINDIR = ./bin
BC_DIR = $(LOCAL_BINDIR)/bytecode

# Source files.
SRC = $(wildcard *.c)

EXEC = gf

# Bytecode object files.
BC_OBJ = $(patsubst %.c, $(BC_DIR)/%.bc, $(SRC))

# Final linked bytecode file.
LINKED_BC = $(BC_DIR)/$(EXEC)_linked.bc
OBJ_FILES = $(foreach alg, $(REG_ALLOC_ALGOS), $(BINDIR)/$(alg).intel)

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
$(BINDIR)/$(1).intel: $(LINKED_BC) | $(BINDIR)
	$(LLC) -relocation-model=pic -regalloc=$(1) -filetype=obj $(LINKED_BC) -o $(LOCAL_BINDIR)/$(EXEC)_$(1).o
	$(CC) $(LOCAL_BINDIR)/$(EXEC)_$(1).o $(LIBS) -o $(BINDIR)/$(1).intel $(LIBARGS)
endef

$(foreach alg, $(REG_ALLOC_ALGOS), $(eval $(call compile_for_regalloc,$(alg))))

# Build all object files.
all: $(LINKED_BC) $(OBJ_FILES)

# Cleans compilation files.
clean:
	@rm -rf $(LOCAL_BINDIR)
