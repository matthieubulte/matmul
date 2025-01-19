# Compiler settings
CC := gcc
CFLAGS := -O3 -g -funroll-loops -march=native
INCLUDES := -I/opt/OpenBLAS/include
LIBS := -L/opt/OpenBLAS/lib -lopenblas

# Code signing settings
ENTITLEMENTS := debug.entitlements
CODESIGN := codesign
CODESIGN_FLAGS := --entitlements $(ENTITLEMENTS) -f -s -

# Directories
BUILD_DIR := build
TARGET := $(BUILD_DIR)/main

# Source files
SRCS := main.c

# Default target
all: sign
	./$(TARGET)

# Compilation
$(TARGET): $(SRCS) | $(BUILD_DIR)
	@echo "Compiling..."
	$(CC) $(CFLAGS) $(INCLUDES) $^ $(LIBS) -o $@

# Signing
sign: $(TARGET)
	@echo "Signing binary..."
	$(CODESIGN) $(CODESIGN_FLAGS) $(TARGET)
	@echo "Verifying signature..."
	$(CODESIGN) -v $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean sign