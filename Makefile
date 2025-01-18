# Compiler settings
CC := gcc
CFLAGS := -O3 -g -march=native
INCLUDES := -I/opt/OpenBLAS/include
LIBS := -L/opt/OpenBLAS/lib -lopenblas

# Directories
BUILD_DIR := build
TARGET := $(BUILD_DIR)/main

# Source files
SRCS := main.c

# Default target
all: $(TARGET)
	./$(TARGET)

# Create build directory and compile
$(TARGET): $(SRCS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $^ $(LIBS) -o $@

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean build files
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean