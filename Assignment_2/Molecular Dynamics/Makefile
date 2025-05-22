# Compiler settings
CXX      = g++
NVCC     = nvcc
CXXFLAGS = -O3 -std=c++17
NVCCFLAGS = -O3 -std=c++17

# Output
TARGET   = md_sim
BUILD_DIR = build

# Source files
SRCS = src/main.cu src/md_kernel.cu src/io.cpp src/vtk_writer.cpp

# Default target
.PHONY: all
all: mk-target-dir $(BUILD_DIR)/$(TARGET)

mk-target-dir:
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRCS)

# Aliases without build directory
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
