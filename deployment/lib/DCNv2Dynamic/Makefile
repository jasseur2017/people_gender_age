
BUILD_DIR := build

CFLAGS = -c -Xcompiler -fPIC -w

LFLAGS = -shared -lcublas -lnvinfer
# support Jetson Nano, TX1: 53, TX2: 62, Xavier: 72, Quadro RTX 8000 75
GENCODES = -gencode arch=compute_53,code=sm_53 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75

OBJS := $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(wildcard *.cpp))
CUOBJS := $(patsubst %.cu, $(BUILD_DIR)/%.cu.o, $(wildcard *.cu))

NVCC := /usr/local/cuda/bin/nvcc

.PHONY: all
all: $(BUILD_DIR)/dcn_plugin.so

$(BUILD_DIR)/%.o: %.cpp
	$(NVCC) $^ $(CFLAGS) -o $@

$(BUILD_DIR)/%.cu.o: %.cu
	$(NVCC) $^ $(GENCODES) $(CFLAGS) -o $@

$(BUILD_DIR)/dcn_plugin.so: $(OBJS) $(CUOBJS)
	$(NVCC) $^ $(LFLAGS) -o $@

# Make sure that the build directory exists before compilation
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(OBJS): | $(BUILD_DIR)
$(CUOBJS): | $(BUILD_DIR)

.PHONY: clean
clean:
	rm -f -r build
