APP_NAME := cuda_benchmark
SRC := src/main.cu
NVCC := nvcc
NVCC_FLAGS := -O3 -std=c++17
JSON_URL := https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp
JSON_FILE := src/json.hpp

all: build

$(JSON_FILE):
	curl -L -o $(JSON_FILE) $(JSON_URL)

build: $(JSON_FILE)
	@echo "Building $(APP_NAME)..."
	$(NVCC) $(SRC) -o $(APP_NAME) $(NVCC_FLAGS)

debug: $(JSON_FILE)
	@echo "Building debug version..."
	$(NVCC) $(SRC) -o $(APP_NAME)_debug -G -O0 -std=c++17

clean:
	@echo "Cleaning up..."
	rm -f $(APP_NAME) $(APP_NAME)_debug cuda_benchmark_results.json benchmark.log

run: build
	@echo "Running benchmark..."
	./$(APP_NAME) 2> benchmark.log
	@echo "Log saved to benchmark.log"

docker/build:
	@echo "Building Docker image..."
	docker buildx build -t cuda-benchmark --load .

docker/run:
	@echo "Running benchmark inside Docker..."
	docker run --rm --gpus all cuda-benchmark

.PHONY: all build debug clean run docker/build docker/run
