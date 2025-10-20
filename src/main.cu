#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include "json.hpp"

using json = nlohmann::json;

__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " >>> " << cudaGetErrorString(result) << "\n";
        exit(EXIT_FAILURE);
    }
}

bool tryAlloc(size_t bytes) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err == cudaSuccess) {
        cudaFree(ptr);
        return true;
    }
    cudaGetLastError();
    return false;
}

size_t findMaxAllocatableVRAM(size_t totalMem) {
    size_t low = 0, high = totalMem, best = 0;
    while (low <= high) {
        size_t mid = (low + high) / 2;
        if (tryAlloc(mid)) {
            best = mid;
            low = mid + (16 * 1024 * 1024);
        } else {
            if (mid < 16 * 1024 * 1024) break;
            high = mid - (16 * 1024 * 1024);
        }
    }
    return best;
}

size_t findTotalUsableVRAM(size_t blockSizeMB) {
    size_t blockSize = blockSizeMB * 1024 * 1024;
    std::vector<void*> blocks;
    size_t totalAllocated = 0;

    while (true) {
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, blockSize);
        if (err == cudaSuccess) {
            blocks.push_back(ptr);
            totalAllocated += blockSize;
        } else if (err == cudaErrorMemoryAllocation) {
            cudaGetLastError();
            break;
        } else {
            std::cerr << "Unexpected error during allocation: " << cudaGetErrorString(err) << "\n";
            break;
        }
    }

    for (void* ptr : blocks) cudaFree(ptr);
    return totalAllocated;
}

std::string formatDouble(double val, int precision = 1) {
    std::ostringstream oss;
    oss.imbue(std::locale(""));
    oss << std::fixed << std::setprecision(precision) << val;
    return oss.str();
}

int main() {
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "get device count");
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found.\n";
        return 1;
    }

    std::vector<json> results;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        checkCuda(cudaGetDeviceProperties(&prop, dev), "get device properties");

        std::cerr << "\n=== Device " << dev << " ===\n";
        std::cerr << "Name: " << prop.name << "\n";
        std::cerr << "Total VRAM: " << formatDouble(prop.totalGlobalMem / (1024.0 * 1024.0)) << " MB\n";
        std::cerr << "Compute Capability: " << prop.major << "." << prop.minor << "\n";

        checkCuda(cudaSetDevice(dev), "set device");

        std::cerr << "[Finding max single allocatable chunk...]\n";
        auto t1 = std::chrono::high_resolution_clock::now();
        size_t maxAlloc = findMaxAllocatableVRAM(prop.totalGlobalMem);
        auto t2 = std::chrono::high_resolution_clock::now();
        double allocTime = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::cerr << "Max single alloc: " << formatDouble(maxAlloc / (1024.0 * 1024.0)) << " MB\n";

        std::cerr << "[Finding total usable VRAM using multiple allocations...]\n";
        auto t3 = std::chrono::high_resolution_clock::now();
        size_t totalUsable = findTotalUsableVRAM(128);
        auto t4 = std::chrono::high_resolution_clock::now();
        double totalAllocTime = std::chrono::duration<double, std::milli>(t4 - t3).count();
        std::cerr << "Total usable: " << formatDouble(totalUsable / (1024.0 * 1024.0)) << " MB\n";

        int N = 1 << 24;
        size_t bytes = N * sizeof(float);
        float *A, *B, *C;
        checkCuda(cudaMalloc(&A, bytes), "alloc A");
        checkCuda(cudaMalloc(&B, bytes), "alloc B");
        checkCuda(cudaMalloc(&C, bytes), "alloc C");

        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;

        auto start = std::chrono::high_resolution_clock::now();
        vectorAdd<<<gridSize, blockSize>>>(A, B, C, N);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gbTransferred = 3.0 * bytes / (1024.0 * 1024.0 * 1024.0);
        double bandwidth = gbTransferred / (ms / 1000.0);

        cudaFree(A);
        cudaFree(B);
        cudaFree(C);

        json entry = {
            {"device_id", dev},
            {"name", prop.name},
            {"compute_capability", std::to_string(prop.major) + "." + std::to_string(prop.minor)},
            {"total_vram_mb", prop.totalGlobalMem / (1024.0 * 1024.0)},
            {"max_single_alloc_mb", maxAlloc / (1024.0 * 1024.0)},
            {"alloc_test_time_ms", allocTime},
            {"total_usable_vram_mb", totalUsable / (1024.0 * 1024.0)},
            {"multi_alloc_time_ms", totalAllocTime},
            {"vector_add_time_ms", ms},
            {"bandwidth_gbps", bandwidth}
        };

        results.push_back(entry);
    }

    std::cerr << "\n================ CUDA BENCHMARK SUMMARY ================\n";
    std::cerr << std::left << std::setw(25) << "Device"
              << std::setw(12) << "VRAM(MB)"
              << std::setw(15) << "SingleAlloc"
              << std::setw(15) << "TotalUsable"
              << std::setw(15) << "BW(GB/s)"
              << "\n--------------------------------------------------------\n";

    for (auto& r : results) {
        std::cerr << std::left
                  << std::setw(25) << r["name"].get<std::string>()
                  << std::setw(12) << formatDouble(r["total_vram_mb"].get<double>())
                  << std::setw(15) << formatDouble(r["max_single_alloc_mb"].get<double>())
                  << std::setw(15) << formatDouble(r["total_usable_vram_mb"].get<double>())
                  << std::setw(15) << formatDouble(r["bandwidth_gbps"].get<double>())
                  << "\n";
    }

    std::string file = "cuda_benchmark_results.json";
    std::ifstream ifs(file);
    json existing = json::array();
    if (ifs) { try { ifs >> existing; } catch (...) {} }
    ifs.close();

    for (auto& r : results) existing.push_back(r);
    std::ofstream ofs(file);
    ofs << std::setw(4) << existing;
    ofs.close();

    std::cerr << "\nResults appended to " << file << "\n";
    return 0;
}
