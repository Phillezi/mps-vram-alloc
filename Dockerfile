FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p ./src && curl -L -o ./src/json.hpp https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp

COPY src/main.cu ./src/main.cu

RUN nvcc src/main.cu -o cuda_benchmark -O3 -std=c++17 \
    -Xcompiler="-static-libgcc -static-libstdc++" \
    -lcudart_static -ldl -lrt

FROM debian:stable-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libc6 libgcc-s1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/cuda_benchmark .

ENTRYPOINT ["./cuda_benchmark"]
