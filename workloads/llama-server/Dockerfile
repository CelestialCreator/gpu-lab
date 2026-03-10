FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# Build llama.cpp with CUDA
RUN apt-get update && apt-get install -y git cmake build-essential python3 python3-pip curl && \
    rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/ggml-org/llama.cpp /opt/llama.cpp
WORKDIR /opt/llama.cpp
RUN cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON && \
    cmake --build build --config Release -j$(nproc)

EXPOSE 8080

ENTRYPOINT ["/opt/llama.cpp/build/bin/llama-server"]
