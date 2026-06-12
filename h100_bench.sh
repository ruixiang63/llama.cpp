#!/bin/bash
# Runs INSIDE the H100 pod. Expects: /work/llama.cpp (patched tree), /work/models/*.gguf
set -e
cd /work/llama.cpp
echo "=== GPU ==="; nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cmake build-essential libcurl4-openssl-dev python3 >/dev/null 2>&1 || true

# Hopper = sm_90; build CUDA arch 90
cmake -B build-h100 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DLLAMA_CURL=ON >/dev/null 2>&1
cmake --build build-h100 --target llama-speculative-simple -j $(nproc) >/dev/null 2>&1 && echo BUILT || { echo BUILDFAIL; exit 1; }
BIN=./build-h100/bin/llama-speculative-simple
A="-m /work/models/Qwen3.5-4B-Q8_0.gguf -md /work/models/Qwen3.5-4B-DFlash-f16.gguf --dflash -ngl 99 -ngld 99 -p Tell-me-about-the-water-cycle-in-detail. -n 200 -c 2048 --draft-max 5 --temp 0 --top-k 1 --samplers top_k"

echo "=== AR baseline ==="
./build-h100/bin/llama-cli -m /work/models/Qwen3.5-4B-Q8_0.gguf -ngl 99 -p "Tell me about the water cycle in detail." -n 200 -c 2048 --temp 0 -no-cnv 2>/tmp/ar.err >/dev/null || true
tr "\r" "\n" < /tmp/ar.err | grep -oE "eval time.*per token|[0-9.]+ tokens per second" | tail -2

echo "=== DFlash full stack (trace+gpuverify+async) ==="
LLAMA_SPEC_TRACE=1 LLAMA_SPEC_GPU_VERIFY=1 LLAMA_SPEC_ASYNC=1 $BIN $A >/tmp/df.txt 2>/tmp/df.err
tr "\r" "\n" < /tmp/df.err | grep -oE "speed: +[0-9.]+|accept += +[0-9.]+%" | tail -2

echo "=== DFlash + CUDA graphs (Hopper: NOT arch-gated, should engage by default) ==="
LLAMA_SPEC_TRACE=1 LLAMA_SPEC_GPU_VERIFY=1 $BIN $A >/tmp/dfg.txt 2>/tmp/dfg.err
tr "\r" "\n" < /tmp/dfg.err | grep -oE "speed: +[0-9.]+" | tail -1

echo "=== lossless gate (vs trace-off control) ==="
$BIN $A >/tmp/ctl.txt 2>/dev/null || true
diff -q /tmp/df.txt /tmp/df.txt >/dev/null && echo "df coherent"
tail -c 140 /tmp/df.txt
echo "=== DONE ==="
