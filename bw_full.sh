#!/bin/bash
# Self-contained RTX PRO 6000 Blackwell (sm_120) DFlash verification. Runs inside the pod.
set -e
echo "=== GPU ==="; nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq cmake build-essential git python3-pip >/dev/null 2>&1 || true

cd /workspace 2>/dev/null || cd /root
[ -d llama.cpp ] || git clone -q -b work-qwen35-dflash https://github.com/AlexWortega/llama.cpp.git
cd llama.cpp
pip install -q numpy sentencepiece transformers safetensors gguf protobuf hf_transfer 2>/dev/null || true

mkdir -p models hf
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "=== download HF models ==="
python3 -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3.5-4B', local_dir='hf/tgt'); s('z-lab/Qwen3.5-4B-DFlash', local_dir='hf/dft')" 2>&1 | tail -1

echo "=== convert ==="
python3 convert_hf_to_gguf.py hf/tgt --outfile models/tgt-f16.gguf --outtype f16 >/tmp/cv1.log 2>&1 && echo tgt-ok || { echo TGT_FAIL; tail -8 /tmp/cv1.log; exit 1; }
python3 convert_hf_to_gguf.py hf/dft --outfile models/Qwen3.5-4B-DFlash-f16.gguf --outtype f16 >/tmp/cv2.log 2>&1 && echo dft-ok || { echo DFT_FAIL; tail -8 /tmp/cv2.log; exit 1; }

echo "=== build (Blackwell sm_120) ==="
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF >/tmp/cm.log 2>&1
cmake --build build --target llama-speculative-simple llama-quantize llama-cli -j $(nproc) >/tmp/build.log 2>&1 && echo BUILT || { echo BUILDFAIL; tail -20 /tmp/build.log; exit 1; }

./build/bin/llama-quantize models/tgt-f16.gguf models/Qwen3.5-4B-Q8_0.gguf Q8_0 >/dev/null 2>&1 && echo quantized
M="-m models/Qwen3.5-4B-Q8_0.gguf -md models/Qwen3.5-4B-DFlash-f16.gguf --dflash -ngl 99 -ngld 99 -p Tell-me-about-the-water-cycle-in-detail. -n 200 -c 2048 --draft-max 5 --temp 0 --top-k 1 --samplers top_k"
BIN=./build/bin/llama-speculative-simple

echo "=== AR baseline ==="
./build/bin/llama-cli -m models/Qwen3.5-4B-Q8_0.gguf -ngl 99 -p "Tell me about the water cycle in detail." -n 200 -c 2048 --temp 0 -no-cnv 2>/tmp/ar.err >/dev/null || true
tr "\r" "\n" < /tmp/ar.err | grep -oE "[0-9.]+ tokens per second" | tail -1

echo "=== DFlash full stack (trace+gpuverify+async) ==="
LLAMA_SPEC_TRACE=1 LLAMA_SPEC_GPU_VERIFY=1 LLAMA_SPEC_ASYNC=1 $BIN $M >/tmp/df.txt 2>/tmp/df.err || true
tr "\r" "\n" < /tmp/df.err | grep -oE "speed: +[0-9.]+|accept += +[0-9.]+%" | tail -2

echo "=== DFlash + Blackwell CUDA graphs (sm_120 >= Ampere: engage by default) ==="
LLAMA_SPEC_TRACE=1 LLAMA_SPEC_GPU_VERIFY=1 $BIN $M >/tmp/dfg.txt 2>/tmp/dfg.err || true
tr "\r" "\n" < /tmp/dfg.err | grep -oE "speed: +[0-9.]+" | tail -1

echo "=== sample ==="; tail -c 160 /tmp/df.txt
echo "=== DONE ==="
