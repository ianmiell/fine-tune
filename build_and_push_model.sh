#!/bin/bash

# Set up home folder
export HOME=/workspace

set -eu

MODEL_BASENAME="${MODEL_BASENAME:-qwen2.5-3b-fine-tune-sft}"
FINE_TUNE_BASE_MODEL="${FINE_TUNE_BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
LLAMAFACTORY_TRAIN_MAX_STEPS="${LLAMAFACTORY_TRAIN_MAX_STEPS:-2000}"

HF_REPO="${HF_USERNAME}/${MODEL_BASENAME}"
HF_REPO_GGUF="${HF_USERNAME}/${MODEL_BASENAME}-GGUF"

MODEL_DIR="${HOME}/model"
OUT_DIR="${MODEL_DIR}/out"
GGUF_DIR="${MODEL_DIR}/gguf"
MERGED_DIR="${MODEL_DIR}/merged"
LLAMA_FACTORY_DIR="${HOME}/LLaMA-Factory"
LLAMA_CPP_DIR="${HOME}/llama.cpp"

mkdir -p "$MODEL_DIR"
mkdir -p "$MERGED_DIR"
mkdir -p "$OUT_DIR"
mkdir -p "$GGUF_DIR"

# Go home
cd || exit

# decompress instruction files
GMAIL_JSONL_FILENAME="gmail_sft_instruct.jsonl"
[ -f "${GMAIL_JSONL_FILENAME}" ] || xz -d "${GMAIL_JSONL_FILENAME}.xz"

# apts
apt install build-essential vim -y

# Clones
git clone https://github.com/hiyouga/LLaMA-Factory.git "${LLAMA_FACTORY_DIR}"
git clone https://github.com/ggerganov/llama.cpp.git "${LLAMA_CPP_DIR}"

# Set up data files
mv "${HOME}/${GMAIL_JSONL_FILENAME}" "${LLAMA_FACTORY_DIR}/data"
# Add dataset file
cat > "${LLAMA_FACTORY_DIR}/data/dataset_info.json" <<'JSON'
{ "custom_gmail": { "file_name": "gmail_sft_instruct.jsonl" } }
JSON

# Install pips
cd ${LLAMA_FACTORY_DIR}
pip install -e ".[torch,metrics]" --no-build-isolation
pip uninstall -y torch torchvision torchaudio bitsandbytes || true
pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchaudio
pip install bitsandbytes>=0.39.0

hf auth login --token "$(cat ${HOME}/huggingface_token)"
hf repo create "$HF_REPO" --repo-type model --private || true
hf repo create "$HF_REPO_GGUF" --repo-type model --private || true

llamafactory-cli train \
   --stage sft \
   --do_train true \
   --model_name_or_path "${FINE_TUNE_BASE_MODEL}" \
   --template qwen \
   --dataset custom_gmail \
   --output_dir "${OUT_DIR}" \
   --per_device_train_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --max_steps "${LLAMAFACTORY_TRAIN_MAX_STEPS}" \
   --learning_rate 5e-5 \
   --report_to none \
   --fp16 true \
   --bf16 false \
   --optim adamw_bnb_8bit \
   --gradient_checkpointing true \
   --finetuning_type lora \
   --lora_rank 16 \
   --lora_alpha 32 \
   --lora_dropout 0.05 \
   --lora_target q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj \
   --quantization_bit 4

llamafactory-cli export \
    --model_name_or_path "${FINE_TUNE_BASE_MODEL}" \
    --adapter_name_or_path "${OUT_DIR}" \
    --template qwen \
    --export_dir "${MERGED_DIR}" \
    --max_new_tokens 128 \
    --temperature 0.7

cd /workspace/llama.cpp
cmake -B build
cmake --build build --config Release -j"$(nproc)"

python3 ${LLAMA_CPP_DIR}/convert_hf_to_gguf.py "${MERGED_DIR}" --outfile "${GGUF_DIR}/${MODEL_BASENAME}.f16.gguf" --outtype f16

# Quantize model, ignoring conda libs
/usr/bin/env -i \
  PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin" \
  LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu" \
  "${LLAMA_CPP_DIR}"/build/bin/llama-quantize \
  "${GGUF_DIR}"/${MODEL_BASENAME}.f16.gguf \
  "${GGUF_DIR}"/${MODEL_BASENAME}.Q4_K_M.gguf \
  Q4_K_M

## Run Quantized model (if you want to test)
#"${LLAMA_FACTORY_DIR}"/build/bin/llama-server--model "${MODEL_DIR}/${MODEL_BASENAME}".Q4_K_M.gguf --host 0.0.0.0 --port 8000 --n-gpu-layers 0 --threads 8

# Upload to hf
hf auth login --token "$(cat ${HOME}/huggingface_token)"
i=10
OK=0
while [[ $i -gt 0 ]]; do ((i--)); hf upload "${HF_REPO}" "${OUT_DIR}" && OK=1 && break; done
if [[ $OK != "1" ]]; then echo FAILED && exit; fi
i=10
OK=0
while [[ $i -gt 0 ]]; do ((i--)); hf upload "${HF_REPO_GGUF}" "${GGUF_DIR}" && OK=1 && break; done
if [[ $OK != "1" ]]; then echo FAILED && exit; fi

echo =====================
echo Vast run completed OK
echo =====================
