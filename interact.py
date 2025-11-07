#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- CLI args ---
import argparse

parser = argparse.ArgumentParser(description="Interact with a fine-tuned Qwen model")
parser.add_argument(
  "--base",
  default="Qwen/Qwen2.5-3B-Instruct",
  help="Base model identifier to load",
)
parser.add_argument(
  "--adapter",
  default="ianmiell/qwen2.5-3b-iansft",
  help="Adapter/LoRA identifier to load",
)
parser.add_argument(
  "--compare-base",
  action="store_true",
  help="Also run the base model for side-by-side responses",
)
args = parser.parse_args()

# --- Model setup ---
base = args.base
adapter = args.adapter

dtype = torch.float32  # safe for CPU

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(base, use_fast=True)

print("Loading base model (" + base + ")...")
base_model = AutoModelForCausalLM.from_pretrained(
  base,
  torch_dtype=dtype,
  device_map="auto",
)
base_model.eval()

base_compare_model = None
if args.compare_base:
  print("Loading base model for comparison (" + base + ")...")
  base_compare_model = AutoModelForCausalLM.from_pretrained(
    base,
    torch_dtype=dtype,
    device_map="auto",
  )
  base_compare_model.eval()

print("Loading adapter (" + adapter + ")...")
model = PeftModel.from_pretrained(base_model, adapter)
model.eval()

# --- Interactive loop ---
def generate_response(model, tok, prompt):
  inputs = tok(prompt, return_tensors="pt")
  with torch.no_grad():
    out = model.generate(
      **inputs,
      max_new_tokens=40,
      do_sample=True,
      temperature=0.8,
    )
  return tok.decode(out[0], skip_special_tokens=True)


def chat(model, tok, base_compare_model=None):
  print("\nReady. Ctrl-C to exit.\n")
  while True:
    try:
      prompt = input("> ")
    except KeyboardInterrupt:
      print("\nExiting.")
      break
    if base_compare_model is not None:
      base_response = generate_response(base_compare_model, tok, prompt)
      adapter_response = generate_response(model, tok, prompt)
      print("[BASE]    " + base_response)
      print("[ADAPTER] " + adapter_response)
    else:
      response = generate_response(model, tok, prompt)
      print(response)


chat(model, tok, base_compare_model)
