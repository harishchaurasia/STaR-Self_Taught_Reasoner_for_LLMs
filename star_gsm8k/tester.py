# from huggingface_hub import snapshot_download
# model_id = "meta-llama/Llama-3.2-3B-Instruct"
# dst = "/home/hchaura1/hf_models/llama3.2-3b"
# print("Downloading to", dst, "… this may take a while (many GBs)")
# path = snapshot_download(repo_id=model_id, cache_dir=dst, use_auth_token=True, resume_download=True)
# print("Download finished ->", path)


# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch, os, sys

# local = "/home/hchaura1/hf_models/llama3.2-3b"
# print("PATH:", local)
# # Ensure we load from the local snapshot (transformers will detect the snapshot layout)
# tok = AutoTokenizer.from_pretrained(local)
# print("Tokenizer OK — vocab size:", tok.vocab_size if hasattr(tok, "vocab_size") else "n/a")

# # Use device_map="auto" to place weights onto GPU(s). If OOM, see note below.
# print("Loading model (this may use GPU and take ~10-60s)...")
# model = AutoModelForCausalLM.from_pretrained(local, device_map="auto", torch_dtype="auto")
# dev = next(model.parameters()).device
# print("Model loaded. Parameter device:", dev)

# prompt = "Q: What is 2 + 3? A:"
# inputs = tok(prompt, return_tensors="pt").to(dev)
# out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
# print("Generated:", tok.decode(out[0], skip_special_tokens=True))

from huggingface_hub import whoami
print("whoami ->", whoami())