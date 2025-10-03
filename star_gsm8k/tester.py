from huggingface_hub import snapshot_download
model_id = "meta-llama/Llama-3.2-3B-Instruct"
dst = "/home/hchaura1/hf_models/llama3.2-3b"   # writable path in your home
print("Downloading to", dst)
try:
    path = snapshot_download(repo_id=model_id, cache_dir=dst, use_auth_token=True)
    print("Downloaded to:", path)
except Exception as e:
    print("Download failed:", repr(e))