from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="distilgpt2",
    local_dir="./distilgpt2",
    local_dir_use_symlinks=False  # optional, will ignore in recent versions
)
print("Download complete.")
