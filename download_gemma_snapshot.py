from huggingface_hub import snapshot_download

# Download the full model repository (all files) to the specified local folder
snapshot_download(
    repo_id="google/gemma-3n-E2B-it",
    local_dir="./gemma-3n-E2B-it",
    local_dir_use_symlinks=False
)
print("Download complete.")
