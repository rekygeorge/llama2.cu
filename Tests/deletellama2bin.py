import modal

app = modal.App("delete-large-file")

# Correct way to lookup a volume in newer Modal API
volume = modal.Volume.from_name("huggingface-cache", create_if_missing=False)

@app.function(volumes={"/cache": volume})
def delete_file():
    import os

    path = "/cache/llama2_7b.bin"  # The actual filename we've been using

    if os.path.exists(path):
        os.remove(path)
        print(f"✅ Deleted: {path}")
    else:
        print(f"❌ File not found: {path}")
        
    # Also check what files exist in the cache directory
    print("\n📁 Current files in /cache:")
    try:
        files = os.listdir("/cache")
        for file in files:
            file_path = os.path.join("/cache", file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path) / (1024**3)  # Size in GB
                print(f"  - {file} ({size:.1f} GB)")
        if not files:
            print("  (empty)")
    except Exception as e:
        print(f"  Error listing files: {e}")

if __name__ == "__main__":
    with app.run():
        delete_file.remote()
