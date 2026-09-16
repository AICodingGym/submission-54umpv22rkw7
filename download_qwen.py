"""Download the official model into the repository's ignored model directory."""
from pathlib import Path
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    snapshot = snapshot_download(
        "Qwen/Qwen3-1.7B",
        cache_dir=Path(__file__).resolve().parent / "models" / "hub",
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
    )
    pointer = Path(__file__).resolve().parent / "models" / "qwen_snapshot.txt"
    pointer.write_text(str(snapshot) + "\n")
    print(snapshot, flush=True)
