"""Download the original, pinned Qwen3-4B model for local GEPA."""
from pathlib import Path
from huggingface_hub import snapshot_download

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    snapshot = snapshot_download('Qwen/Qwen3-4B',
        revision='1cfa9a7208912126459214e8b04321603b3df60c', cache_dir=root/'models/hub',
        allow_patterns=['*.json','*.safetensors','*.txt','*.jinja'])
    (root/'models/qwen4b_snapshot.txt').write_text(str(Path(snapshot).resolve())+'\n')
