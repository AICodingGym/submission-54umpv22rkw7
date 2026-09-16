"""Download the pinned original Microsoft DeBERTa-v3-small checkpoint."""
from pathlib import Path

from huggingface_hub import snapshot_download


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    snapshot = snapshot_download(
        'microsoft/deberta-v3-small',
        revision='a36c739020e01763fe789b4b85e2df55d6180012',
        cache_dir=root / 'models/hub',
        allow_patterns=['config.json', 'pytorch_model.bin', 'spm.model', 'tokenizer_config.json'],
    )
    (root / 'models/deberta_snapshot.txt').write_text(str(Path(snapshot).resolve()) + '\n')
    print(snapshot)
