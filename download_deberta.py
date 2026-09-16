"""Download original Microsoft DeBERTa-v3 weights at an immutable revision."""
import argparse
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-size', choices=['small', 'base'], default='small')
    parser.add_argument('--revision', help='Optional model revision; resolved to a commit before downloading')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    repo = f'microsoft/deberta-v3-{args.model_size}'
    revision = args.revision or ('a36c739020e01763fe789b4b85e2df55d6180012'
                                 if args.model_size == 'small' else 'main')
    revision = HfApi().model_info(repo, revision=revision).sha
    print(f'Downloading {repo} at {revision}', flush=True)
    snapshot = snapshot_download(
        repo,
        revision=revision,
        cache_dir=root / 'models/hub',
        allow_patterns=['config.json', 'pytorch_model.bin', 'spm.model', 'tokenizer_config.json'],
    )
    suffix = '' if args.model_size == 'small' else f'_{args.model_size}'
    (root / f'models/deberta{suffix}_snapshot.txt').write_text(str(Path(snapshot).resolve()) + '\n')
    print(snapshot)
