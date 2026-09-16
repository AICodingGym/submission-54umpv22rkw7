"""Export only an audited supervised baseline's encoder for explicit warm starts."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from deberta_baseline import EssayRegressor, ROOT, write_json
from verify_deberta_run import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    source, output = args.source_run.resolve(), args.output_dir.resolve()
    if output.exists():
        raise ValueError(f'Refusing to overwrite {output}')
    report = json.loads((source / 'report.json').read_text())
    config = json.loads((source / 'config.json').read_text())
    split_metadata = json.loads((ROOT / 'splits/deberta_seed42.json').read_text())
    audit = json.loads((source / 'verification.json').read_text())
    digest = sha256(source / 'model/model.pt')
    if (audit['model_sha256'] != digest or audit['max_reload_raw_difference'] != 0
            or not audit['all_integer_predictions_match']
            or not audit['all_report_metrics_recomputed']):
        raise ValueError('Source checkpoint must match a successful baseline audit')
    if (report['final_validation_evaluated']
            or config['split_sha256'] != sha256(ROOT / 'splits/deberta_seed42.csv')
            or split_metadata['data_sha256'] != sha256(ROOT / 'train.csv')):
        raise ValueError('Expected baseline trained on the current unchanged split')
    model = EssayRegressor.load(source / 'model', device='cpu')
    tokenizer = AutoTokenizer.from_pretrained(source / 'model', local_files_only=True)
    model.encoder.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    restored = AutoModel.from_pretrained(output, local_files_only=True)
    reference, actual = model.encoder.state_dict(), restored.state_dict()
    if reference.keys() != actual.keys() or any(not torch.equal(value, actual[key])
                                              for key, value in reference.items()):
        raise ValueError('Exported encoder failed exact tensor verification')
    restored_tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
    if (tokenizer.get_vocab() != restored_tokenizer.get_vocab()
            or tokenizer.all_special_ids != restored_tokenizer.all_special_ids):
        raise ValueError('Exported tokenizer mismatch')
    provenance = {'source_run': str(source), 'source_model_sha256': digest,
                  'source_selected_epoch': report['selected_epoch'],
                  'source_max_length': config['max_length'],
                  'split_sha256': config['split_sha256'], 'data_sha256': split_metadata['data_sha256'],
                  'encoder_tensors_verified': len(reference), 'tokenizer_vocab_verified': True,
                  'encoder_weights_sha256': sha256(output / 'model.safetensors'),
                  'initialization': 'Supervised train-split encoder; checkpoint selected on selection',
                  'baseline_score_head_exported': False, 'final_validation_evaluated': False}
    write_json(output / 'supervised_source.json', provenance)
    print(json.dumps(provenance), flush=True)


if __name__ == '__main__':
    main()
