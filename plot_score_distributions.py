"""Export score counts and matched per-true-score prediction distributions."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'reports/score_distributions'
MODELS = [
    ('small512', 'small-512 · 原始回归', 'outputs_deberta'),
    ('base512', 'base-512 · 原始回归', 'outputs_deberta_base'),
    ('base2048', 'base-2048 · 原始回归', 'outputs_deberta_base_2048'),
    ('small1024_baseline', 'small-1024 · 首轮回归', 'runs/small1024_pooling_v1/baseline'),
    ('small1024_frozen_mean', 'small-1024 · 冻结 mean', 'runs/small1024_pooling_v1/frozen_mean'),
    ('small1024_frozen_attention', 'small-1024 · 冻结 attention', 'runs/small1024_pooling_v1/frozen_attention'),
    ('small1024_regression_control', 'small-1024 · 本轮回归对照', 'runs/small1024_dualhead_v1/regression'),
    ('small1024_dual01', 'small-1024 · 双头 λ=0.1（回归输出）', 'runs/small1024_dualhead_v1/dual_01'),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, path, pdf=None):
    fig.savefig(path.with_suffix('.png'), dpi=170, facecolor='white', bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'), facecolor='white', bbox_inches='tight')
    if pdf is not None:
        pdf.savefig(fig, facecolor='white')


def qwk(matrix):
    weights = (np.arange(6)[:, None] - np.arange(6)[None, :]) ** 2
    expected = np.outer(matrix.sum(1), matrix.sum(0)) / matrix.sum()
    return float(1 - (weights * matrix).sum() / (weights * expected).sum())


def heatmap(ax, matrix, title, compact=False):
    totals = matrix.sum(1)
    percent = matrix / totals[:, None] * 100
    image = ax.imshow(percent, cmap='Blues', vmin=0, vmax=100, aspect='equal')
    ax.set_title(title, fontsize=10 if compact else 13, pad=12, loc='left')
    ax.set_xticks(range(6), range(1, 7))
    ax.set_yticks(range(6), [f'{k + 1} 分  (n={totals[k]:,})' for k in range(6)])
    ax.set_xlabel('模型预测分数')
    ax.set_ylabel('真实分数')
    ax.tick_params(length=0, labelsize=9 if compact else 11)
    for row in range(6):
        for col in range(6):
            text = f'{percent[row, col]:.1f}%'
            if not compact:
                text += f'\n{matrix[row, col]:,} 篇'
            ax.text(col, row, text, ha='center', va='center',
                    fontsize=8 if compact else 10,
                    color='white' if percent[row, col] >= 55 else '#14324a')
        ax.add_patch(Rectangle((row - .5, row - .5), 1, 1, fill=False,
                               edgecolor='#e19a26', linewidth=1.8))
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def main():
    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
                         'axes.unicode_minus': False, 'font.size': 11, 'axes.titleweight': 'bold',
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(ROOT / 'train.csv', dtype={'essay_id': str})
    splits = pd.read_csv(ROOT / 'splits/deberta_seed42.csv', dtype={'essay_id': str})
    assert source.essay_id.is_unique and splits.essay_id.is_unique
    assert set(source.essay_id) == set(splits.essay_id)
    source = source.merge(splits[['essay_id', 'split']], on='essay_id', validate='one_to_one')
    source = source.set_index('essay_id')
    assert source.score.isin(range(1, 7)).all()
    split_hash = sha256(ROOT / 'splits/deberta_seed42.csv')
    manifest = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
                'split_sha256': split_hash, 'data_sha256': sha256(ROOT / 'train.csv'),
                'primary_evaluation_split': 'selection', 'final_validation_predictions_used': False,
                'normalization': 'Each true-score row sums to 100%; counts also shown in individual figures.',
                'models': [], 'train_counts': {}}
    count_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), layout='constrained')
    fig.suptitle('作文评分数据：各真实分数的数量分布', fontsize=18, fontweight='bold')
    for ax, (key, title, data) in zip(axes, [
        ('all_labeled', '完整标注数据（train.csv）', source),
        ('train', '实际模型训练子集（split = train）', source[source.split == 'train'])]):
        counts = data.score.value_counts().reindex(range(1, 7), fill_value=0).sort_index()
        manifest['train_counts'][key] = {str(k): int(v) for k, v in counts.items()}
        ax.bar(counts.index, counts.values, color=plt.cm.Blues(np.linspace(.4, .85, 6)), width=.65)
        ax.set_title(f'{title}\n共 {len(data):,} 篇', fontsize=13, loc='left')
        for score, count in counts.items():
            ax.text(score, count + counts.max() * .025,
                    f'{count:,}\n{count / len(data):.1%}', ha='center', fontsize=10)
            count_rows.append({'dataset': key, 'true_score': score, 'count': count, 'fraction': count / len(data)})
        ax.set_xticks(range(1, 7))
        ax.set_xlabel('真实分数'); ax.set_ylabel('作文数量')
        ax.set_ylim(0, counts.max() * 1.23)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
    save(fig, OUTPUT / 'training_score_counts')
    fig.savefig(OUTPUT / 'training_score_counts.pdf')
    plt.close(fig)
    pd.DataFrame(count_rows).to_csv(OUTPUT / 'training_score_counts.csv', index=False)

    matrices = {}
    cells, statistics = [], []
    for slug, label, relative in MODELS:
        run = ROOT / relative
        config = json.loads((run / 'config.json').read_text())
        report = json.loads((run / 'report.json').read_text())
        assert (run / 'verification.json').exists() and not config['smoke']
        assert config['split_sha256'] == split_hash
        metadata = {'id': slug, 'label': label, 'run_dir': relative,
                    'selected_epoch': report['selected_epoch'], 'prediction_files': {}}
        for split in ['selection', 'train']:
            path = run / f'{split}_predictions.csv'
            pred = pd.read_csv(path, dtype={'essay_id': str}, float_precision='round_trip')
            ids = source.index[source.split == split]
            assert pred.essay_id.is_unique and set(pred.essay_id) == set(ids)
            assert np.array_equal(pred.score.to_numpy(), source.loc[pred.essay_id, 'score'].to_numpy())
            metadata['prediction_files'][split] = {'rows': len(pred), 'sha256': sha256(path)}
            for version in ['B0', 'B1']:
                assert pred[version].isin(range(1, 7)).all()
                matrix = np.zeros((6, 6), dtype=np.int64)
                np.add.at(matrix, (pred.score.to_numpy(dtype=int) - 1, pred[version].to_numpy(dtype=int) - 1), 1)
                assert matrix.sum() == len(pred)
                expected_counts = source.loc[ids, 'score'].value_counts().reindex(range(1, 7), fill_value=0).to_numpy()
                assert np.array_equal(matrix.sum(1), expected_counts)
                assert np.isclose(qwk(matrix), report['splits'][split][version]['qwk'], atol=1e-12, rtol=0)
                assert np.array_equal(matrix, np.asarray(report['splits'][split][version]['confusion_matrix']))
                matrices[slug, split, version] = matrix
                for truth in range(1, 7):
                    row = matrix[truth - 1]
                    for prediction in range(1, 7):
                        cells.append({'model': slug, 'split': split, 'version': version,
                                      'true_score': truth, 'predicted_score': prediction,
                                      'count': int(row[prediction - 1]), 'row_percent': 100 * row[prediction - 1] / row.sum()})
                    statistics.append({'model': slug, 'split': split, 'version': version, 'true_score': truth,
                                       'rows': int(row.sum()), 'exact_fraction': row[truth - 1] / row.sum(),
                                       'under_fraction': row[:truth - 1].sum() / row.sum(),
                                       'over_fraction': row[truth:].sum() / row.sum(),
                                       'mean_predicted_score': float(row @ np.arange(1, 7) / row.sum()),
                                       'mean_raw_prediction': float(pred.loc[pred.score == truth, 'raw_prediction'].mean())})
        manifest['models'].append(metadata)

    for split, split_label in [('selection', '选择集'), ('train', '训练子集')]:
        folder = OUTPUT / split
        folder.mkdir(exist_ok=True)
        n = int((source.split == split).sum())
        with PdfPages(OUTPUT / f'{split}_model_distributions.pdf') as pdf:
            fig, axes = plt.subplots(4, 2, figsize=(15, 21), layout='constrained')
            fig.suptitle(f'各模型预测分数分布 · B1 校准后 · 同一{split_label} {n:,} 篇\n'
                         '每行按真实分数归一化；橙框表示预测正确；所有图共用 0–100% 色标', fontsize=16)
            for ax, (slug, label, _) in zip(axes.flat, MODELS):
                matrix = matrices[slug, split, 'B1']
                im = heatmap(ax, matrix, f'{label}\nQWK = {qwk(matrix):.5f}', compact=True)
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.5, pad=.025, label='该真实分数内的占比（%）')
            save(fig, folder / 'all_models_B1', pdf)
            plt.close(fig)
            for slug, label, _ in MODELS:
                fig, axes = plt.subplots(1, 2, figsize=(14, 7.5))
                fig.subplots_adjust(left=.10, right=.89, bottom=.13, top=.77, wspace=.38)
                fig.suptitle(f'{label}\n同一{split_label} {n:,} 篇 · 每格：占比 / 作文数量', fontsize=17)
                for ax, version, description in zip(axes, ['B0', 'B1'], ['固定半整数阈值', 'calibration 集拟合阈值']):
                    matrix = matrices[slug, split, version]
                    im = heatmap(ax, matrix, f'{version} · {description}\nQWK = {qwk(matrix):.5f}')
                color_axis = fig.add_axes([.93, .18, .018, .56])
                fig.colorbar(im, cax=color_axis, label='该真实分数内的占比（%）')
                save(fig, folder / slug, pdf)
                plt.close(fig)
    pd.DataFrame(cells).to_csv(OUTPUT / 'prediction_distribution_cells.csv', index=False)
    pd.DataFrame(statistics).to_csv(OUTPUT / 'per_grade_statistics.csv', index=False)
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    lines = ['# 作文分数与模型预测分布', '',
             '主比较使用相同的 1,557 篇 selection 作文；train 图仅用于检查拟合情况，不能代表泛化。', '',
             'B0 使用固定半整数阈值，B1 使用各自 calibration 集拟合的阈值。双头模型展示回归头输出。', '',
             '热图每行是一个真实分数，横轴为预测分数。每行合计 100%，单模型图同时标注篇数。橙框是正确预测。', '',
             '覆盖 8 个已完成的 DeBERTa 相关模型。λ=0.3 尚未完成，因此未纳入；不包含历史 RLT/融合实验。', '',
             '## 训练数据', '', '![数量分布](training_score_counts.png)', '',
             '## 选择集汇总', '', '![B1 汇总](selection/all_models_B1.png)', '',
             '[选择集完整 PDF](selection_model_distributions.pdf) · [训练子集完整 PDF](train_model_distributions.pdf)', '',
             '## 每个模型（左 B0，右 B1）', '']
    for slug, label, _ in MODELS:
        lines += [f'### {label}', '', f'![{label}](selection/{slug}.png)', '']
    lines += ['原始计数与比例见 `prediction_distribution_cells.csv`；按等级的偏低/偏高比例见 `per_grade_statistics.csv`。',
              '来源、检查点轮次和输入哈希见 `manifest.json`。每个 PNG 同目录均有 SVG 矢量版本。', '']
    (OUTPUT / 'README.md').write_text('\n'.join(lines))
    print(json.dumps({'output': str(OUTPUT), 'models': len(MODELS), 'counts': manifest['train_counts'],
                      'all_prediction_ids_labels_confusion_matrices_and_qwk_verified': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
