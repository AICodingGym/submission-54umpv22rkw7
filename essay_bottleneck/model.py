"""Learned latent readout; optional reconstruction of a fixed teacher's features.

This borrows RLT's representation bottleneck, not its reinforcement learning.
The reconstruction decoder sees only latents and position queries, never input
tokens or encoder features. Teacher features are targets only.
"""
import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel


@dataclass
class BottleneckConfig:
    pooling: str = 'latent'
    num_latents: int = 4
    latent_dim: int = 256
    num_heads: int = 4
    dropout: float = 0.1
    reconstruction_weight: float = 0.1
    finetune_encoder: bool = False
    normalize_queries: bool = False
    score_objective: str = 'mse'

    def validate(self):
        if self.pooling not in {'latent', 'mean'}:
            raise ValueError('pooling must be latent or mean')
        if self.score_objective not in {'mse', 'ordinal_bce'}:
            raise ValueError('score_objective must be mse or ordinal_bce')
        if min(self.num_latents, self.latent_dim, self.num_heads) < 1:
            raise ValueError('latent counts and dimensions must be positive')
        if self.latent_dim % self.num_heads or self.latent_dim % 2:
            raise ValueError('latent_dim must be even and divisible by num_heads')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        if not math.isfinite(self.reconstruction_weight) or self.reconstruction_weight < 0:
            raise ValueError('reconstruction_weight must be finite and nonnegative')
        if self.pooling == 'mean' and self.reconstruction_weight:
            raise ValueError('mean pooling is a score-only control; set reconstruction_weight=0')
        if self.pooling == 'mean' and self.normalize_queries:
            raise ValueError('query normalization requires latent pooling')


def positions(length, dimension, device):
    """Deterministic sinusoidal positions, without a learned length limit."""
    pos = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    frequency = torch.exp(torch.arange(0, dimension, 2, device=device).float()
                          * (-math.log(10000.0) / dimension))
    result = torch.empty(length, dimension, device=device)
    result[:, 0::2] = torch.sin(pos * frequency)
    result[:, 1::2] = torch.cos(pos * frequency)
    return result


class LatentReadout(nn.Module):
    def __init__(self, hidden_size, config):
        super().__init__()
        dim = config.latent_dim
        self.projection = nn.Linear(hidden_size, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.randn(config.num_latents, dim) * 0.02)
        self.query_norm = nn.LayerNorm(dim) if config.normalize_queries else nn.Identity()
        self.attention = nn.MultiheadAttention(dim, config.num_heads,
                                               dropout=config.dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                 nn.Dropout(config.dropout), nn.Linear(dim * 4, dim))
        self.norm2 = nn.LayerNorm(dim)

    def query_vectors(self, batch_size):
        return self.query_norm(self.queries)[None].expand(batch_size, -1, -1)

    def forward(self, hidden, mask):
        memory = self.projection(hidden)
        memory = self.input_norm(memory + positions(hidden.shape[1], memory.shape[-1], hidden.device))
        queries = self.query_vectors(hidden.shape[0])
        readout, _ = self.attention(queries, memory, memory,
                                    key_padding_mask=~mask.bool(), need_weights=False)
        latents = self.norm1(queries + readout)
        return self.norm2(latents + self.mlp(latents))


class ReconstructionDecoder(nn.Module):
    def __init__(self, hidden_size, config):
        super().__init__()
        self.dim = config.latent_dim
        self.attention = nn.MultiheadAttention(self.dim, config.num_heads,
                                               dropout=config.dropout, batch_first=True)
        self.norm = nn.LayerNorm(self.dim)
        self.output = nn.Sequential(nn.Linear(self.dim, self.dim * 2), nn.GELU(),
                                    nn.Linear(self.dim * 2, hidden_size))

    def forward(self, latents, length):
        queries = positions(length, self.dim, latents.device)[None].expand(latents.shape[0], -1, -1)
        values, _ = self.attention(queries, latents, latents, need_weights=False)
        return self.output(self.norm(queries + values))


class BottleneckRegressor(nn.Module):
    def __init__(self, encoder, config, *, for_training=True):
        super().__init__()
        config.validate()
        self.config = config
        self.encoder = encoder
        self.teacher = None
        if not config.finetune_encoder:
            self.encoder.requires_grad_(False)
        elif config.reconstruction_weight and for_training:
            # Keep targets stationary even as the student's encoder is updated.
            self.teacher = copy.deepcopy(encoder).requires_grad_(False).eval()
        hidden = encoder.config.hidden_size
        if config.pooling == 'latent':
            self.readout = LatentReadout(hidden, config)
            score_dim = config.num_latents * config.latent_dim
        else:
            self.readout = nn.Identity()
            score_dim = hidden
        self.head = nn.Sequential(nn.LayerNorm(score_dim), nn.Linear(score_dim, config.latent_dim),
                                  nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.latent_dim, 1))
        # Four positive gaps define five ordered cutpoints; centering avoids a redundant intercept.
        self.ordinal_gaps = (nn.Parameter(torch.full((4,), math.log(math.expm1(1.))))
                             if config.score_objective == 'ordinal_bce' else None)
        self.decoder = (ReconstructionDecoder(hidden, config)
                        if config.reconstruction_weight else None)
        self.train(for_training)

    def ordinal_cutpoints(self):
        if self.ordinal_gaps is None:
            raise ValueError('Ordinal cutpoints require ordinal_bce')
        gaps = F.softplus(self.ordinal_gaps.float()) + 1e-4
        points = torch.cat((gaps.new_zeros(1), gaps.cumsum(0)))
        return points - points.mean()

    def train(self, mode=True):
        super().train(mode)
        if not self.config.finetune_encoder:
            self.encoder.eval()
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def forward(self, input_ids, attention_mask, labels=None, encoded_hidden=None,
                score_weights=None, **encoder_inputs):
        if score_weights is not None and labels is None:
            raise ValueError('Score weights require labels')
        if attention_mask.ndim != 2 or not attention_mask.bool().any(dim=1).all():
            raise ValueError('Each essay needs at least one unmasked token')
        encoder_config = self.encoder.config
        if (encoder_config.model_type == 'deberta-v2'
                and getattr(encoder_config, 'position_biased_input', True)
                and input_ids.shape[1] > encoder_config.max_position_embeddings):
            raise ValueError('This encoder uses a fixed absolute position table; choose an explicit '
                             'max length or an encoder supporting longer input')
        inputs = dict(input_ids=input_ids, attention_mask=attention_mask, **encoder_inputs)
        if encoded_hidden is not None:
            if self.config.finetune_encoder:
                raise ValueError('Cached features cannot bypass a trainable encoder')
            if (encoded_hidden.shape != (*input_ids.shape, encoder_config.hidden_size)
                    or encoded_hidden.device != input_ids.device or encoded_hidden.requires_grad
                    or not encoded_hidden.is_floating_point()):
                raise ValueError('Invalid frozen encoder feature batch')
            hidden = encoded_hidden.float()
        elif self.config.finetune_encoder:
            hidden = self.encoder(**inputs).last_hidden_state.float()
        else:
            with torch.no_grad():
                hidden = self.encoder(**inputs).last_hidden_state.float()
        if self.config.pooling == 'latent':
            latents = self.readout(hidden, attention_mask)
            representation = latents.flatten(1)
        else:
            mask = attention_mask.unsqueeze(-1).float()
            representation = (hidden * mask).sum(1) / mask.sum(1)
        head_scores = self.head(representation).squeeze(-1).float()
        ordinal_logits = None
        if self.config.score_objective == 'ordinal_bce':
            ordinal_logits = head_scores[:, None] - self.ordinal_cutpoints()[None]
            scores = 1 + ordinal_logits.sigmoid().sum(-1)
        else:
            scores = head_scores
        result = {'scores': scores}
        if ordinal_logits is not None:
            result['ordinal_logits'] = ordinal_logits
        if labels is None:
            return result  # Inference never invokes the teacher or decoder.
        if labels.shape != scores.shape:
            raise ValueError('Expected one label per essay')
        if score_weights is not None:
            if (score_weights.shape != scores.shape or score_weights.device != scores.device
                    or not torch.isfinite(score_weights).all() or (score_weights <= 0).any()):
                raise ValueError('Expected one finite positive score weight per essay')
        if ordinal_logits is not None:
            if (not torch.isfinite(labels).all() or (labels < 1).any() or (labels > 6).any()
                    or (labels != labels.round()).any()):
                raise ValueError('Ordinal scoring requires integer grades in 1..6')
            targets = (labels[:, None] > torch.arange(1, 6, device=labels.device)[None]).float()
            per_essay = F.binary_cross_entropy_with_logits(ordinal_logits, targets,
                                                           reduction='none').mean(-1)
            score_loss = (per_essay if score_weights is None else per_essay * score_weights.float()).mean()
        elif score_weights is None:
            score_loss = F.mse_loss(scores, labels.float())
        else:
            # Global train-average normalization happens before batching, not inside each microbatch.
            score_loss = (score_weights.float() * (scores - labels.float()).square()).mean()
        reconstruction_loss = scores.new_zeros(())
        if self.decoder is not None:
            if self.config.finetune_encoder:
                if self.teacher is None:
                    raise ValueError('Inference checkpoint has no training teacher')
                with torch.no_grad():
                    target = self.teacher(**inputs).last_hidden_state.float()
            else:
                target = hidden.detach()
            # Normalize frozen targets per token to stabilize the auxiliary scale.
            target = F.layer_norm(target.detach(), (target.shape[-1],))
            recovered = self.decoder(latents, hidden.shape[1]).float()
            per_token = (recovered - target).square().mean(-1)
            # Equal weight per essay, excluding padding at both pooling and loss.
            mask = attention_mask.float()
            reconstruction_loss = ((per_token * mask).sum(1) / mask.sum(1)).mean()
        result.update(score_loss=score_loss, reconstruction_loss=reconstruction_loss,
                      loss=score_loss + self.config.reconstruction_weight * reconstruction_loss)
        return result

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.encoder.config.save_pretrained(directory)
        (directory / 'bottleneck.json').write_text(json.dumps(asdict(self.config), indent=2) + '\n')
        # Deployment checkpoint: student, readout, score head and optional decoder.
        # Teacher is deliberately excluded; this is not a resumable training state.
        torch.save({k: v for k, v in self.state_dict().items() if not k.startswith('teacher.')},
                   directory / 'model.pt')

    @classmethod
    def load(cls, directory, device='cpu'):
        directory = Path(directory)
        config = BottleneckConfig(**json.loads((directory / 'bottleneck.json').read_text()))
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(directory, local_files_only=True))
        model = cls(encoder, config, for_training=False)
        model.load_state_dict(torch.load(directory / 'model.pt', map_location='cpu', weights_only=True))
        return model.to(device).eval()
