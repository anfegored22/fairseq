# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from dataclasses import dataclass
from functools import partial
from omegaconf import MISSING, II
from typing import Optional, Callable
from fairseq.data.data_utils import compute_mask_indices
from fairseq.modules import GradMultiply
from fairseq.utils import index_put
from examples.data2vec.data.modality import Modality
from .modules import D2vDecoderConfig

logger = logging.getLogger(__name__)


@dataclass
class D2vModalityConfig:
    type: Modality = MISSING
    prenet_depth: int = 4
    prenet_layerdrop: float = 0
    prenet_dropout: float = 0
    start_drop_path_rate: float = 0
    end_drop_path_rate: float = 0

    num_extra_tokens: int = 0
    init_extra_token_zero: bool = True

    mask_noise_std: float = 0.01
    mask_prob_min: Optional[float] = None
    mask_prob: float = 0.7
    mask_prob_start: Optional[float] = None
    mask_prob_end: Optional[float] = None
    mask_prob_warmup_updates: int = 0
    mask_prob_schedule: str = "linear"
    inverse_mask: bool = False
    mask_prob_adjust: float = 0
    keep_masked_pct: float = 0
    mask_loss_bins: int = 0
    mask_loss_ema_decay: float = 0.99
    mask_loss_warmup_updates: int = 0
    mask_loss_easy_to_hard_updates: int = 0
    mask_loss_explore_pct: float = 0.1

    mask_length: int = 5
    mask_length_start: Optional[int] = None
    mask_length_end: Optional[int] = None
    mask_length_warmup_updates: int = 0
    add_masks: bool = False
    remove_masks: bool = False
    mask_dropout: float = 0.0
    encoder_zero_mask: bool = True

    mask_channel_prob: float = 0.0
    mask_channel_length: int = 64

    ema_local_encoder: bool = False  # used in data2vec_multi
    local_grad_mult: float = 1.0

    use_alibi_encoder: bool = False
    alibi_scale: float = 1.0
    learned_alibi: bool = False
    alibi_max_pos: Optional[int] = None
    learned_alibi_scale: bool = False
    learned_alibi_scale_per_head: bool = False
    learned_alibi_scale_per_layer: bool = False

    num_alibi_heads: int = II("model.num_heads")
    model_depth: int = II("model.depth")

    decoder: Optional[D2vDecoderConfig] = D2vDecoderConfig()


MaskSeed = namedtuple("MaskSeed", ["seed", "update", "ids"])
MaskInfo = namedtuple("MaskInfo", ["x_unmasked", "mask", "ids_restore", "ids_keep"])


class ModalitySpecificEncoder(nn.Module):
    def __init__(
        self,
        modality_cfg: D2vModalityConfig,
        embed_dim: int,
        local_encoder: nn.Module,
        project_features: nn.Module,
        fixed_positional_encoder: Optional[nn.Module],
        relative_positional_encoder: Optional[nn.Module],
        context_encoder: nn.Module,
        decoder: nn.Module,
        get_alibi_bias: Optional[Callable[[int, int, str, str], torch.Tensor]],
    ):
        super().__init__()

        self.modality_cfg = modality_cfg
        self.local_encoder = local_encoder
        self.project_features = project_features
        self.fixed_positional_encoder = fixed_positional_encoder
        self.relative_positional_encoder = relative_positional_encoder
        self.context_encoder = context_encoder

        self.decoder = decoder
        self.get_alibi_bias = get_alibi_bias if modality_cfg.use_alibi_encoder else None

        self.local_grad_mult = self.modality_cfg.local_grad_mult

        self.extra_tokens = None
        if modality_cfg.num_extra_tokens > 0:
            self.extra_tokens = nn.Parameter(
                torch.zeros(1, modality_cfg.num_extra_tokens, embed_dim)
            )
            if not modality_cfg.init_extra_token_zero:
                nn.init.normal_(self.extra_tokens)
            elif self.extra_tokens.size(1) > 1:
                nn.init.normal_(self.extra_tokens[:, 1:])

        self.alibi_scale = None
        if self.get_alibi_bias is not None:
            self.alibi_scale = nn.Parameter(
                torch.full(
                    (
                        (modality_cfg.prenet_depth + modality_cfg.model_depth)
                        if modality_cfg.learned_alibi_scale_per_layer
                        else 1,
                        1,
                        self.modality_cfg.num_alibi_heads
                        if modality_cfg.learned_alibi_scale_per_head
                        else 1,
                        1,
                        1,
                    ),
                    modality_cfg.alibi_scale,
                    dtype=torch.float,
                ),
                requires_grad=modality_cfg.learned_alibi_scale,
            )

        if modality_cfg.learned_alibi and self.get_alibi_bias is not None:
            assert modality_cfg.alibi_max_pos is not None
            alibi_bias = self.get_alibi_bias(
                batch_size=1,
                time_steps=modality_cfg.alibi_max_pos,
                heads=modality_cfg.num_alibi_heads,
                scale=1.0,
                dtype=torch.float,
                device="cpu",
            )
            self.alibi_bias = nn.Parameter(alibi_bias)
            self.get_alibi_bias = partial(
                _learned_alibi_bias, alibi_bias=self.alibi_bias
            )

        self.current_mask_prob = self.modality_cfg.mask_prob
        self.current_mask_length = self.modality_cfg.mask_length
        self.current_num_updates = 0

        self.mask_loss_bins_ready = False
        if self._mask_loss_curriculum_enabled():
            bins = self.modality_cfg.mask_loss_bins
            self.register_buffer(
                "mask_loss_bin_ema",
                torch.zeros(bins, dtype=torch.float),
                persistent=False,
            )
            self.register_buffer(
                "mask_loss_bin_count",
                torch.zeros(bins, dtype=torch.float),
                persistent=False,
            )

    def upgrade_state_dict_named(self, state_dict, name):
        k = f"{name}.alibi_scale"
        if k in state_dict and state_dict[k].dim() == 4:
            state_dict[k] = state_dict[k].unsqueeze(0)

        return state_dict

    def convert_padding_mask(self, x, padding_mask):
        return padding_mask

    def decoder_input(self, x, mask_info: MaskInfo):
        inp_drop = self.modality_cfg.decoder.input_dropout
        if inp_drop > 0:
            x = F.dropout(x, inp_drop, training=self.training, inplace=True)

        num_extra = self.modality_cfg.num_extra_tokens

        if mask_info is not None:
            num_masked = mask_info.ids_restore.shape[1] - x.shape[1] + num_extra

            mask_tokens = x.new_empty(
                x.size(0),
                num_masked,
                x.size(-1),
            ).normal_(0, self.modality_cfg.mask_noise_std)

            x_ = torch.cat([x[:, num_extra:], mask_tokens], dim=1)
            x = torch.gather(x_, dim=1, index=mask_info.ids_restore)

            if self.modality_cfg.decoder.add_positions_masked:
                assert self.fixed_positional_encoder is not None
                pos = self.fixed_positional_encoder(x, None)
                x = x + (pos * mask_info.mask.unsqueeze(-1))
        else:
            x = x[:, num_extra:]

        if self.modality_cfg.decoder.add_positions_all:
            assert self.fixed_positional_encoder is not None
            x = x + self.fixed_positional_encoder(x, None)

        return x, mask_info

    def local_features(self, features):
        if self.local_grad_mult > 0:
            if self.local_grad_mult == 1.0:
                x = self.local_encoder(features)
            else:
                x = GradMultiply.apply(
                    self.local_encoder(features), self.local_grad_mult
                )
        else:
            with torch.no_grad():
                x = self.local_encoder(features)

        x = self.project_features(x)
        return x

    def contextualized_features(
        self,
        x,
        padding_mask,
        mask,
        remove_masked,
        clone_batch: int = 1,
        mask_seeds: Optional[torch.Tensor] = None,
        precomputed_mask=None,
    ):

        if padding_mask is not None:
            padding_mask = self.convert_padding_mask(x, padding_mask)

        local_features = x
        if mask and clone_batch == 1:
            local_features = local_features.clone()

        orig_B, orig_T, _ = x.shape
        pre_mask_B = orig_B
        mask_info = None

        x_pos = None
        if self.fixed_positional_encoder is not None:
            x = x + self.fixed_positional_encoder(x, padding_mask)

        if mask:
            if clone_batch > 1:
                x = x.repeat_interleave(clone_batch, 0)
                if mask_seeds is not None:
                    clone_hash = [
                        int(hash((mask_seeds.seed, ind)) % 1e10)
                        for ind in range(clone_batch - 1)
                    ]
                    clone_hash = torch.tensor([0] + clone_hash).long().view(1, -1)

                    id = mask_seeds.ids
                    id = id.repeat_interleave(clone_batch, 0)
                    id = id.view(-1, clone_batch) + clone_hash.to(id)
                    id = id.view(-1)
                    mask_seeds = MaskSeed(
                        seed=mask_seeds.seed, update=mask_seeds.update, ids=id
                    )
                if padding_mask is not None:
                    padding_mask = padding_mask.repeat_interleave(clone_batch, 0)

            x, mask_info = self.compute_mask(
                x,
                padding_mask,
                mask_seed=mask_seeds,
                apply=self.relative_positional_encoder is not None or not remove_masked,
                precomputed_mask=precomputed_mask,
            )

        if self.relative_positional_encoder is not None:
            x_pos = self.relative_positional_encoder(x)

        masked_padding_mask = padding_mask
        if mask and remove_masked:
            x = mask_info.x_unmasked
            if x_pos is not None:
                x = x + gather_unmasked(x_pos, mask_info)

            if padding_mask is not None and padding_mask.any():
                masked_padding_mask = gather_unmasked_mask(padding_mask, mask_info)
                if not masked_padding_mask.any():
                    masked_padding_mask = None
            else:
                masked_padding_mask = None

        elif x_pos is not None:
            x = x + x_pos

        alibi_bias = None
        alibi_scale = self.alibi_scale

        if self.get_alibi_bias is not None:
            alibi_bias = self.get_alibi_bias(
                batch_size=pre_mask_B,
                time_steps=orig_T,
                heads=self.modality_cfg.num_alibi_heads,
                dtype=torch.float32,
                device=x.device,
            )

            if alibi_scale is not None:
                alibi_scale = alibi_scale.clamp_min(0)
                if alibi_scale.size(0) == 1:
                    alibi_bias = alibi_bias * alibi_scale.squeeze(0).type_as(alibi_bias)
                    alibi_scale = None

            if clone_batch > 1:
                alibi_bias = alibi_bias.repeat_interleave(clone_batch, 0)

            if mask_info is not None and remove_masked:
                alibi_bias = masked_alibi(alibi_bias, mask_info)

        if self.extra_tokens is not None:
            num = self.extra_tokens.size(1)
            x = torch.cat([self.extra_tokens.expand(x.size(0), -1, -1), x], dim=1)
            if masked_padding_mask is not None:
                # B x T
                masked_padding_mask = F.pad(masked_padding_mask, (num, 0))
            if alibi_bias is not None:
                # B x H x T x T
                alibi_bias = F.pad(alibi_bias, (num, 0, num, 0))

        x = self.context_encoder(
            x,
            masked_padding_mask,
            alibi_bias,
            alibi_scale[: self.modality_cfg.prenet_depth]
            if alibi_scale is not None
            else None,
        )

        return {
            "x": x,
            "local_features": local_features,
            "padding_mask": masked_padding_mask,
            "alibi_bias": alibi_bias,
            "alibi_scale": alibi_scale[self.modality_cfg.prenet_depth :]
            if alibi_scale is not None and alibi_scale.size(0) > 1
            else alibi_scale,
            "encoder_mask": mask_info,
        }

    def forward(
        self,
        features,
        padding_mask,
        mask: bool,
        remove_masked: bool,
        clone_batch: int = 1,
        mask_seeds: Optional[torch.Tensor] = None,
        precomputed_mask=None,
    ):
        x = self.local_features(features)
        return self.contextualized_features(
            x,
            padding_mask,
            mask,
            remove_masked,
            clone_batch,
            mask_seeds,
            precomputed_mask,
        )

    def reset_parameters(self):
        pass

    def _anneal(self, start, end, curr_step, total_steps):
        if total_steps <= 0 or curr_step >= total_steps:
            return end
        if curr_step <= 0:
            return start
        pct = curr_step / total_steps
        return start + (end - start) * pct

    def _anneal_cosine(self, start, end, curr_step, total_steps):
        if total_steps <= 0 or curr_step >= total_steps:
            return end
        if curr_step <= 0:
            return start
        return end + 0.5 * (start - end) * (
            1 + math.cos(math.pi * curr_step / total_steps)
        )

    def set_num_updates(self, num_updates):
        cfg = self.modality_cfg
        self.current_num_updates = num_updates

        if (
            cfg.mask_prob_start is not None
            and cfg.mask_prob_end is not None
            and cfg.mask_prob_warmup_updates > 0
        ):
            if cfg.mask_prob_schedule == "cosine":
                self.current_mask_prob = self._anneal_cosine(
                    cfg.mask_prob_start,
                    cfg.mask_prob_end,
                    num_updates,
                    cfg.mask_prob_warmup_updates,
                )
            else:
                self.current_mask_prob = self._anneal(
                    cfg.mask_prob_start,
                    cfg.mask_prob_end,
                    num_updates,
                    cfg.mask_prob_warmup_updates,
                )
        else:
            self.current_mask_prob = cfg.mask_prob
        self.current_mask_prob = max(0.0, min(1.0, self.current_mask_prob))

        if (
            cfg.mask_length_start is not None
            and cfg.mask_length_end is not None
            and cfg.mask_length_warmup_updates > 0
        ):
            curr_mask_length = self._anneal(
                cfg.mask_length_start,
                cfg.mask_length_end,
                num_updates,
                cfg.mask_length_warmup_updates,
            )
            self.current_mask_length = max(1, int(round(curr_mask_length)))
        else:
            self.current_mask_length = cfg.mask_length

    def _mask_loss_curriculum_enabled(self):
        return (
            self.modality_cfg.type == Modality.AUDIO
            and self.modality_cfg.mask_loss_bins is not None
            and self.modality_cfg.mask_loss_bins > 1
        )

    def _mask_loss_curriculum_progress(self):
        cfg = self.modality_cfg
        if not self._mask_loss_curriculum_enabled():
            return 0.0

        start = cfg.mask_loss_warmup_updates
        if self.current_num_updates <= start:
            return 0.0

        ramp = cfg.mask_loss_easy_to_hard_updates
        if ramp <= 0:
            return 1.0

        return min(1.0, (self.current_num_updates - start) / ramp)

    def _mask_loss_bin_weights(self, device, dtype):
        bins = self.modality_cfg.mask_loss_bins
        weights = torch.ones(bins, device=device, dtype=dtype)

        if not self.mask_loss_bins_ready:
            return weights / weights.sum()

        loss = self.mask_loss_bin_ema.to(device=device, dtype=dtype)
        seen = self.mask_loss_bin_count.to(device=device) > 0
        if seen.any() and (~seen).any():
            loss = torch.where(seen, loss, loss[seen].mean())

        lo = loss.min()
        hi = loss.max()
        if (hi - lo).abs().item() < 1e-8:
            return weights / weights.sum()

        difficulty = (loss - lo) / (hi - lo)
        progress = self._mask_loss_curriculum_progress()
        easy = 1 - difficulty
        hard = difficulty
        weights = (1 - progress) * easy + progress * hard
        weights = weights.clamp_min(1.0e-6)
        return weights / weights.sum()

    @torch.no_grad()
    def update_mask_loss_bins(self, mask: torch.Tensor, token_loss: torch.Tensor):
        if not self._mask_loss_curriculum_enabled():
            return {}
        if mask is None or token_loss is None or token_loss.numel() == 0:
            return self.get_mask_curriculum_logs()

        mask = mask.bool()
        B, T = mask.shape

        flat_count = int(mask.sum().item())
        token_loss = token_loss.detach().float().view(-1)
        if flat_count == 0 or token_loss.numel() != flat_count:
            return self.get_mask_curriculum_logs()

        bins = self.modality_cfg.mask_loss_bins
        pos = torch.arange(T, device=mask.device)
        bin_ids = torch.clamp((pos * bins) // max(T, 1), max=bins - 1)
        flat_bins = bin_ids.unsqueeze(0).expand(B, -1)[mask]

        bin_sum = torch.bincount(flat_bins, weights=token_loss, minlength=bins).float()
        bin_cnt = torch.bincount(flat_bins, minlength=bins).float()

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(bin_sum)
            dist.all_reduce(bin_cnt)

        seen = bin_cnt > 0
        if seen.any():
            batch_avg = torch.zeros_like(bin_sum)
            batch_avg[seen] = bin_sum[seen] / bin_cnt[seen]
            decay = self.modality_cfg.mask_loss_ema_decay

            if not self.mask_loss_bins_ready:
                self.mask_loss_bin_ema[seen] = batch_avg[seen]
                self.mask_loss_bin_count[seen] = bin_cnt[seen]
            else:
                self.mask_loss_bin_ema[seen] = (
                    decay * self.mask_loss_bin_ema[seen] + (1 - decay) * batch_avg[seen]
                )
                self.mask_loss_bin_count[seen] = (
                    decay * self.mask_loss_bin_count[seen] + (1 - decay) * bin_cnt[seen]
                )

            self.mask_loss_bins_ready = True

        return self.get_mask_curriculum_logs()

    @torch.no_grad()
    def get_mask_curriculum_logs(self):
        if not self._mask_loss_curriculum_enabled():
            return {}

        logs = {
            "mask_curriculum_progress": self._mask_loss_curriculum_progress(),
            "mask_curriculum_ready": 1.0 if self.mask_loss_bins_ready else 0.0,
        }

        if self.mask_loss_bins_ready:
            loss = self.mask_loss_bin_ema
            logs["mask_curriculum_bin_loss_min"] = float(loss.min().item())
            logs["mask_curriculum_bin_loss_max"] = float(loss.max().item())

        return logs

    def _compute_curriculum_mask(
        self,
        B,
        T,
        device,
        padding_mask,
        mask_prob,
        mask_length,
    ):
        cfg = self.modality_cfg
        mask = torch.zeros(
            (B, T),
            dtype=torch.bool,
            device=device,
        )

        pos_weights = self._mask_loss_bin_weights(mask.device, torch.float)
        bin_ids = torch.clamp(
            (torch.arange(T, device=mask.device) * cfg.mask_loss_bins) // max(T, 1),
            max=cfg.mask_loss_bins - 1,
        )
        pos_weights = pos_weights[bin_ids]

        explore = max(0.0, min(1.0, cfg.mask_loss_explore_pct))

        for i in range(B):
            valid_len = T
            if padding_mask is not None:
                valid_len = int((~padding_mask[i].bool()).sum().item())
            if valid_len <= 0:
                continue

            target_mask = int(round(valid_len * mask_prob))
            if mask_prob > 0 and target_mask == 0:
                target_mask = 1
            if target_mask <= 0:
                continue

            m = torch.zeros(valid_len, dtype=torch.bool, device=mask.device)
            w = pos_weights[:valid_len]

            if mask_length <= 1:
                probs = w / w.sum()
                if explore > 0:
                    probs = (1 - explore) * probs + explore * (1.0 / probs.numel())
                n = min(valid_len, target_mask)
                if n > 0:
                    idx = torch.multinomial(probs, n, replacement=False)
                    m[idx] = True
            else:
                n_spans = max(1, int(round(valid_len * mask_prob / mask_length)))
                n_starts = max(1, valid_len - mask_length + 1)
                n_spans = min(n_spans, n_starts)

                if valid_len >= mask_length:
                    csum = torch.cat([w.new_zeros(1), torch.cumsum(w, dim=0)])
                    span_scores = csum[mask_length:] - csum[:-mask_length]
                    span_scores = span_scores / mask_length
                else:
                    span_scores = w.new_full((1,), w.mean())

                probs = span_scores.clamp_min(1.0e-8)
                probs = probs / probs.sum()
                if explore > 0:
                    probs = (1 - explore) * probs + explore * (1.0 / probs.numel())

                starts = torch.multinomial(probs, n_spans, replacement=False)
                for s in starts.tolist():
                    e = min(valid_len, s + mask_length)
                    m[s:e] = True

            if cfg.mask_dropout > 0 and m.any():
                masked_idx = torch.nonzero(m, as_tuple=False).flatten()
                n_drop = min(
                    masked_idx.numel(),
                    int(round(masked_idx.numel() * cfg.mask_dropout)),
                )
                if n_drop > 0:
                    drop_idx = masked_idx[
                        torch.randperm(masked_idx.numel(), device=masked_idx.device)[
                            :n_drop
                        ]
                    ]
                    m[drop_idx] = False

            n_masked = int(m.sum().item())
            if n_masked > target_mask:
                masked_idx = torch.nonzero(m, as_tuple=False).flatten()
                to_unmask = masked_idx[
                    torch.randperm(masked_idx.numel(), device=masked_idx.device)[
                        : n_masked - target_mask
                    ]
                ]
                m[to_unmask] = False
            elif n_masked < target_mask:
                unmasked_idx = torch.nonzero(~m, as_tuple=False).flatten()
                n_add = min(unmasked_idx.numel(), target_mask - n_masked)
                if n_add > 0:
                    to_mask = unmasked_idx[
                        torch.randperm(
                            unmasked_idx.numel(), device=unmasked_idx.device
                        )[:n_add]
                    ]
                    m[to_mask] = True

            mask[i, :valid_len] = m

        return mask

    def compute_mask(
        self,
        x,
        padding_mask,
        mask_seed: Optional[MaskSeed],
        apply,
        precomputed_mask,
    ):
        if precomputed_mask is not None:
            mask = precomputed_mask
            mask_info = self.make_maskinfo(x, mask)
        else:
            B, T, C = x.shape
            cfg = self.modality_cfg

            mask_prob = self.current_mask_prob
            mask_length = self.current_mask_length

            if (
                cfg.mask_prob_min is not None
                and cfg.mask_prob_min >= 0
                and cfg.mask_prob_min < mask_prob
            ):
                mask_prob = np.random.uniform(cfg.mask_prob_min, mask_prob)

            if mask_prob > 0:
                if mask_length == 1:
                    mask_info = random_masking(x, mask_prob, mask_seed)
                else:
                    use_loss_curriculum = (
                        self._mask_loss_curriculum_enabled()
                        and self.current_num_updates >= cfg.mask_loss_warmup_updates
                    )

                    if self.modality_cfg.inverse_mask:
                        mask_prob = 1 - mask_prob

                    if use_loss_curriculum:
                        mask = self._compute_curriculum_mask(
                            B,
                            T,
                            x.device,
                            padding_mask,
                            mask_prob,
                            mask_length,
                        )
                    else:
                        mask = compute_mask_indices(
                            (B, T),
                            padding_mask,
                            mask_prob,
                            mask_length,
                            min_masks=1,
                            require_same_masks=True,
                            mask_dropout=cfg.mask_dropout,
                            add_masks=cfg.add_masks,
                            seed=mask_seed.seed if mask_seed is not None else None,
                            epoch=mask_seed.update if mask_seed is not None else None,
                            indices=mask_seed.ids if mask_seed is not None else None,
                        )

                        mask = torch.from_numpy(mask).to(device=x.device)
                    if self.modality_cfg.inverse_mask:
                        if padding_mask is None:
                            mask = ~mask.bool()
                        else:
                            valid = ~padding_mask.bool()
                            mask = (~mask.bool()) & valid
                    mask_info = self.make_maskinfo(x, mask)
            else:
                mask_info = None

        if apply:
            x = self.apply_mask(x, mask_info)

        return x, mask_info

    def make_maskinfo(self, x, mask, shape=None):
        if shape is None:
            B, T, D = x.shape
        else:
            B, T, D = shape

        mask = mask.to(torch.uint8)
        ids_shuffle = mask.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1).unsqueeze(-1).expand(-1, -1, D)

        len_keep = T - mask[0].sum()
        if self.modality_cfg.keep_masked_pct > 0:
            len_keep += round((T - int(len_keep)) * self.modality_cfg.keep_masked_pct)

        ids_keep = ids_shuffle[:, :len_keep]

        if shape is not None:
            x_unmasked = None
        else:
            ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, D)
            x_unmasked = torch.gather(x, dim=1, index=ids_keep)

        mask_info = MaskInfo(
            x_unmasked=x_unmasked,
            mask=mask,
            ids_restore=ids_restore,
            ids_keep=ids_keep,
        )
        return mask_info

    def apply_mask(self, x, mask_info):
        cfg = self.modality_cfg
        B, T, C = x.shape

        if mask_info is not None:
            mask = mask_info.mask
            if cfg.encoder_zero_mask:
                x = x * (1 - mask.type_as(x).unsqueeze(-1))
            else:
                num_masks = mask.sum().item()
                masks = x.new_empty(num_masks, x.size(-1)).normal_(
                    0, cfg.mask_noise_std
                )
                x = index_put(x, mask, masks)
        if cfg.mask_channel_prob > 0:
            mask_channel = compute_mask_indices(
                (B, C),
                None,
                cfg.mask_channel_prob,
                cfg.mask_channel_length,
            )
            mask_channel = (
                torch.from_numpy(mask_channel)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x = index_put(x, mask_channel, 0)
        return x

    def remove_pretraining_modules(self, keep_decoder=False):
        if not keep_decoder:
            self.decoder = None


def get_annealed_rate(start, end, curr_step, total_steps):
    if curr_step >= total_steps:
        return end
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining


# adapted from MAE
def random_masking(x, mask_ratio, mask_seed: Optional[MaskSeed]):
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    generator = None
    if mask_seed is not None:
        seed = int(
            hash((mask_seed.seed, mask_seed.update, mask_seed.ids.sum().item())) % 1e6
        )
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)

    noise = torch.rand(N, L, generator=generator, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = noise.argsort(dim=1)  # ascend: small is keep, large is remove
    ids_restore = ids_shuffle.argsort(dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, D)
    x_unmasked = torch.gather(x, dim=1, index=ids_keep)

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], dtype=x.dtype, device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    ids_restore = ids_restore.unsqueeze(-1).expand(-1, -1, D)

    return MaskInfo(
        x_unmasked=x_unmasked, mask=mask, ids_restore=ids_restore, ids_keep=ids_keep
    )


def gather_unmasked(x: torch.Tensor, mask_info: MaskInfo) -> torch.Tensor:
    return torch.gather(
        x,
        dim=1,
        index=mask_info.ids_keep,
    )


def gather_unmasked_mask(x: torch.Tensor, mask_info: MaskInfo) -> torch.Tensor:
    return torch.gather(
        x,
        dim=1,
        index=mask_info.ids_keep[..., 0],  # ignore the feature dimension
    )


def get_alibi(
    max_positions: int,
    attention_heads: int,
    dims: int = 1,
    distance: str = "manhattan",
):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        # In the paper, we only train models that have 2^a heads for some
        # a. This function has some good properties that only occur when
        # the input is a power of 2. To maintain that even when the number
        # of heads is not a power of 2, we use this workaround.
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    maxpos = max_positions
    attn_heads = attention_heads
    slopes = torch.Tensor(get_slopes(attn_heads))

    if dims == 1:
        # prepare alibi position linear bias. Note that wav2vec2 is non
        # autoregressive model so we want a symmetric mask with 0 on the
        # diagonal and other wise linear decreasing valuees
        pos_bias = (
            torch.abs(
                torch.arange(maxpos).unsqueeze(0) - torch.arange(maxpos).unsqueeze(1)
            )
            * -1
        )
    elif dims == 2:
        if distance == "manhattan":
            df = lambda x1, y1, x2, y2: abs(x1 - x2) + abs(y1 - y2)
        elif distance == "euclidean":
            df = lambda x1, y1, x2, y2: math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        n = math.sqrt(max_positions)
        assert n.is_integer(), n
        n = int(n)

        pos_bias = torch.zeros((max_positions, max_positions))

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        new_x = i * n + j
                        new_y = k * n + l
                        pos_bias[new_x, new_y] = -df(i, j, k, l)

    else:
        raise Exception(f"unsupported number of alibi dims: {dims}")

    alibi_bias = slopes.unsqueeze(1).unsqueeze(1) * pos_bias.unsqueeze(0).expand(
        attn_heads, -1, -1
    )

    return alibi_bias


def get_alibi_bias(
    alibi_biases,
    batch_size,
    time_steps,
    heads,
    dtype,
    device,
    dims=1,
    distance="manhattan",
):
    cache_key = f"{dims}_{heads}_{distance}"

    buffered = alibi_biases.get(cache_key, None)

    target_size = heads * batch_size
    if (
        buffered is None
        or buffered.size(0) < target_size
        or buffered.size(1) < time_steps
        or buffered.dtype != dtype
        or buffered.device != device
    ):
        bt = max(time_steps, buffered.size(1) if buffered is not None else 0)
        bn = max(target_size, buffered.size(0) if buffered is not None else 0) // heads

        buffered = (
            get_alibi(bt, heads, dims=dims, distance=distance)
            .to(dtype=dtype, device=device)
            .repeat(bn, 1, 1)
        )

        alibi_biases[cache_key] = buffered

    b = buffered[:target_size, :time_steps, :time_steps]
    b = b.view(batch_size, heads, time_steps, time_steps)
    return b


def _learned_alibi_bias(
    alibi_bias,
    batch_size,
    time_steps,
    heads,
    scale,
    dtype,
    device,
):
    assert alibi_bias.size(1) == heads, alibi_bias.shape
    assert alibi_bias.dtype == dtype, alibi_bias.dtype
    assert alibi_bias.device == device, alibi_bias.device

    if alibi_bias.size(-1) < time_steps:
        psz = math.ceil((time_steps - alibi_bias.size(-1)) / 2)
        alibi_bias = F.pad(alibi_bias, (psz, psz, psz, psz), mode="replicate")

    alibi_bias = alibi_bias.expand(batch_size, -1, -1, -1) * scale
    return alibi_bias[..., :time_steps, :time_steps]


def masked_alibi(alibi_bias, mask_info):
    H = alibi_bias.size(1)

    orig_bias = alibi_bias

    index = mask_info.ids_keep.unsqueeze(1)[..., 0].unsqueeze(-1)
    alibi_bias = torch.gather(
        orig_bias,
        dim=-2,
        index=index.expand(-1, H, -1, mask_info.ids_restore.size(1)),
    )
    alibi_bias = torch.gather(
        alibi_bias,
        dim=-1,
        index=index.transpose(-1, -2).expand(-1, H, alibi_bias.size(-2), -1),
    )

    return alibi_bias
