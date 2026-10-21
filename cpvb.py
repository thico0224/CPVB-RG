"""Core PGFA and CPVB modules for CPVB-RG."""

import torch
from torch import nn
import torch.nn.functional as F


class ProgressionGuidedFocusAttention(nn.Module):

    def __init__(self, hidden_size, num_heads=8, reduction=2):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.reduction = max(1, reduction)
        self.lambda_spatial = nn.Parameter(torch.tensor(0.5))

        self.q_current = nn.Linear(hidden_size, hidden_size)
        self.k_current = nn.Linear(hidden_size, hidden_size)
        self.v_current = nn.Linear(hidden_size, hidden_size)
        self.k_prior = nn.Linear(hidden_size, hidden_size)
        self.v_prior = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.normalise = nn.LayerNorm(hidden_size)

    def _reduce(self, tokens):
        if self.reduction == 1 or tokens.size(1) <= 1:
            return tokens
        tokens = tokens.transpose(1, 2)
        tokens = F.avg_pool1d(tokens, kernel_size=self.reduction, stride=self.reduction, ceil_mode=True)
        return tokens.transpose(1, 2)

    def _as_heads(self, tokens):
        batch, length, _ = tokens.shape
        tokens = tokens.view(batch, length, self.num_heads, self.head_dim)
        return tokens.transpose(1, 2)

    def _from_heads(self, tokens):
        batch, _, length, _ = tokens.shape
        tokens = tokens.transpose(1, 2).contiguous()
        return tokens.view(batch, length, self.hidden_size)

    def forward(self, current_tokens, prior_tokens):
        residual = current_tokens
        current_tokens = self._reduce(current_tokens)
        prior_tokens = self._reduce(prior_tokens)

        query = self._as_heads(self.q_current(current_tokens))
        current_key = self._as_heads(self.k_current(current_tokens))
        current_value = self._as_heads(self.v_current(current_tokens))
        prior_key = self._as_heads(self.k_prior(prior_tokens))
        prior_value = self._as_heads(self.v_prior(prior_tokens))

        change_key = current_key - prior_key
        change_value = current_value - prior_value
        shared_key = 0.5 * (current_key + prior_key)

        scale = self.head_dim ** -0.5
        change_attention = torch.softmax(torch.matmul(query, change_key.transpose(-1, -2)) * scale, dim=-1)
        shared_attention = torch.softmax(torch.matmul(query, shared_key.transpose(-1, -2)) * scale, dim=-1)
        attention = change_attention - self.lambda_spatial.sigmoid() * shared_attention

        update = torch.matmul(attention, change_value)
        update = self.output(self._from_heads(update))
        if update.size(1) != residual.size(1):
            update = F.interpolate(update.transpose(1, 2), size=residual.size(1), mode="linear", align_corners=False)
            update = update.transpose(1, 2)
        return self.normalise(residual + update)


class CrossPriorVisualBridge(nn.Module):

    def __init__(self, hidden_size, num_heads=8, num_latents=128, num_pgfa_layers=3, reduction=2):
        super().__init__()
        self.pgfa_layers = nn.ModuleList([
            ProgressionGuidedFocusAttention(hidden_size, num_heads=num_heads, reduction=reduction)
            for _ in range(num_pgfa_layers)
        ])
        self.bridge_queries = nn.Parameter(torch.randn(num_latents, hidden_size) * 0.02)
        self.text_cross_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.visual_cross_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.text_norm = nn.LayerNorm(hidden_size)
        self.visual_norm = nn.LayerNorm(hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, current_tokens, prior_tokens, clinical_tokens):
        prior_tokens = current_tokens if prior_tokens is None else prior_tokens
        linked_visual_tokens = current_tokens
        for pgfa in self.pgfa_layers:
            linked_visual_tokens = pgfa(linked_visual_tokens, prior_tokens.detach())

        batch_size = current_tokens.shape[0]
        anchors = self.bridge_queries.unsqueeze(0).expand(batch_size, -1, -1)
        clinical_tokens, _ = self.text_cross_attention(anchors, clinical_tokens, clinical_tokens, need_weights=False)
        clinical_tokens = self.text_norm(anchors + clinical_tokens)

        visual_tokens, _ = self.visual_cross_attention(
            clinical_tokens, linked_visual_tokens, linked_visual_tokens, need_weights=False
        )
        visual_tokens = self.visual_norm(clinical_tokens + visual_tokens)
        return self.output_norm(torch.cat([clinical_tokens, visual_tokens], dim=1))

