# CPVB-RG: Cross-Prior Visual Bridging for Prior-Aware Chest X-Ray Report Generation

![Overview of CPVB-RG](assets/overall.png)

[Paper (arXiv)](https://arxiv.org/abs/2610.28289) · Accepted at WACV 2027

This repository contains the core PyTorch implementation of **Progression-Guided Focus Attention (PGFA)** and the **Cross-Prior Visual Bridge (CPVB)**. PGFA contrasts progression-difference attention with shared-spatial attention. CPVB combines the resulting visual features with clinical text through two cross-attention units.

## Usage

```bash
pip install -r requirements.txt
```

```python
import torch
from cpvb import CrossPriorVisualBridge

model = CrossPriorVisualBridge(hidden_size=768)
current = torch.randn(1, 196, 768)
prior = torch.randn(1, 196, 768)
clinical = torch.randn(1, 32, 768)

output = model(current, prior, clinical)
print(output.shape)  # torch.Size([1, 256, 768])
```

Inputs are encoded token sequences. Current and prior visual tokens must have the same shape `(batch, visual_tokens, hidden_size)`; clinical tokens have shape `(batch, text_tokens, hidden_size)`. The hidden size must be divisible by the number of attention heads. With the default 128 anchor tokens, CPVB returns `(batch, 256, hidden_size)`.

Pass `prior=None` to use the current tokens as the prior placeholder. The original implementation detaches the prior tokens inside CPVB. This release includes only the core modules, without encoders, the language decoder, data processing, training scripts or pretrained weights.

## Citation

```bibtex
@misc{xiang2026cpvbrgcrosspriorvisualbridging,
  title={CPVB-RG: Cross-Prior Visual Bridging for Prior-Aware Chest X-Ray Report Generation},
  author={Sike Xiang and Shuang Chen and Julie Cox and Amir Atapour-Abarghouei},
  year={2026},
  eprint={2610.28289},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.28289}
}
```
