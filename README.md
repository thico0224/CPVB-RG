# CPVB-RG: Cross-Prior Visual Bridging for Prior-Aware Chest X-Ray Report Generation

## Abstract
Radiologists interpret follow-up chest X-rays by comparing current and prior examinations to identify new, resolved or progressing findings. Most report generation systems either omit this longitudinal context or fuse prior features without distinguishing interval changes from anatomy that remains stable over time. We propose CPVB-RG, a prior-aware framework that combines current and prior radiographs with available clinical text. Its Cross-Prior Visual Bridge (CPVB) contains Progression-Guided Focus Attention (PGFA) and two Cross-Attention units. PGFA is designed to contrast progression-difference attention with shared-spatial attention, aiming to reduce invariant responses whilst retaining coarse spatial correspondence between examinations. The Cross-Attention units then connect the resulting longitudinal visual evidence with clinical semantics. CPVB-RG is trained in two stages, using prior-guided image-text alignment followed by autoregressive report generation with a lightweight language decoder. Experiments on MIMIC-CXR and IU X-Ray show consistent improvements across language and clinical metrics. On MIMIC-CXR, CPVB-RG achieves 0.2847 F1RadGraph and 0.6692 VLScore, with statistically significant gains over the closest decoder-matched prior-aware baseline. These results support the effectiveness of the proposed difference-aware and spatially guided fusion design for prior-aware chest X-ray report generation.

##  Model Overview
<p align="center">
  <img src="assets/overall.png" alt="Framework of CPVB-RG" width="1000"/>
</p>


## Citation

If you find our work useful, please consider citing:

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
