# Provenance

This repository contains an independent implementation. No third-party source
code is included.

## History, stated plainly

Development began as a fork of [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
(MIT). The training loop, data loader and baseline model were subsequently
rewritten from scratch, and the repository history was restarted so that no
upstream code remains in either the working tree or the git history. The
original MIT notice was retained for as long as derived code was present.

This is recorded because provenance is worth being able to check, not because
any obligation remains.

## What the architecture is based on

The techniques implemented here are published research, not original to this
project or to nanoGPT. Architectural ideas are not subject to copyright; the
implementations in `pythonos/` are this project's own expression of them.

| Component | Source |
|---|---|
| Transformer, multi-head attention | Vaswani et al., *Attention Is All You Need* (2017) |
| Decoder-only LM, GPT-2 scale conventions | Radford et al. (2018, 2019) |
| Pre-norm residual placement | Xiong et al. (2020) |
| Weight tying | Press & Wolf (2017) |
| GELU | Hendrycks & Gimpel (2016) |
| RMSNorm | Zhang & Sennrich (2019) |
| Rotary position embeddings | Su et al., *RoFormer* (2021) |
| Sliding-window attention | Beltagy et al., *Longformer* (2020) |
| No positional encoding (NoPE) | Kazemnejad et al. (2023) |
| Multihead Latent Attention | DeepSeek-V2 / V3 (2024, 2025) |
| Fine-grained MoE, shared experts | DeepSeek-MoE (2024) |
| Load-balancing auxiliary loss | Fedus et al., *Switch Transformer* (2021) |
| Hyper-Connections | Zhu et al. (2024) |
| Manifold-constrained Hyper-Connections | DeepSeek (2025) |
| Sinkhorn-Knopp projection | Sinkhorn & Knopp (1967) |
| AdamW | Loshchilov & Hutter (2017) |
| Cosine learning-rate schedule | Loshchilov & Hutter, *SGDR* (2016) |
| MFU estimator | Chowdhery et al., *PaLM* (2022), Appendix B |

Where this project's reading of a technique departs from, or goes beyond, the
published description — the MoE orthogonality and variance losses, and the
hyper-connection initialisation schemes — that is stated in the relevant
docstring and in `STAGES.md`. Those are design choices under test, not
reproductions.
