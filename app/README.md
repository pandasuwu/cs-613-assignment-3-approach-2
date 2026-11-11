---
title: COAST Embedding Post-Processing & Compression Explorer
emoji: 📊
colorFrom: gray
colorTo: blue
sdk: static
pinned: false
---

# COAST: Text Embedding Post-Processing & Compression Explorer

Interactive results explorer for **Crowding-Orthogonal Adaptive Spectral Tempering (COAST)** across 4 transformer backbones, 2 token pooling strategies, 6 MTEB v2 retrieval and semantic similarity benchmarks, 14 main operators, and 16 component ablation variants (`18,630` evaluated metric cells).

## Running the Local Gradio Server

```bash
uv run --with gradio --with polars --with pandas python app/app.py
```
