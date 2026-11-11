# Literature Analysis and Multi-Axis Presentation Strategy for ACL 2027

This document synthesizes structural, theoretical, and empirical presentation insights from reading the full texts of four foundational precursor papers (`Ren et al., ICML 2026`; `Li et al., SIGIR 2026`; `Yokoi et al., TACL 2024`; `Diera et al., 2024`) alongside recent ACL, EMNLP, EACL, and NeurIPS literature on multi-benchmark evaluation and aggregation pitfalls.

---

## 1. How Top Papers in This Area Structure Their Narrative and Evidence

### 1.1 Ren et al. (ICML 2026) - *Correcting Mean Bias in Text Embeddings (R2)*
- **Core Narrative Structure**:
  - Opens with a crisp empirical observation: unit-normalized embeddings $e \in \mathbb{S}^{d-1}$ decompose into $e = \tilde{e} + \mu$ with a large, model-intrinsic mean shift $\|\mu\|_2 > 0$.
  - Formulates three falsifiable hypotheses upfront in the Introduction (`H1: R2 > R1` via parallel error cancellation; `H2: ||mu|| predicts benefit`; `H3: Goldilocks dose-response` where full whitening hurts contrastively trained encoders).
  - Separates the **mathematical error-propagation proof** (first-order Taylor expansion showing R2 cancels parallel sample-mean estimation error $\alpha \hat{\mu}$) from the **dose-response empirical audit** across 9 methods (`R1`, `R2`, `ABTT-1..3`, `mc`, `rand`, `whitening`).
- **How They Handle Multi-Axis Data Without Misleading Aggregation**:
  - Explicitly refuses to pool raw scores across heterogeneous task families or evaluation snapshots.
  - Uses **paired head-to-head comparisons** (`Win/Tie/Loss` counts and paired deltas $\Delta$ relative to each model's own baseline) and orders models by their intrinsic geometric diagnostic $\|\mu\|_2$.
  - Dedicates explicit sections to **scientific negative controls** (`mc` without renormalization; `rand` direction deflation) and **mechanistic probes** (comparing $\cos(\hat{\mu}, \text{PC1}_{\text{centered}})$ vs $\cos(\hat{\mu}, \text{PC1}_{\text{uncentered}})$).
- **Limitation We Resolve**:
  - Ren et al. only test contrastively trained embedding models on full-rank whitening ($\epsilon = 10^{-6}, \gamma = 1$) and conclude that spectral whitening universally hurts (`-0.64` to `-5.18` pp). They do not evaluate un-fine-tuned base causal LLMs (where 1D deflation closes $<5\%$ of the retrieval gap while spectral tempering closes up to $55\%$) and do not connect $\|\mu\|_2^2$ to the covariance trace $\operatorname{Tr}(\Sigma_\perp) \approx 1 - \|\mu\|_2^2$ or adaptive spectral tempering.

### 1.2 Li et al. (SIGIR 2026) - *Spectral Tempering for Embedding Compression (SpecTemp)*
- **Core Narrative Structure**:
  - Motivates the compression dilemma via a **dual-regime failure analysis**: unweighted PCA ($\gamma = 0$) fails at small $k$ ($k = 64$) due to head-eigenvalue skewness, whereas full Whitening ($\gamma = 1$) fails at large $k$ ($k = 512$) by amplifying the low-SNR tail floor.
  - Introduces local SNR estimation from the bottom $10\%$ eigenvalue tail $\sigma_{\text{noise}}^2$ and Kneedle knee detection $k_{\text{knee}}$ to define $\gamma_{\text{SNR}}(k) = \min(1, \mathrm{SNR}(k) / \mathrm{SNR}(k_{\text{knee}}))$.
- **How They Present Multi-Model, Multi-Dataset, Multi-$k$ Results**:
  - **Zero cross-dataset averaging**: In their main result table (Table 2), columns are the 4 individual retrieval datasets (`MS MARCO`, `NQ`, `FEVER`, `FiQA`), each subdivided into $k \in \{768, 512, 256, 128, 64\}$, and rows are grouped by model block. Every dataset remains completely independent.
  - Pairs the main table with a **diagnostic table** (Table 3) comparing predicted $\gamma(k)$ against grid-searched oracle $\gamma^*(k)$ across $k$.
- **Limitation We Resolve**:
  - SpecTemp assumes that as $k \to d$, $\gamma(k)$ should decay to $0$ (reverting to unwhitened PCA). This assumption holds only for contrastively trained encoders in the dispersion-dominated regime ($\rho_{\text{iso}} < 1$). On un-fine-tuned base causal LLMs ($\rho_{\text{iso}} > 1$), driving $\gamma(512) \to 0$ turns off whitening when cone crowding is severe, causing SpecTemp to lose **24/24 times** at $k = 512$. Furthermore, SpecTemp uses unregularized $\lambda_i^{-\gamma/2}$ ($\epsilon = 10^{-6}$), leaving it vulnerable when $N < k$ or when $k = d$.

### 1.3 Yokoi et al. (TACL 2024) - *Zipfian Whitening*
- **Core Narrative Structure**:
  - Combines a **clean first-principles mathematical decomposition** (1st-order symmetry = origin centering $1 - \|\mathbb{E}[v]\| / \mathbb{E}[\|v\|]$; 2nd-order symmetry = normalized spectral entropy $\mathrm{NID} = H(\tilde{\lambda}) / \log d$) with a **side-by-side theoretical summary table** (their Table 4) contrasting existing paradigms.
  - Demonstrates that neither 1st-moment centering alone nor 2nd-moment isotropy alone explains downstream performance; rather, **joint 1st-moment and 2nd-moment symmetry** governs representation quality (their Figure 2 and Table 3).
  - Proves in Section 5.1 that uniform covariance estimation over contextual token occurrences in natural text implicitly weights word types by their Zipfian frequency.

### 1.4 Diera et al. (2024) - *Soft-ZCA Whitening*
- **Core Narrative Structure**:
  - Introduces additive spectral regularization $W_{\text{ZCA}} = U(\Lambda + \epsilon I)^{-1/2} U^\top$ and plots downstream ranking metrics alongside `IsoScore` across logarithmic grids of $\epsilon \in \{10^{-1}, 10^{-2}, 10^{-4}, 0\}$.
  - Shows that base models and fine-tuned models require **orders-of-magnitude different $\epsilon$ values** when $\epsilon$ is tuned manually via grid search.
- **Limitation We Resolve**:
  - Soft-ZCA requires manual grid search over $\epsilon$ and fixes $\gamma = 1$. When fixed to its default $\epsilon = 10^{-4}$, it sits $40\times$ below the spectral knee $\lambda_{k_{\text{knee}}} \approx 4 \times 10^{-3}$ on dedicated contrastive encoders, degrading them in **17/18 evaluations**.

---

## 2. Why Naive Aggregation Fails in Multi-Axis Embedding Benchmarks

Recent benchmarking methodology papers (`Peyrard et al., 2021`; `Shavrina and Malykh, 2021`; `Colombo et al., NeurIPS 2022`; `Rofin et al., EACL 2023`; `Reliability Audit of Aggregated MTEB Scores, AACL 2026`) identify four mathematical pathologies when averaging raw scores across heterogeneous datasets, models, and metrics—all four of which are present in our experimental matrix:

1. **Baseline Scale Incommensurability Across Datasets**:
   - On un-fine-tuned base causal LLMs (`mean_pooling`), baseline nDCG@10 is `0.3050` on `ArguAna`, `0.0504` on `FiQA2018`, and `0.0285` on `SCIDOCS`—an order-of-magnitude ($10.7\times$) difference in absolute scale.
   - An unweighted arithmetic mean across IR datasets is dominated by `ArguAna` ($79\%$ of the sum). A method that more than doubles `SCIDOCS` nDCG@10 (`0.0285 -> 0.0671`, $+135\%$ relative gain) moves a 3-dataset arithmetic mean by only $+0.0129$, whereas a $4\%$ fluctuation on `ArguAna` moves the mean by just as much.

2. **Sequence-Length Phase Transitions (Simpson's Paradox)**:
   - Our six benchmarks span $10$ mean tokens (`SICK-R`) and $12$ mean tokens (`STSBenchmark`) to $174\text{--}226$ mean tokens (`FiQA2018`, `ArguAna`, `SCIDOCS`) and $638$ mean tokens / up to $2,048$ tokens (`STS22.v2`).
   - On short sentences (`SICK-R`), causal `last_token_pooling` outperforms `mean_pooling` at baseline (`49.55` vs `47.44`, $+2.11$ Spearman points) because the final token attends to the full clause without decay.
   - On long multi-paragraph documents (`STS22.v2`), `last_token_pooling` collapses catastrophically (`14.17` vs `59.49`, $-45.33$ Spearman points).
   - Averaging across STS datasets conflates short-sentence lexical composition with long-context autoregressive attention decay.

3. **Ratio Explosion When Baseline Approaches Zero**:
   - In compression tables, relative retention $\text{Score}(k) / \text{Score}(d) \times 100\%$ is well-conditioned when native baseline scores are bounded away from zero (such as Dedicated Encoders, where nDCG@10 $\in [0.19, 0.49]$ and Spearman $\rho \in [66.5, 84.9]$).
   - However, on `FiQA2018` under `Qwen3.5-0.8B-Base (last_token_pooling)`, the native baseline score is `0.0026`. A tiny absolute shift to `0.0060` yields a `232.8%` retention ratio. Averaging retention percentages across base models weights near-zero denominators disproportionately.

4. **Opposing Spectral Regimes Across Model Tiers ($\rho_{\text{iso}} < 1$ vs $\rho_{\text{iso}} > 1$)**:
   - Dedicated contrastive encoders reside in the **Dispersion-Dominated Regime** ($\rho_{\text{iso}} = \frac{\|\mu\|_2^2}{\operatorname{Tr}(\Sigma_\perp)\,\mathrm{NID}} \in [0.23, 0.81] < 1$), whereas un-fine-tuned base LLMs reside in the **Crowding-Dominated Regime** ($\rho_{\text{iso}} \in [1.13, 9.09] > 1$).
   - Averaging across model tiers cancels out opposite signed effects (e.g., `Soft-ZCA` losing `-0.0635` nDCG@10 on dedicated `FiQA2018` while gaining `+0.0287` on base `FiQA2018`).

---

## 3. Principled Presentation Principles for Our ACL 2027 Paper

To achieve top-tier clarity, rigor, and zero data distortion within an 8-page ACL main body (plus unlimited Appendix), we adopt five strict presentation principles:

1. **Never Average Across Datasets or Across Model Tiers**:
   - Every main-body table reports **per-dataset** numbers (`FiQA2018`, `ArguAna`, `SCIDOCS`, `STSBenchmark`, `SICK-R`, `STS22.v2`) without collapsing datasets into a single opaque "MTEB Average" column.
   - Model tiers are strictly stratified into **Dedicated Contrastive Encoders**, **Base Causal LLMs (Mean Pooling)**, and **Base Causal LLMs (Last-Token Pooling)**. Within each homogeneous 2-model tier (`EmbGemma` + `Qwen3-Emb`, or `Gemma-3` + `Qwen3.5`), both modelos share the exact same training paradigm and metric scale, and all per-model tables are provided in full in the Appendix.

2. **Lead with the Unified Spectral Operator Equation and Regime Separation Theorem**:
   - Frame Section 3 around the unified spectral family:
     $$f(x; k) = \phi\!\left(\big(\Lambda_k + \epsilon I_k\big)^{-\gamma / 2} U_k^\top g_{\text{center}}(x)\right)$$
   - Present a side-by-side **Taxonomy & Regime Table** (inspired by Yokoi et al., Table 4) showing how `PCA`, `Whitening`, `R2`, `ABTT-D`, `Soft-ZCA`, `SpecTemp`, and **KRAST** are special parameterizations of $(g_{\text{center}}, \epsilon, \gamma, k)$, and why only **KRAST** satisfies all four regime constraints simultaneously.
   - Present the **Unit-Sphere Energy Partition and Anisotropic Crowding Ratio**:
     $$E_{\text{crowd}} = \|\mu\|_2^2, \qquad E_{\text{disp}} = \operatorname{Tr}(\Sigma_\perp) \approx 1 - \|\mu\|_2^2, \qquad E_{\text{iso}} = E_{\text{disp}}\,\mathrm{NID}(\Sigma_\perp)$$
     $$\rho_{\text{iso}} = \frac{E_{\text{crowd}}}{E_{\text{iso}}} < 1 \;\; (\text{12/12 Dedicated Encoders}) \quad \text{vs.} \quad \rho_{\text{iso}} > 1 \;\; (\text{24/24 Base LLMs})$$

3. **Use Paired Win/Loss Records and Gap-Closure alongside Per-Dataset Scores**:
   - When summarizing across the 36 configurations (or 180 compression configurations), use **exact paired head-to-head win counts** (e.g., `17/18` wins over `Soft-ZCA` on dedicated encoders; `16/24` wins over `Soft-ZCA` on base LLMs; `96/96` wins over `SpecTemp` across $k \in \{512, 256, 128, 64\}$ on base LLMs; `10/12` wins over native `MRL` at $k = 64$) rather than cross-dataset arithmetic means.
   - For base-to-dedicated comparisons, report **per-dataset Gap Closure %**:
     $$\text{Gap Closed (\%)} = \frac{\text{Score}_{\text{post}} - \text{Score}_{\text{base, raw}}}{\text{Score}_{\text{dedicated, raw}} - \text{Score}_{\text{base, raw}}} \times 100\%$$
     which normalizes each dataset by its own baseline scale and ceiling.

4. **Report Absolute Scores (Not Unstable Retention Ratios) for Base LLMs Under Compression**:
   - For Dedicated Encoders, native scores are high ($0.19\text{--}0.85$), so both absolute scores and retention percentages are well-behaved.
   - For Base LLMs, where raw baselines on `FiQA2018` and `SCIDOCS` are near zero, main-body compression tables should report **absolute scores** (or delta vs native baseline) rather than unstable ratio percentages, while keeping full retention tables in the Appendix with an explicit methodological note on small-denominator amplification.

5. **Map Main-Body Claims Directly to Exhaustive Appendix Tables (A1–A55)**:
   - Keep 5–6 high-density, self-contained tables in the 8-page main body that cover all 6 datasets and all 4 regimes, and cross-reference the 49 granular per-dataset, per-model, per-metric tables in the Appendix so reviewers have both immediate clarity in the main text and 100% full-granularity verification in the Appendix.
