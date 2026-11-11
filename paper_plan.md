# ACL 2027 Paper Plan: Crowding-Orthogonal Adaptive Spectral Tempering (COAST)

Target repository: `/Users/devanshlodha/orca/spectral-postprocessing/`
Source code and tables repository: `/Users/devanshlodha/orca/cs-613-assignment-2-approach-2/`
Target venue: ACL 2027 Main Conference (Long paper, up to 8 pages main body plus unnumbered Limitations, Ethical Considerations, References, and Appendix, compiled with `latexmk -lualatex`)

---

## 1. Title and Core Logic

### Candidate Titles
1. Crowding-Orthogonal Adaptive Spectral Tempering for Training-Free Text Embedding Post-Processing and Compression
2. Unifying Subspace Deflation and Spectral Tempering Across Contrastive and Autoregressive Language Models
3. From Cone Crowding to Spectral Tempering: A Parameter-Free Operator for Text Embedding Post-Processing and Compression

### Logical Progression of the Paper
1. Unit-normalized text embeddings $x \in \mathbb{S}^{d-1}$ satisfy an exact three-part energy partition:
   $$1 = \frac{1}{N}\sum_{i=1}^N \|x_i\|_2^2 = E_\parallel + \frac{N - 1}{N}\big(\sigma_\parallel^2 + E_\perp\big)$$
   where $E_\parallel = \|\mu\|_2^2$ is the squared norm of the mean vector $\mu$ along $\hat{\mu} = \mu / \|\mu\|_2$, $\sigma_\parallel^2 = \operatorname{Var}(x^\top \hat{\mu})$ is the variance along $\hat{\mu}$, and $E_\perp = \operatorname{Tr}(\Sigma_\perp)$ is the total variance in the $(d-1)$-dimensional subspace orthogonal to $\hat{\mu}$.
2. This partition separates pretrained models into two geometric regimes across all 36 model-dataset-pooling configurations:
   - Contrastively trained embedding models spread vectors across the sphere, so orthogonal variance exceeds mean energy ($E_\parallel < E_\perp$, with $E_\parallel / E_\perp \in [0.18, 0.60]$).
   - Un-fine-tuned causal language models pack vectors into a narrow cone around $\hat{\mu}$, so mean energy dominates orthogonal variance ($E_\parallel \gtrsim E_\perp$, with $E_\parallel / E_\perp \in [0.80, 6.04]$).
3. Existing training-free post-processing methods fail across regimes because of three mathematical issues:
   - Standard centered covariance $\Sigma = \operatorname{cov}(X - \mu)$ does not have $\hat{\mu}$ as an eigenvector. Variations along $\hat{\mu}$ leak into small-eigenvalue tail eigenvectors of $\Sigma$ ($\|U_{\text{tail}}^\top \hat{\mu}\|_2 \in [0.37, 0.97]$) and are amplified during spectral scaling.
   - Fixed-regularizer whitening (`Soft-ZCA` with $\gamma = 1, \epsilon = 10^{-4}$; `Whitening-k` with $\gamma = 1, \epsilon = 10^{-6}$) sets $\epsilon$ far below the spectral knee $\lambda_{\perp, k_{\text{knee}}} \approx 4 \times 10^{-3}$ on dedicated encoders. This amplifies tail noise by up to $100\times$ and degrades dedicated encoders on 17 of 18 full-dimension benchmarks and 41 of 48 compression benchmarks.
   - SNR-guided tempering (`SpecTemp` with $\gamma = \gamma_{\text{SNR}}(k), \epsilon = 10^{-6}$) drives $\gamma_{\text{SNR}}(k) \to 0$ at $k \in \{512, d\}$. This turns off whitening on crowded base language models (losing 24 of 24 evaluations at $k = 512$) and fails when sample size $N < k$ because zero eigenvalues are scaled by $\epsilon^{-1/2} = 10^3$.
4. Crowding-Orthogonal Adaptive Spectral Tempering (`COAST`) resolves all three issues without tuned hyperparameters:
   - It projects vectors onto the subspace orthogonal to $\hat{\mu}$ via $x_\perp = x - (x^\top \hat{\mu})\hat{\mu}$ and decomposes $\Sigma_\perp = \operatorname{cov}(X_\perp) = U_\perp \Lambda_\perp U_\perp^\top$, eliminating mean-axis tail leakage and canceling parallel sample-mean estimation error to first order.
   - It defines the crowding ratio $\gamma_{\text{crowd}} = \min\big(1, (E_\parallel / E_\perp)^2\big)$ to floor the tempering exponent $\gamma^*(k) = \max\big(\gamma_{\text{crowd}}, \gamma_{\text{SNR}}(k)\big)$.
   - It interpolates the spectral regularizer between the spectral knee $\lambda_{\perp, k_{\text{knee}}}$ and the subspace-scaled dispersion variance $\tau(k)^2 E_\perp \bar{\lambda}_\perp$, where $\tau(k) = \min\big(1, k / \min(N - 1, d - 1)\big)$ is the retained fraction of the sample covariance rank.

---

## 2. Section-by-Section Plan and 8-Page Budget

### Abstract (`sections/00_abstract.tex`, under 180 words)
- State the cross-regime trade-off in training-free embedding post-processing and compression: methods that fix anisotropy in base language models degrade calibrated embedding models, and vice versa.
- Introduce the orthogonal-complement spectral operator and the `COAST` equations governed by the unit-sphere energy partition $E_\parallel$ and $E_\perp$.
- Report exact empirical results across 4 models, 2 pooling strategies, 6 MTEB v2 retrieval and similarity benchmarks, 17 main transformation variants, and 44 factorial ablation variants (44,010 metric evaluations).

### Section 1: Introduction (`sections/01_introduction.tex`, ~1.0 page)
- Explain how cosine similarity on $\mathbb{S}^{d-1}$ degrades when embeddings have a large mean shift $\mu$ and skewed covariance eigenvalues.
- Describe why current methods work in only one regime:
  - Rank-1 to rank-3 removal (`R1`, `R2`, `ABTT-D`) leaves covariance anisotropy intact on base language models.
  - Full whitening (`Soft-ZCA`, `Whitening-k`) fixes base language models (`24/24` wins over baseline) but degrades dedicated encoders (`17/18` losses at $k = d$, `41/48` losses across $k < d$) and rank-deficient datasets ($N < k$).
  - SNR tempering (`SpecTemp`) preserves dedicated encoders at $k = 512$ by decaying $\gamma_{\text{SNR}}(k) \to 0$, which disables whitening on base language models (`24/24` losses vs `Whitening` at $k = 512$).
- Summarize how `COAST` unifies full-dimension calibration and dimensional compression using the energy ratio $E_\parallel / E_\perp$ and subspace rank fraction $\tau(k)$.
- List the four contributions in plain bullets:
  - A unified spectral operator formulation and proofs of three failure modes in prior methods: mean-axis tail leakage, under-regularized tail amplification, and SNR decay on crowded cones.
  - The parameter-free `COAST` operator derived from the unit-sphere energy partition and subspace rank scaling.
  - A comprehensive evaluation across 4 models, 6 MTEB v2 datasets, 17 main post-processing and compression methods, 3 negative controls, and a 44-state factorial ablation lattice across 5 task metrics and 5 geometry metrics.
  - An empirical analysis of sequence-length pooling collapse on causal language models ($10$ to $638$ average tokens) and sample-mean error cancellation (`R2` vs `R1`).

### Section 2: Related Work (`sections/02_related_work.tex`, ~0.75 page)
- Group prior work into three logical threads:
  - Representation anisotropy and outlier coordinates (Ethayarajh, 2019; Bis et al., 2021; Timkey and van Schijndel, 2021; Rudman et al., 2022; Yokoi et al., 2024).
  - Subspace deflation and mean-bias correction (Arora et al., 2017; Mu and Viswanath, 2018; Rajaee and Pilehvar, 2021; Ren et al., 2026).
  - Spectral whitening and dimensionality reduction (Su et al., 2021; Huang et al., 2021; Kusupati et al., 2022; Forooghi et al., 2024; Diera et al., 2024; Li et al., 2026).

### Section 3: Methodology (`sections/03_methodology.tex`, ~1.75 pages)

#### 3.1 Unified Spectral Operator
- Write the general operator mapping $x \in \mathbb{S}^{d-1}$ to $\mathbb{S}^{k-1}$:
  $$f(x; k) = \phi\!\left(\big(\Lambda_k + \epsilon I_k\big)^{-\gamma / 2} U_k^\top g_{\text{center}}(x)\right), \qquad \phi(z) = \frac{z}{\|z\|_2}$$
- Include **Table 1 (`table*`, top of Page 3): Mathematical Specification of Post-Processing and Compression Operators**:
  - Columns: `Method`, `Centering g(x)`, `Covariance Matrix`, `Target Dim k`, `Regularizer \epsilon`, `Exponent \gamma(k)`.
  - Rows: `Baseline`, `Standardization`, `R1`, `R2`, `ABTT-D`, `Prefix (MRL)`, `PCA-k`, `Whitening-k`, `Soft-ZCA`, `SpecTemp`, and `COAST`.

#### 3.2 Failure Modes of Existing Operators
- **Proposition 1 (Mean-Axis Leakage in Centered Covariance and First-Order Error Cancellation)**:
  - Show that in centered covariance $\Sigma = \operatorname{cov}(X - \mu) = U \Lambda U^\top$, the residual $(x^\top \hat{\mu} - \|\mu\|_2)\hat{\mu}$ projects onto every eigenvector $u_j$ with variance $(\hat{\mu}^\top u_j)^2 \sigma_\parallel^2$.
  - Show that projecting onto $\mathcal{H}_\perp = \operatorname{span}(\hat{\mu})^\perp$ via $P_\perp = I_d - \hat{\mu}\hat{\mu}^\top$ and decomposing $\Sigma_\perp = \operatorname{cov}(P_\perp X) = U_\perp \Lambda_\perp U_\perp^\top$ guarantees $\Sigma_\perp \hat{\mu} = 0$, eliminates mean-axis leakage into the tail eigenvectors, weights each sample by $\sin^2\theta_i = 1 - (x_i^\top \hat{\mu})^2$, and cancels parallel sample-mean error $\Delta\mu_\parallel = \alpha\hat{\mu}$ to first order.
- **Proposition 2 (Tail Amplification Bound)**:
  - Bound the relative amplification of component $k$ over component $1$ under $w_j = (\lambda_j + \epsilon)^{-\gamma / 2}$:
    $$\frac{w_k}{w_1} = \left(\frac{\lambda_1 + \epsilon}{\lambda_k + \epsilon}\right)^{\gamma / 2} \le \left(1 + \frac{\lambda_1}{\epsilon}\right)^{\gamma / 2}$$
  - Explain how fixed $\epsilon = 10^{-4}$ (`Soft-ZCA`) and $\epsilon = 10^{-6}$ (`Whitening-k`, `SpecTemp`) amplify noise directions when $\epsilon \ll \lambda_{\text{knee}}$ or when $N - 1 < k$ ($\lambda_k = 0$).
- **SNR Decay at High Target Dimensions**:
  - Show why $\gamma_{\text{SNR}}(k) = \min\big(1, \mathrm{SNR}(k) / \mathrm{SNR}(k_{\text{knee}})\big) \to 0$ as $k \to 512, d$, disabling whitening on crowded base language models.

#### 3.3 Unit-Sphere Energy Partition and COAST
- Derive the exact three-term Pythagorean partition on $\mathbb{S}^{d-1}$:
  $$1 = E_\parallel + \frac{N - 1}{N}\big(\sigma_\parallel^2 + E_\perp\big), \qquad E_\parallel = \|\mu\|_2^2, \quad \sigma_\parallel^2 = \operatorname{Var}(x^\top \hat{\mu}), \quad E_\perp = \operatorname{Tr}(\Sigma_\perp) = \sum_{j=1}^{d-1} \lambda_{\perp, j}$$
  with mean orthogonal eigenvalue $\bar{\lambda}_\perp = \frac{E_\perp}{d - 1}$ and effective sample rank $r = \min(N - 1, d - 1)$.
- Define the crowding parameter $\gamma_{\text{crowd}}$ and subspace rank fraction $\tau(k)$:
  $$\gamma_{\text{crowd}} = \min\!\left(1,\, \left(\frac{E_\parallel}{E_\perp}\right)^2\right), \qquad \tau(k) = \min\!\left(1,\, \frac{k}{\min(N - 1,\, d - 1)}\right)$$
- Define the `COAST` operator:
  $$f_{\text{COAST}}(x; k) = \phi\!\left(\big(\Lambda_{\perp, k} + \epsilon^*(k) I_k\big)^{-\gamma^*(k) / 2} U_{\perp, k}^\top \big(x - (x^\top \hat{\mu})\hat{\mu}\big)\right)$$
  $$\epsilon^*(k) = \lambda_{\perp, k_{\text{knee}}}^{\,1 - \gamma_{\text{crowd}}} \Big(\tau(k)^2 E_\perp \bar{\lambda}_\perp\Big)^{\gamma_{\text{crowd}}}, \qquad \gamma^*(k) = \max\!\left(\gamma_{\text{crowd}},\, \min\!\left(1,\, \frac{\mathrm{SNR}(k)}{\mathrm{SNR}(k_{\text{knee}})}\right)\right)$$
- **Proposition 3 (Limiting Regimes of `COAST`)**:
  - State the four limiting cases: calibrated encoder at large $k$ ($\gamma_{\text{crowd}} \ll 1, k \gg k_{\text{knee}}$), calibrated encoder at small $k$ ($\gamma_{\text{crowd}} \ll 1, k \to k_{\text{knee}}$), crowded base model ($\gamma_{\text{crowd}} = 1$ across all $k$), and small-sample rank deficiency ($N - 1 \le k \implies \tau(k) = 1$).
- Include **Figure 1 (`figure*`, top of Page 4): Unit-Sphere Energy Partition and Spectral Scaling Curves**:
  - Panel (a): Plot of $E_\perp$ vs $E_\parallel$ across all 36 configurations showing the line $E_\parallel + E_\perp = 1$ and the boundary $E_\parallel = E_\perp$ separating dedicated encoders from base language models.
  - Panel (b): Normalized spectral weights $w_j / w_1$ across eigenvalue index $j$ for `PCA`, `Whitening`, `Soft-ZCA`, `SpecTemp`, and `COAST` on dedicated vs base models.

### Section 4: Experimental Setup (`sections/04_experimental_setup.tex`, ~1.0 page)
- **4.1 Models and Benchmark Datasets**:
  - Include **Table 2 (`table*`, top of Page 5): Evaluated Models and MTEB v2 Datasets** (parameters, native dimensions, context length, corpus/query counts, pair counts, and average token lengths).
- **4.2 Calibration Protocol, Negative Controls, and Complexity**:
  - Corpus-only calibration in asymmetric retrieval ($\mu_D, \Sigma_{\perp, D}$ fitted on documents $D$ and applied to queries $Q$ without query leakage) and pooled calibration in symmetric similarity.
  - Offline calibration complexity $O(N d^2 + d^3)$ ($< 0.15\text{s}$ per dataset) and online query complexity $O(d k)$.
  - Negative controls: `rand` (random unit vector deflation), `mc` (unnormalized mean centering), and `random_truncation` (random coordinate selection).
- **4.3 Evaluation Metrics**:
  - Task metrics: `nDCG@10`, `MRR@10`, `Recall@100` for retrieval; `Spearman` $\rho \times 100$ and `Pearson` $r \times 100$ for similarity.
  - Geometry metrics: `centroid_norm`, `average_cosine`, `mev_top1`, `nid_spectral_entropy`, and `isoscore`.

### Section 5: Results and Analysis (`sections/05_results_and_analysis.tex`, ~3.0 pages)
All tables report each dataset and each model tier (`Dedicated`, `Base Mean`, `Base Last`) separately without cross-dataset or cross-tier averaging.

#### 5.1 Spectral Regime Separation and Mean-Leakage Verification
- Include **Table 3 (`table*`, Page 5): Spectral Energy Partition and Mean-Leakage Diagnostics Across Datasets and Model Tiers** (summarizing Table A43).
- Verify three empirical properties:
  - The three-term identity $E_\parallel + \frac{N-1}{N}(\sigma_\parallel^2 + E_\perp) = 1$ holds across all 36 configurations.
  - Standard centered covariance $\Sigma$ exhibits large tail leakage $\|U_{\text{tail}}^\top \hat{\mu}\|_2 \in [0.37, 0.97]$, whereas $\Sigma_\perp$ has zero leakage.
  - Every dedicated encoder satisfies $E_\parallel < E_\perp$ ($E_\parallel / E_\perp \in [0.18, 0.60]$), while base language models satisfy $E_\parallel / E_\perp \in [0.80, 6.04]$ (with 23 of 24 above $1.18$).

#### 5.2 Full-Dimension Evaluation ($k = d$)
- Include **Table 4 (`table*`, Page 6): Full-Dimension Retrieval (`nDCG@10`) and Similarity (`Spearman` $\rho$) Across All 6 Benchmarks and 3 Model Tiers** (summarizing Tables A1–A15).
- Analyze results across regimes:
  - On base language models, `COAST` beats `Soft-ZCA` in 18 of 24 evaluations (`75.0%`) and improves over `Baseline` and `R2` in 24 of 24 evaluations.
  - On dedicated encoders, `COAST` beats `Soft-ZCA` in 17 of 18 evaluations (`94.4%`), achieves the highest overall score on `FiQA2018` (`0.4748`) and `STSBenchmark` (`85.23`), and avoids the `-9.65` point drop of `Soft-ZCA` on `SICK-R` (`79.36` vs `71.02`).
  - Compare `R2` vs `R1` (`29/36 = 80.6%` wins for `R2`, including `16/18` on smaller-sample STS tasks) and verify the `rand` and `mc` negative controls.

#### 5.3 Dimensional Compression Evaluation ($k \in \{512, 256, 128, 64\}$)
- Include **Table 5 (`table*`, Page 7): Dimensional Compression Performance and Retention at $k = 512$ and $k = 64$** (summarizing Tables A16–A30).
- Include **Figure 2 (`figure*`, Page 7): Compression Retention Curves Across $k \in \{512, 256, 128, 64\}$ and Token Pooling Collapse Across Sequence Length**.
- Analyze three compression findings:
  - On base language models across $k \in \{512, 256, 128, 64\}$, `COAST` beats `SpecTemp` in 94 of 96 evaluations (`97.9%`) and beats or matches `Whitening` in 71 of 96 evaluations (`74.0%`).
  - On dedicated encoders, `COAST` beats `Whitening` in 41 of 48 evaluations (`85.4%`) and outperforms native Matryoshka (`Prefix`) slicing in 10 of 12 evaluations (`83.3%`) at $k = 64$ (improving `FiQA2018` retention from `70.2%` to `78.2%` and `SCIDOCS` retention from `62.6%` to `76.4%`).
  - Under small-sample rank deficiency (`STS22.v2`, $N = 394 < 512$), `Whitening` drops to `52.1%` Pearson retention at $k = 512$, while `COAST` maintains `99.9%` on dedicated encoders and `132.1%` on base mean pooling.

#### 5.4 Factorial Component Ablations
- Include **Table 6 (`table`, Page 8): Factorial Ablation of `COAST` Components at $k = d$ and $k \in \{512, 64\}$** (summarizing Suite 6 Tables A44–A55).
- Trace the effect of each factor:
  - Orthogonal basis ($O = 1$ vs $O = 0$): improves retrieval and similarity at high compression and under small sample sizes by eliminating mean-axis tail leakage.
  - Crowding floor ($C = 1$ vs $C = 0$ and fixed $\gamma = 0.5$): prevents the `-15` Spearman point and `-68%` retrieval collapse of `SpecTemp` ($C = 0$) on base language models at $k = 512$, and beats fixed $\gamma = 0.5$ across both tiers.
  - SNR schedule ($C = 1$ vs $C = 2$): provides the progressive whitening boost needed by dedicated encoders at $k = 128$ and $k = 64$.
  - Adaptive regularizer and subspace rank scaling ($E_\tau = 2$ vs $E_\tau = 1$ and $E_\tau = 0$): prevents tail explosion on dedicated encoders and rank-deficient `STS22.v2` ($E_\tau = 0$), while $\tau(k)^2$ ($E_\tau = 2$ vs $E_\tau = 1$) increases base model compression wins vs `Whitening` from `46/96` to `71/96`.

#### 5.5 Intrinsic Space Geometry and Causal Token Pooling Dynamics
- Include **Table 7 (`table`, Page 8): Intrinsic Geometry Metrics Across Representative Transformations** (summarizing Suite 4 Tables A31–A42).
- Explain two geometric findings:
  - `COAST` reduces centroid norm ($\|\mu\|_2 < 0.02$) and average cosine ($\approx 0.00$) across all models while matching the degree of spectral equalization (`NID`, `IsoScore`) to each regime.
  - On base causal language models, `last_token_pooling` outperforms `mean_pooling` by `+2.11` Spearman points on 10-token `SICK-R`, matches within `-0.97` on 12-token `STSBenchmark`, drops by `64%` to `82%` on 174-to-226-token retrieval corpora, and collapses by `-45.33` Spearman points on 638-token `STS22.v2` due to autoregressive recency dilution.

### Section 6: Conclusion (`sections/06_conclusion.tex`, ~0.4 page)
- Summarize the theoretical and empirical results and provide direct recommendations for applying `COAST` in full-dimension and compressed retrieval and similarity pipelines.

### Unnumbered Post-Conclusion Sections (Pages 9+)
- **Limitations (`sections/07_limitations.tex`)**:
  - Linear post-processing closes `7.8%` to `28.5%` of the retrieval gap and `37.7%` to `54.5%` of the similarity gap between base and dedicated models, so it complements rather than replaces contrastive fine-tuning.
  - When sample size $N < d$ (`STS22.v2`, $N = 394$), sample covariance estimation noise causes a small drop (`-1.22` Spearman points) on calibrated dedicated encoders at $k = d$.
  - On `SICK-R`, `ABTT` achieves higher linear Pearson correlation (`63.75`) than spectral whitening (`59.98` Spearman, `56.20` Pearson) because rank correlation is invariant to monotonic tail compression whereas linear Pearson correlation is not.
  - Experiments are scoped to sub-1B parameter models and English MTEB v2 benchmarks.
- **Ethical Considerations (`sections/08_ethics_statement.tex`)**:
  - Dataset and model licenses, compute savings of training-free calibration, and potential effects of linear projection on demographic subspace associations.
- **Acknowledgments (`sections/09_acknowledgments.tex`)**:
  - Guarded by `\ifacl@anonymize\else ... \fi`.

---

## 3. Complete 55-Table Appendix Plan (`sections/10_appendix.tex`)

All 55 non-redundant CSV tables in `tables/` map 1-to-1 to Appendix Tables A1–A55:
- **Appendix A**: Complete proofs of Propositions 1, 2, and 3.
- **Appendix B (`Tables A1–A9`)**: Suite 1 full-dimension retrieval tables (`FiQA2018`, `ArguAna`, `SCIDOCS` across `nDCG@10`, `MRR@10`, `Recall@100`).
- **Appendix C (`Tables A10–A15`)**: Suite 2 full-dimension similarity tables (`STSBenchmark`, `SICK-R`, `STS22.v2` across `Spearman`, `Pearson`).
- **Appendix D (`Tables A16–A30`)**: Suite 3 dimensional compression tables (`9` retrieval + `6` similarity across $k \in \{512, 256, 128, 64\}$).
- **Appendix E (`Tables A31–A42`)**: Suite 4 intrinsic geometry tables (`6` full-dimension + `6` compression across all 5 geometry metrics).
- **Appendix F (`Table A43`)**: Suite 5 empirical spectral and crowding parameter table (`36` rows $\times$ `33` columns).
- **Appendix G (`Tables A44–A55`)**: Suite 6 complete factorial ablation tables (`3` full retrieval + `2` full similarity + `5` compression + `2` geometry).
