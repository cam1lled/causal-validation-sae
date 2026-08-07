# Decodability is Not Causality: Validating Behavioral Probes with SAE Decomposition

[![COLM 2026](https://img.shields.io/badge/COLM-2026-6f42c1)](https://colmweb.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Open the main experiment in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cam1lled/causal-validation-sae/blob/main/TTPD_probe_experiments.ipynb)

Official experiment code for the COLM 2026 paper **"Decodability is Not Causality: Validating Behavioral Probes with SAE Decomposition."**

**Devesh Tiwari\*, Camille Davis\*, Shivank Sinha, Talia Weaver, Aditya Shah, and Maheep Chaudhary**<br>
Algoverse AI Research<br>
\*Equal contribution

## Overview

Linear probes can predict safety-relevant concepts from language-model activations, but strong probe performance establishes only that a concept is *decodable*. It does not show that the features weighted by the probe are the features that causally drive the model's behavior.

We test that distinction at the level of sparse-autoencoder (SAE) features. Our primary case study applies the TTPD truth probe of [Buerger et al. (2024)](https://arxiv.org/abs/2407.12831) in the instructed truth/deception setting of [Long et al. (2025)](https://aclanthology.org/2025.emnlp-main.826/) using Gemma-2-9B-Instruct. We independently rank SAE features from two perspectives:

1. **Probe alignment:** how strongly each SAE decoder direction aligns with the probe weight vector.
2. **Model attribution:** how sensitive the model's `True`/`False` behavior is to each SAE feature, measured with gradients.
3. **Causal intervention:** what happens to model behavior, probe readout, and output coherence when selected features are ablated.

This separates features into three diagnostic groups:

| Feature group | Probe-ranked | Model-ranked | Interpretation under intervention |
|---|:---:|:---:|---|
| Shared | Yes | Yes | Strongest candidates for behaviorally relevant features |
| Probe-only | Yes | No | Candidate diagnostic readouts or correlates |
| Random control | No | No | Matched baseline for nonspecific intervention effects |

The central lesson is simple: **a probe's weight vector alone does not identify the features the model causally uses.** Probe attribution must be paired with model-side attribution, direct intervention, and a coherence check.

## Main findings

- **The replicated TTPD probe is highly predictive.** It reaches AUROC 0.98 on deceptive-instruction activations from Gemma-2-9B-Instruct.
- **Probe and model rankings overlap only weakly.** Their top feature sets share roughly 11-14% of features, with Spearman rank correlation `rho = 0.10`.
- **Shared features have the largest behavioral effect.** Ablating shared probe-model features flips the model's `True`/`False` output on as many as 27% of statements, compared with 6% for size-matched probe-only features and 1% for random controls.
- **Probe-only features mainly move the probe.** They produce a larger shift in the probe's own readout while causing substantially fewer behavioral flips than shared features.
- **The effect is coherent and robust.** Output coherence remains 100% in the headline ablations, and the shared > probe-only > random ordering holds across five seeds and a held-out split.
- **Activation-aware selection outperforms geometric alignment.** A learned sparse mask flips behavior on `17.6 +/- 0.7%` of statements, nearly three times the geometric probe top-16 (`6.1 +/- 0.5%`), despite sharing only 3 of those 16 features.
- **SAE substrate quality is a prerequisite.** In the paper's auxiliary obfuscation-probe analysis, only 26% of the raw probe margin survives SAE reconstruction. This is a boundary case: SAE-feature interventions should not be interpreted causally unless the SAE basis actually preserves the probe signal.

## Method at a glance

For a linear probe with weight vector `w` and an SAE decoder direction `d_i`, the probe-alignment score for feature `i` is:

```text
c_i = w^T d_i
```

The model-side score is the mean gradient sensitivity of the target-token margin to that feature's activation:

```text
A_i = E_x[d(logit(True) - logit(False)) / d f_i]
```

For a selected feature set `S`, the intervention subtracts its decoded contribution from the residual stream:

```text
h_intervened = h - sum_{i in S}(f_i d_i)
```

We then measure:

- **Behavior flip rate:** whether the sign of the `True` minus `False` logit margin changes.
- **Probe-readout shift:** how much the intervention moves the probe margin.
- **Coherence:** whether the model still emits a valid `True` or `False` response.

## Repository contents

| File | Purpose |
|---|---|
| [`TTPD_probe_experiments.ipynb`](TTPD_probe_experiments.ipynb) | Primary paper reproduction: TTPD replication, SAE probe/model attribution, overlap analysis, sparse-mask learning, feature ablations, robustness tests, coherence diagnostics, and appendix analyses. |
| [`reward_hacking_experiments.ipynb`](reward_hacking_experiments.ipynb) | Companion probe/SAE-substrate diagnostic on a Gemma-3-12B reward-hacking probe. It measures how much of the published probe signal survives SAE reconstruction and compares raw-activation with reconstruction-space ablations. |

The TTPD notebook corresponds to the paper's main Gemma-2/TTPD analysis. The reward-hacking notebook is a related diagnostic and should not be treated as a reproduction of the primary case study.

## Reproducing the main experiment

### Requirements

- A Google Colab runtime with a CUDA GPU and enough memory for `google/gemma-2-9b-it` in bfloat16.
- A Hugging Face account with access to the Gemma model weights.
- A Hugging Face token saved in Colab Secrets as `HF_TOKEN`.
- Google Drive access if you want to cache and restore the extracted activations.

The notebook installs its Python dependencies and downloads the required public code, datasets, model, and SAE artifacts:

- [`ivyllll/truthful-representation-flip`](https://github.com/ivyllll/truthful-representation-flip)
- [`sciai-lab/Truth_is_Universal`](https://github.com/sciai-lab/Truth_is_Universal)
- [`google/gemma-2-9b-it`](https://huggingface.co/google/gemma-2-9b-it)
- Gemma Scope layer-20 residual SAE: `gemma-scope-9b-it-res`, `layer_20/width_16k/average_l0_91`

### Run instructions

1. Open [`TTPD_probe_experiments.ipynb`](TTPD_probe_experiments.ipynb) in Colab using the badge above.
2. Select a GPU runtime.
3. Add `HF_TOKEN` under **Colab > Secrets** and enable notebook access to it.
4. Run Sections 1-3 to clone the upstream repositories, install dependencies, patch current-library compatibility, and extract layer-20 activations for the 12 canonical factual-statement sets.
5. Optionally run Section 4 to reproduce the layer-wise Long et al. probe curves. This section is not required for the causal experiments.
6. Run Sections 5-6 to load Gemma-2-9B-Instruct and its matching Gemma Scope SAE, reconstruct the TTPD direction, and build the probe and model feature rankings.
7. Run the desired cells in Section 7. Once Section 6 is complete, the experiment cells are independent unless a cell states an additional dependency.

The activation cache is written to:

```text
/content/drive/MyDrive/reproduceTTPD/acts_backup.tar.gz
```

If that archive already exists, use the notebook's restore path instead of rerunning activation extraction.

### Experiment map

| Notebook section | Paper output |
|---|---|
| 7.1 | Probe-model overlap curve and Spearman correlation |
| 7.2 | Shared, geometric, mask, and random ablations; Figure 2 and Table 3 |
| 7.3 | Five-seed stability and held-out split |
| 7.4 | Gradient-top comparison in Table 3 |
| 7.5 | Non-monotonic margin-dip diagnostic |
| 7.6 | Behavior-flip exemplars |
| 7.7 | Directional flip breakdown and confusion matrix |
| 7.8 | Per-category appendix table |
| 7.9 | SAE reconstruction and coherence diagnostics |
| 7.10 | Full-vocabulary KL divergence under intervention |
| 7.11 | Full TTPD-direction ablation baseline |

The appendix cells also write `appendix_examples.csv`, `directional_confusion.png`, and `category_breakdown.csv` to the active Colab working directory.

## Running the reward-hacking diagnostic

Open [`reward_hacking_experiments.ipynb`](reward_hacking_experiments.ipynb) in Colab and edit the configuration before running it. The notebook expects:

- five activation shards named `eval_000.pt` through `eval_004.pt`;
- 1,000 total examples, ordered with deceptive examples first and honest examples second;
- `cfg.shard_dir` set to the local or Google Drive directory containing those shards.

It downloads the public `AlignmentResearch/diverse-deception-probe-gemma-3-12b-it` checkpoint and the matching Gemma Scope 2 layer-31 SAE. Its outputs include SAE round-trip diagnostics, feature rankings, raw-versus-reconstruction ablation sweeps, sparse-mask results, compact comparison tables, and publication-style figures under:

```text
/content/farai_gemma3_sae_mask_diagnostics/
```

The activation shards are not included in this repository.

## Reproducibility notes

- The paper reports multi-seed means. Individual notebook runs resample statements and random controls, so exact values can vary slightly with the seed, hardware, and library versions.
- The conclusions should be evaluated from the qualitative and statistical pattern, not a single point estimate: weak probe-model overlap, shared > probe-only > random behavioral effects, and the dissociation between behavior change and probe-readout change.
- The notebooks clone upstream repositories at runtime. Future upstream changes may affect reproducibility; record commit hashes when producing archival results.
- Coherence gating is essential. A behavioral flip is interpretable only when the intervention preserves a valid output and does not merely degrade generation.

## Limitations

The primary causal analysis studies one model, layer, SAE, probe, and factual-verification setting. The interventions establish behavioral relevance in this setting; they do not demonstrate a universal mechanism for deception. Generalization across model families, SAE architectures, layers, probe types, and broader behavioral domains remains open.

## Citation

If this repository is useful in your work, please cite:

```bibtex
@inproceedings{tiwari2026decodability,
  title     = {Decodability is Not Causality: Validating Behavioral Probes with SAE Decomposition},
  author    = {Tiwari, Devesh and Davis, Camille and Sinha, Shivank and Weaver, Talia and Shah, Aditya and Chaudhary, Maheep},
  booktitle = {Conference on Language Modeling},
  year      = {2026}
}
```

## Acknowledgments

This work builds on the TTPD truth probe from Buerger et al. (2024), the instructed truth/deception setting and released artifacts from Long et al. (2025), Gemma Scope SAEs, and the public deception probes released with [The Obfuscation Atlas](https://arxiv.org/abs/2602.15515).
