# Decodability is Not Causality: Validating Behavioral Probes with SAE Decomposition

Reproducibility code for **"Decodability is Not Causality: Validating Behavioral Probes with SAE Decomposition."**

Accepted to the COLM 2026 AIMS Workshop, Actionable Interpretability Workshop, and Workshop on Agent Behavior.

**Devesh Tiwari\*, Camille Davis\*, Shivank Sinha, Talia Weaver, Aditya Shah, and Maheep Chaudhary**<br>
Algoverse AI Research<br>
\*Equal contribution

## TL;DR

Probe-aligned SAE features overlap only ~12% with gradient-sensitive ones; ablating shared features flips behavior far more than probe-only or random sets, showing that probe weights alone do not identify causally relevant features.

## Repository contents

| File | Purpose |
|---|---|
| [`TTPD_probe_experiments.ipynb`](TTPD_probe_experiments.ipynb) | Primary analysis: TTPD replication, SAE probe/model attribution, overlap analysis, sparse-mask learning, feature ablations, robustness tests, coherence diagnostics, and appendix analyses. |
| [`reward_hacking_experiments.ipynb`](reward_hacking_experiments.ipynb) | Companion probe/SAE-substrate diagnostic on a Gemma-3-12B reward-hacking probe. It measures how much of the published probe signal survives SAE reconstruction and compares raw-activation with reconstruction-space ablations. |

The TTPD notebook contains the main Gemma-2/TTPD analysis. The reward-hacking notebook is a related diagnostic and should not be treated as a reproduction of the primary case study.

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

1. Open [`TTPD_probe_experiments.ipynb`](TTPD_probe_experiments.ipynb) in Google Colab.
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

| Notebook section | Reproduced output |
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

- Reported results use multi-seed means. Individual notebook runs resample statements and random controls, so exact values can vary slightly with the seed, hardware, and library versions.
- The conclusions should be evaluated from the qualitative and statistical pattern, not a single point estimate: weak probe-model overlap, shared > probe-only > random behavioral effects, and the dissociation between behavior change and probe-readout change.
- The notebooks clone upstream repositories at runtime. Future upstream changes may affect reproducibility; record commit hashes when producing archival results.
- Coherence gating is essential. A behavioral flip is interpretable only when the intervention preserves a valid output and does not merely degrade generation.
