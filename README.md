# FairSkew-GCN

Reproducible PyTorch research implementation of **FairSkew-GCN**, a fairness-aware recommender built on sparse LightGCN and evaluated on MovieLens-1M. The default experiment runs exactly 150 epochs and reports full-catalog NDCG/Recall plus gender-specific and Jensen-Shannon fairness metrics at K=10,20,30.

## Setup and runs

```powershell
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/train.py --config configs/movielens1m.yaml
python scripts/evaluate.py --checkpoint outputs/<run>/checkpoints/final.pt --config configs/movielens1m.yaml
```

The download script retrieves GroupLens ML-1M into `data/raw/`; its parsed deterministic split is cached in `data/processed/`. Ratings at least `rating_threshold` (default `4.0`) are implicit positives. For every user with at least three positive interactions, timestamp-ordered interactions are divided 80/10/10 into train/validation/test. Thus no validation/test interaction is a training edge. Validation removes train items from candidates; test removes train and validation items.

## Model

LightGCN has trainable user/item embeddings and applies `E^(l+1)=D^-1/2 A D^-1/2 E^(l)` for `num_layers`, averaging layer 0 through L without nonlinearities.

**A — skew-aware views.** For train interactions, item male/female counts define `p_i=(m_i+a)/(m_i+f_i+2a)` and `s_i=|logit(p_i)-logit(p_bar)|`. Scores are max-normalized. An edge gets dropped with `rho*s_hat_i` only when its user's gender is that item's dominant gender. Two independently sampled views are generated once per training epoch, never per minibatch. They only affect message passing; the original complete train interactions remain BPR positives. The default inference graph is the original full train graph, so a user's gender is not required at inference.

**B — consistency.** A random, configurable set of users is selected each epoch. Its two view embeddings form an InfoNCE matrix: the same user is the diagonal positive and all other sampled users are in-batch negatives. Rows are L2-normalized before division by `contrastive_temperature`.

**C — Group DRO.** BPR is averaged separately for known M/F users. Detached group losses update log weights using `log(w_g)+=eta*L_g`, then log-softmax normalization. The differentiable recommendation term is the new weighted sum. Unknown/missing gender values retain graph/recommendation participation but are excluded from gender-weighted loss and group reporting.

The total objective is `group-weighted BPR + contrastive_coefficient * InfoNCE + regularization * (||U||² + ||I||²)`. Optimizer weight decay is intentionally not used, avoiding duplicate regularization.

## Metrics

NDCG and Recall are per-user averaged top-K metrics over held-out positive items, then averaged overall and by gender. Scoring is full-catalog and known items are removed before ranking.

The request did not include the referenced proposal's precise JS-DP/JS-EO formulas, so this repository makes its operationalization explicit rather than claiming paper equivalence:

- **JS-DP:** Jensen-Shannon divergence (natural-log, `[0, log 2]`) between M and F normalized per-item top-K exposure distributions.
- **JS-EO:** the same divergence using only recommendations that are held-out relevant items (true-positive item exposure), i.e. exposure parity conditional on relevance.

Epsilon smoothing handles zero counts. Both metrics are calculated from actual recommended lists and saved with the result table. Replace `src/evaluation/fairness_metrics.py` if the proposal supplies another definition.

Set `fairness_metric_variant: distance` to reproduce the supplied FairIB/UGRec repositories' reporting convention: the square root of JS (Jensen--Shannon distance, as returned by SciPy's `jensenshannon`). The default `divergence` reports JS itself. This changes only the displayed scale, never the recommendations or the underlying exposure distributions.

## Configuration, artifacts, and resume

`configs/default.yaml` holds all tunables, including `epochs: 150`, device auto-selection, graph-drop strength, contrastive controls, split, and evaluation Ks. Each run gets `outputs/<experiment>-<timestamp>-seed<seed>/` containing the copied configuration, `training.log`, `checkpoints/{best,epoch-*,final}.pt`, and `results/{results.csv,results.json}`. Resume without resetting optimiser or Group DRO state:

```powershell
python scripts/train.py --config configs/movielens1m.yaml --resume outputs/<run>/checkpoints/epoch-25.pt
```

Intermediate selection is validation-only (`validation_interval`, default 10); early stopping is absent so training completes the configured 150 epochs.

## Validation-only tuning

Run configuration-only screening without ever ranking the test set:

```powershell
python scripts/tune.py --config configs/movielens1m.yaml --epochs 30
```

Each candidate writes its training log and checkpoints, records validation metrics in `outputs/tuning-*.json`, and exits before final-test evaluation. `configs/fairskew_selected.yaml` is the selected configuration used for the final 150-epoch run and reports JS distance for comparison with the previous repositories.

## Exposure fairness controls

`configs/fairskew_fairslate.yaml` enables the final direct exposure module. During training, sharp soft exposure distributions are aligned across genders (soft JS-DP), while calibrated mass on each user's training-positive items supplies a soft JS-EO proxy. Neither reads validation or test interactions. At ranking time, a global score-only slate optimizer allocates each gender's Top-K slots toward the same pooled high-utility exposure target. `rerank_strength` is bounded in `[0, 1]` and controls the quality/fairness trade-off; it must be selected on validation only.

`configs/fairskew_literature.yaml` adds a FairGo-inspired adversary to FairSkew-GCN. A discriminator receives both a user's propagated embedding and mean first-order item-neighborhood embedding; it learns to predict gender while LightGCN is trained to make this prediction difficult. The discriminator is optimized separately, preventing the degenerate "weak adversary" outcome. This complements—rather than replaces—the proposed skew augmentation, contrastive consistency, and Group DRO modules.

`configs/fairskew_common_support.yaml` adds an explicit group-aware batch-ranking policy. Its target is the geometric common support of the groups' candidate availability, predicted utility, and (when enabled) training-positive item distributions. This directly discourages group-exclusive exposure, the main source of high item-distribution JS-DP/JS-EO. It uses protected-group labels at ranking time and must therefore only be used where that deployment policy is acceptable. It never reads validation or test relevance; choose its strength on validation only.

The optional `rerank_eo_proxy_strength` is distinct from the shared-exposure term: it estimates each group's item relevance from smoothed **training** counts and counterbalances recommendation exposure by that propensity. This makes the estimated true-positive exposure (`recommendation exposure × training relevance propensity`) approach a common target. It is a proxy, so it is enabled only after validation selection and never fitted to held-out relevance.

Use `python scripts/tune_common_support.py --checkpoint <checkpoint> --config configs/fairskew_common_support.yaml` to select the ranking strength. The selector evaluates **only validation** and minimizes the mean of JS-DP@30 and JS-EO@30 subject to retaining 85% of the un-reranked validation NDCG@30. Its JSON record makes that rule and all candidates auditable.

The 150-epoch selected policy is `configs/fairskew_fairness_selected.yaml`. It retains the direct training-time DP/EO losses and disables the optional post-ranking policies because validation showed that they lowered DP but raised EO. `fairskew_eo_trial.yaml` and `fairskew_eo_final.yaml` are retained only as auditable rejected trials.

### Literature-informed refinements

The ego-graph adversary is adapted from [FairGo](https://www.microsoft.com/en-us/research/wp-content/uploads/2021/06/WWW2021_Learning-Fair-Representations-for-Recommendation-A-Graph-based-Perspective.pdf), which distinguishes sensitive leakage through both node embeddings and local graph structure. The decision to preserve graph augmentation while protecting utility is informed by [FA4GCF](https://github.com/jackmedda/FA4GCF), which targets fair graph augmentation for GNN recommenders, and the representation-plus-structure perspective in [FairDDA](https://arxiv.org/abs/2508.19547). These are implementation inspirations; FairSkew-GCN's proposed Modules A--C and JS metric definitions remain its own documented design.

## Baselines and extensions

For a standard LightGCN baseline, run `python scripts/run_ablation.py --config configs/lightgcn_baseline.yaml`. That YAML disables `enable_augmentation`, `enable_contrastive`, and `enable_group_dro`; the same backbone, split, and evaluator are then used. Create related ablations by changing just those flags.

Dataset adapters produce `RecommendationData` in `src/datasets/base_dataset.py`: contiguous user/item indices, train pairs, held-out maps, ID mappings, and optional sensitive labels. Add a source-specific loader under `src/datasets/`; no model changes are necessary. `MovieLens1MDataset` is intentionally isolated from model code. The skew formula itself is binary M/F as proposed; a nonbinary sensitive attribute needs a documented generalization of its dominant-group/drop rule before enabling fairness modules.

## Limitations

The default CPU implementation uses sparse PyTorch COO propagation and performs full-catalog evaluation; ML-1M can take substantial time without CUDA. Deterministic seeds are set, but exact bitwise equality can still vary across PyTorch/CUDA releases. MovieLens gender is a binary self-reported field and should not be interpreted as a complete model of gender or fairness.
