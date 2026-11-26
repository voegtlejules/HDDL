# HDDL project — Data pruning by latent-space representativeness (NLP)
 
Broad course question: **how much data does a deep learning model really need?**
 
## Research question
 
> At a fixed budget of **k training examples per class**, does selecting examples that are
> **representative in a latent space** (text auto-encoder, TSDAE) train a better classifier than
> **stratified random sampling** or **difficulty-based selection** (EL2N, dataset cartography)?
> And does a subset's representativeness **predict** its test performance?
 
### Hypotheses
 
| | Hypothesis | Refuted if |
|---|---|---|
| **H0** | For small budgets (k ≤ 10 / class), k-means **prototypes** in the TSDAE space beat stratified random sampling (paired test over 5 seeds); the advantage vanishes for large k. | No significant gap at k ≤ 10, or a constant / reversed gap at every k. |
 
"Good performance" = test accuracy ≥ 99 % of the full-pool model. Also reported: macro-F1 and worst-class accuracy.
 
## Data
 
**Banking77** (Casanueva et al., 2020): 13,083 online-banking customer queries labelled with 77 fine-grained intents.
Source: the official PolyAI CSV files (fallback: Hugging Face `PolyAI/banking77`).
 
| Split | Size | Notes |
|---|---|---|
| `pool` | 9,233 | official train minus validation. **Selection happens here.** |
| `val` | 770 | 10 per class, stratified, seed 0. Used only to choose hyper-parameters (then frozen). |
| `test` | 3,080 | official test set, untouched until the final evaluation. |
 
- Per class in the pool: min 25, median 117, max 177 → **budgets k ∈ {3, 5, 10, 20, 25}** are feasible for every class, plus `full`.
- Example ids are content hashes (`tr_<sha1>`), and label ids follow the case-insensitive alphabetical order of intent names → identical on every machine, whatever the download source. `results/split_info.json` stores a checksum of the pool ids to check it.
- Known issue: some labels are wrong (Ying & Thomas, 2022) — exploited in the analysis.
## Setup
 
```bash
git clone <repo-url> && cd hddl-data-pruning
python -m venv .venv && source .venv/bin/activate     # Python ≥ 3.10
pip install -r requirements.txt
pip install -e .                                       # makes `import pruning` work from scripts/
```
 
On Apple Silicon, PyTorch uses the `mps` backend automatically; add `export PYTORCH_ENABLE_MPS_FALLBACK=1`.
 
## Quick start — download the data
 
```bash
python scripts/00_prepare_data.py
```
 
Expected output:
 
```
 pool:   9233 examples, 77 classes
  val:    770 examples, 77 classes
 test:   3080 examples, 77 classes
pool per class: min 25, median 117, max 177
```
 
This writes `artifacts/splits/{pool,val,test}.jsonl` and `label_names.json` (git-ignored, regenerable).
Every later step reads them through `pruning.data.load_split("pool" | "val" | "test")`.
Check that your `results/split_info.json` has the same `pool_ids_sha1` as the committed one.
 
## Pipeline
 
| Step | Script | Input → Output | Owner | Status |
|---|---|---|---|---|
| 00 | `scripts/prepare_data.py` | Banking77 → `artifacts/splits/` | C | ✅ |
| 01 | `scripts/train_tsdae.py` | pool text → `artifacts/models/tsdae/` | A | ⬜ |
| 02 | `scripts/compute_embeddings.py` | pool → `artifacts/embeddings/{tsdae,sbert,tfidf}.npy` (+ `.ids.json`) | A | ⬜ |
| 03 | `scripts/compute_difficulty.py` | full-pool runs × seeds → `artifacts/scores/{el2n,cartography}.parquet` | B | ⬜ |
| 04 | `scripts/select.py` | embeddings / scores → `artifacts/subsets/{enc}_{method}_k{k}_s{seed}.json` | B | ⬜ |
| 05 | `scripts/train_eval.py` | subset + `configs/base.yaml` → `results/runs/{run_id}.json` | C | ⬜ |
| — | `scripts/launch_grid.py` | `configs/grid.yaml` → runs 05 in parallel, skips finished runs | C | ⬜ |
| 06 | `scripts/representativeness.py` | subsets + embeddings → MMD / coverage columns | A | ⬜ |
| 07 | `scripts/analyze.py` | `results/runs.csv` → `figures/`, statistical tests | C | ⬜ |
 
Steps 00–04 are short and sequential; step 05 is the long, embarrassingly parallel part
(one independent fine-tuning per `(encoder, method, k, seed)`).
 
### Selection methods (all stratified per class)
 
| Method | Needs | Idea |
|---|---|---|
| `random` | — | baseline |
| `kmeans` | embeddings | k-means with k clusters per class → nearest real example to each centroid |
| `kcenter` | embeddings | greedy farthest-point coverage |
| `moderate` | embeddings | examples whose distance to the class centre is closest to the median |
| `gmm` | embeddings | examples whose location is in a density estimate or is not ? |
| `el2n` | difficulty runs | highest ‖softmax − one-hot‖₂ after 1–2 epochs, averaged over seeds |
| `cartography` | difficulty runs | most "ambiguous" examples (highest variability of p(y\*\|x) across epochs) |
 
### Models
 
| | Downloaded | Trained by us |
|---|---|---|
| `distilbert/distilbert-base-uncased` | starting point of every run | fine-tuned on each subset (the experiment) and on the full pool (difficulty scores) |
| `google-bert/bert-base-uncased` | base of TSDAE | TSDAE encoder, unsupervised, on pool text only |
| `sentence-transformers/all-MiniLM-L6-v2` | inference only | — |
| TF-IDF (`scikit-learn`) | — | fitted on the pool |
| `FacebookAI/roberta-base` (bonus) | starting point | fine-tuned on a few subsets |
 
## Repository layout
 
```
configs/        base.yaml (frozen training hyper-parameters), grid.yaml (experiment grid)
src/pruning/    library code: data, embed/, select/, train, metrics, utils
scripts/        one CLI entry point per pipeline step
tests/          fast checks (k ids per class, no duplicates, determinism)
artifacts/      git-ignored: splits, models, embeddings, scores, subsets
results/        committed: one JSON per run + aggregated runs.csv
figures/        generated by scripts/07_analyze.py only
notebooks/      exploration only, never the source of a reported number
```
 
## Reproducibility rules
 
1. Every number in the report comes from `results/runs.csv`; every figure from `scripts/07_analyze.py`.
2. Each run records `git_commit`, `config_hash`, seed and machine.
3. Hyper-parameters are chosen on `val`, then frozen in `configs/base.yaml`; the test set is never used for any choice.
4. The same training-budget rule applies to every method at a given k.
## Team
 
| | Role |
|---|---|
| A | Representations (TSDAE, SBERT, TF-IDF), MMD & coverage |
| B | Selection methods, EL2N & cartography |
| C | Data, training / evaluation pipeline, grid launcher, statistical analysis |
 
## Key references
 
- Sorscher et al. 2022 — [Beyond neural scaling laws](https://arxiv.org/abs/2206.14486)
- Swayamdipta et al. 2020 — [Dataset Cartography](https://arxiv.org/abs/2009.10795)
- Paul et al. 2021 — [Deep Learning on a Data Diet (EL2N)](https://arxiv.org/abs/2107.07075)
- Fayyaz et al. 2022 — [BERT on a Data Diet](https://arxiv.org/abs/2211.05610)
- Zheng et al. 2023 — [Coverage-centric Coreset Selection](https://arxiv.org/abs/2210.15809)
- Guo et al. 2022 — [DeepCore](https://arxiv.org/abs/2204.08499)
- Wang et al. 2021 — [TSDAE](https://arxiv.org/abs/2104.06979)
- Casanueva et al. 2020 — [Banking77](https://arxiv.org/abs/2003.04807) · Ying & Thomas 2022 — [Label Errors in BANKING77](https://aclanthology.org/2022.insights-1.19)
 