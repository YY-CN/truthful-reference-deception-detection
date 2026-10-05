# Within-person truthful-reference correction

## Overview

This repository implements truthful-reference correction (TR) for participant-independent deception detection. A base classifier supplies a logit; a shared residual adjusts it using distances to one or a few known truthful responses from the same participant.

The workflow uses participant-level cross-fitting, participant-balanced fitting and participant-macro evaluation. Query labels are not used for test-time fitting or scoring. Reference eligibility is supplied as already-known truthful support.

## Method

`z* = z0 + r(q)`

`r(q) = ln(2) * tanh(beta.T @ q_hat / ln(2))`

For K=1, q is the cosine distance to the single reference. For K>1, q contains distance to the normalized reference centroid and the mean distance to individual references. The method is SELF-ONLY. References are excluded from their corresponding query draw. The correction has no intercept, the base coefficient is fixed at 1, and rho=0.01. Residual fitting uses participant-held-out base logits. MLP uses one shared residual across three seed models and averages corrected probabilities.

## Requirements

Python 3.10 is supported. Runtime dependencies are NumPy, SciPy, scikit-learn, PyTorch and PyYAML, pinned in requirements.txt.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

For the three core test files, install the test extra with `pip install -e '.[test]'` and run `python -m pytest -q`. The three scripts below are also available as installed `tr-primary`, `tr-heads` and `tr-fusion` commands.

## Data preparation

Obtain DOLOS through [NTU ROSE Lab's official dataset page](https://rose1.ntu.edu.sg/dataset/DOLOS/), which provides the request process and research-use terms. Obtain MU3D through [Miami University Scholarly Commons](https://sc.lib.miamioh.edu/items/79ac38da-cb8e-4eff-be92-ee8765a5c72c), following the provider's usage-agreement instructions. This repository distributes no videos, audio, transcripts, embeddings or sample-level metadata, and provides no download mirror.

Prepare a metadata CSV with `sample_id,participant_id,label,embedding_row,truthful_reference_eligibility`, plus a floating-point NPY embedding matrix of shape `[N,768]`. Labels are 0=truthful and 1=deceptive. The boolean eligibility field accepts true/false or 1/0. CSV integer columns use decimal integer tokens; fractional or missing values are rejected. IDs and embedding rows must be unique, row indices in range, and vectors finite and L2 normalized. The code joins by explicit embedding_row.

The repository includes the exact participant membership used in the reported formal DOLOS split. See [JSON membership](protocol/dolos_formal_split.json) and [CSV role flags](protocol/dolos_formal_split.csv). These files preserve the frozen base-training 30, evaluation 16 and residual-fitting 7 memberships; they do not reconstruct the original split-generation procedure or seed. The CSV can be passed directly to `--splits` for DOLOS.

Supply a participant split CSV:

- DOLOS: `participant_id,base_train,residual_train,evaluation`, with boolean role flags. The paper uses 30 training participants/197 responses, a 7-participant/50-response residual subset, and 16 evaluation participants/425 responses; K=3.
- MU3D: `participant_id,outer_fold,role`, with folds 0–4 and train/test roles. Each of 80 participants has two truthful and two deceptive responses; each fold has train64/test16, each participant is evaluated once; K=1.

Training/evaluation scopes must be disjoint. Configuration files contain hyperparameters only. All explicit command-line paths resolve relative to the caller's working directory; absolute paths work from any directory. Outputs default to no overwrite; `--overwrite` cannot replace an input file.

## Semantic representations

The paper uses [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base): a 768-dimensional CLS representation followed by L2 normalization. Model revision: `9bbca17d9273fd0d03d5725c7a4b0f6b45142062`; custom-code revision: `40ced75c3017eb27626c9d4ea981bde21a2662f4`.

Obtain the model from its official source and prepare embeddings separately. This repository begins with prepared 768D embeddings and includes no embedding extraction implementation, model snapshot or model weights.

## Run primary experiment

```bash
python scripts/run_primary.py --config configs/dolos.yaml --samples data/dolos/samples.csv --embeddings data/dolos/embeddings.npy --splits data/dolos/participants.csv --output outputs/dolos_lr.json
python scripts/run_primary.py --config configs/mu3d.yaml --samples data/mu3d/samples.csv --embeddings data/mu3d/embeddings.npy --splits data/mu3d/outer_folds.csv --output outputs/mu3d_lr.json
```

LR uses participant-balanced StandardScaler and L2/liblinear regression with C=0.001. Each output JSON contains Base/TR AP and AUROC, paired differences and participant-bootstrap 95% CIs (B=5000, seed=20260722).

## Run robustness heads

```bash
python scripts/run_heads.py --head svm --config configs/dolos.yaml --samples data/dolos/samples.csv --embeddings data/dolos/embeddings.npy --splits data/dolos/participants.csv --output outputs/dolos_svm.json
python scripts/run_heads.py --head mlp --config configs/mu3d.yaml --samples data/mu3d/samples.csv --embeddings data/mu3d/embeddings.npy --splits data/mu3d/outer_folds.csv --output outputs/mu3d_mlp.json
```

SVM uses LinearSVC C=0.001, squared_hinge, dual=auto and held-participant sigmoid calibration. MLP uses 768→128→1/ReLU, AdamW, 30 epochs, batch64, lr=3e-4, weight_decay=1e-4 and seeds 20260722/23/24.

## Run score-level fusion

Visual probabilities must be generated externally. Provide semantic and visual CSVs with `context,sample_id,probability`. Fusion fits participant-balanced StandardScaler plus two-score LR with C=1/liblinear.

Each context needs probabilities for its fitting participants and held/evaluation participants. Supply a JSON object keyed by `dolos` or MU3D outer-fold strings `0`–`4`. Each value contains `inner` records (`context,fit_participants,held_participants`) and a `full` record (`context,fit_participants,evaluation_participants`). Use five inner groups of six held participants for DOLOS, or four groups of sixteen inside each MU3D outer training fold. Inner held groups partition training participants. Score generators must exclude the corresponding held/evaluation participants from their fitting and selection. Flat test/OOF scores without context-specific training probabilities are insufficient.

```bash
python scripts/run_fusion.py --config configs/mu3d.yaml --samples data/mu3d/samples.csv --embeddings data/mu3d/embeddings.npy --splits data/mu3d/outer_folds.csv --semantic-scores data/mu3d/semantic_scores.csv --visual-scores data/mu3d/visual_scores.csv --contexts data/mu3d/fusion_contexts.json --output outputs/mu3d_sf.json
```

## Reported results

| Dataset | Base AP | TR AP | Delta AP |
| --- | --- | --- | --- |
| DOLOS | 0.6957 | 0.7612 | +0.0655 |
| MU3D | 0.8974 | 0.9609 | +0.0635 |

Reported values are provided only as reference results from the paper.

## Third-party resources

DOLOS and MU3D are obtained from their original providers under their respective terms. GTE is obtained from the official model source above. The paper's visual condition uses frozen probabilities generated with external GenLie models ([GenLie paper](https://doi.org/10.1109/ICASSP55912.2026.11464973)). Users supply visual probabilities themselves. No GenLie implementation, checkpoint, or visual embedding is distributed here.

## Citation

Citation information will be updated upon publication.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
