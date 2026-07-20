# Penguin Species Prediction with TabFM

A small demo of [TabFM](https://github.com/google-research/tabfm) (Google's
Tabular Foundation Model) that predicts penguin species in the classic
[Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/) dataset and
reports how well it does.

TabFM is a pretrained transformer for tabular data that works by **in-context
learning**: calling `fit()` does no gradient training — it simply stores your
training rows as context, and at prediction time the model is shown that
context and asked to label the new rows. That makes it strong on small
datasets (like this one: 344 penguins) with zero tuning.

## Files

| File | Purpose |
|------|---------|
| `predict_species.py` | Runs repeated random train/test splits, predicts species, reports accuracy and a pooled confusion-matrix report |
| `penguins.csv` | Palmer Penguins data (344 rows: species, island, bill/flipper measurements, body mass, sex, year) |
| `requirements.txt` | Dependencies for Linux, Windows, and macOS |

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip
```

Works on Linux, Windows, and macOS (Apple Silicon). The extra package index
in `requirements.txt` gives Windows a CUDA-enabled PyTorch build (PyPI's
Windows wheels are CPU-only); Linux gets CUDA from PyPI directly and macOS
gets the MPS (Metal) build.

The first run downloads the pretrained weights (~13 GB) from Hugging Face
into `~/.cache/huggingface`; later runs use the cache.

## Usage

```bash
python predict_species.py N_REPEATS TEST_PERCENT [--n-estimators N] [--all-stats]
```

- `N_REPEATS` — how many times to randomly select test rows
- `TEST_PERCENT` — percent of rows held out as the test set each time
- `--n-estimators N` — TabFM ensemble size (default: 8); higher is slower
  but can be slightly more accurate
- `--all-stats` — print every statistic pycm computes; without it the
  report shows the subset matching R caret's `confusionMatrix()` output

Example:

```bash
python predict_species.py 5 20
```

Each repeat draws a fresh random split, fits TabFM on the training rows, and
predicts the held-out rows. The script prints per-run accuracy, a summary
(mean/std/min/max) across runs, and a confusion-matrix report computed by
[pycm](https://www.pycm.io/) on the predictions pooled across all runs —
by default the statistics R caret's `confusionMatrix()` reports (accuracy,
95% CI, NIR, Kappa, and per-class sensitivity/specificity/predictive
values), or everything pycm offers with `--all-stats`:

```
Run 1/5: train=275 test=69 accuracy=0.9855 (19.7s)
...
Accuracy over 5 run(s) with 20% test rows: mean=0.9928 std=0.0072 ...

Predict         Adelie          Chinstrap       Gentoo
Actual
Adelie          63              0               1
...
Overall ACC     0.99275
Kappa           0.98853
...
```

The script auto-selects the fastest device: NVIDIA GPU (CUDA) → Apple GPU
(MPS) → CPU. On CPU expect roughly 20 s per repeat at the default ensemble
size; a GPU is much faster.

## Using TabFM in your own code

TabFM exposes a scikit-learn style API. The minimal recipe:

```python
from tabfm import TabFMClassifier
from tabfm.src.pytorch import tabfm_v1_0_0

# Downloads pretrained weights on first use; device can be "cuda", "mps", or "cpu"
model = tabfm_v1_0_0.load("classification", device="cpu")

clf = TabFMClassifier(model=model)
clf.fit(X_train, y_train)        # stores context — instant, no training
preds = clf.predict(X_test)      # in-context inference happens here
probas = clf.predict_proba(X_test)
```

Things worth knowing:

- **Input**: pandas DataFrames with mixed types work directly. Categorical
  string columns, datetime-looking text, and missing values (NaN) are all
  handled by TabFM's built-in preprocessing — no manual encoding needed.
- **Ensembling**: `n_estimators` controls how many differently-preprocessed
  views of the data get averaged (the package default is 32; this script
  defaults to 8 via `--n-estimators`). Higher is slower but can be slightly
  more accurate.
- **Limits**: the pretrained classifier supports up to 10 classes and 500
  features by default; very large training sets get subsampled into context.
- **Regression**: use `TabFMRegressor` with
  `tabfm_v1_0_0.load("regression")` for numeric targets.
- **Checkpoint quirk**: tabfm 1.0.0's loader looks for `pytorch_model.bin`,
  but the current Hugging Face repo ships `model.safetensors`.
  `predict_species.py` includes a fallback loader (`load_model()`) that
  handles this — copy it if you hit `FileNotFoundError` in your own code.

## Data

Palmer Penguins: Gorman KB, Williams TD, Fraser WR (2014), collected at
Palmer Station, Antarctica. Three species (Adelie, Chinstrap, Gentoo) with
bill, flipper, and body-mass measurements plus island, sex, and year.

## License

MIT — see [LICENSE](LICENSE).
