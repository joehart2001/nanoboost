# NanoBoost

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17517843.svg)](https://doi.org/10.5281/zenodo.17517843)

**A Wavelet Transform-Enhanced Machine Learning Algorithm for Next-Generation Nanopore Multiplexing**

NanoBoost is a machine learning pipeline combining XGBoost with a novel event-specific Discrete Wavelet Transform (DWT) for nanopore translocation event classification. It achieved 91% accuracy in nanoparticle size classification (5 nm vs 10 nm nanospheres) and 99% in shape classification (nanospheres vs nanorods).

## Installation

```bash
git clone https://github.com/joehart2001/nanoboost.git
cd nanoboost
pip install -e .
```

For unsupervised learning and plotting dependencies:

```bash
pip install -e ".[full]"
```

## Usage

### End-to-end run

Edit `config.yaml` to point to your ABF files and set labels, then:

```bash
nanoboost run config.yaml
```

### Step-by-step

```bash
# 1. Isolate events from a raw ABF recording
nanoboost preprocess recording.abf events.pkl

# 2. Apply per-event DWT and extract features  (label: 5 or 10 for nm, NR or NS for shape)
nanoboost transform events.pkl features.pkl 10

# 3. Train a classifier across one or more feature sets
nanoboost train features_5nm.pkl features_10nm.pkl --output model.pkl
```

Use `nanoboost <command> --help` for full options on any command.

## Configuration

`config.yaml` controls all parameters. Paper-optimal defaults are pre-set:

```yaml
data:
  files:
    - path: recordings/5nm.abf
      label: 5
    - path: recordings/10nm.abf
      label: 10
  output_dir: results/

preprocessing:
  resistive: false      # true for biphasic events with a resistive (trough) component

transform:
  wavelet: bior3.3      # paper-optimal mother wavelet
  threshold: 0.2        # paper-optimal DWT coefficient threshold

training:
  model: xgboost        # xgboost | rf | svm | dt
  search: random        # random | bayes | sobol | grid
```

CLI flags override config values for individual commands:

```bash
nanoboost transform events.pkl features.pkl 10 --wavelet haar --threshold 0.3
nanoboost train features.pkl --model rf --search bayes
```

## Citation

If you use this code or methodology in your research, please cite this repository (paper details to be added upon publication). Here is a suggested BibTeX entry:

```bibtex
@software{nanoboost,
  author = {Hart, Joseph},
  title = {NanoBoost: A Wavelet Transform-Enhanced Machine Learning Algorithm for Next-Generation Nanopore Multiplexing},
  year = {2025},
  url = {https://github.com/joehart2001/nanoboost}
}
```

For the complete citation information, see [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
